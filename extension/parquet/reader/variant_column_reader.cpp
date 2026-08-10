#include "reader/variant_column_reader.hpp"
#include "reader/variant/variant_binary_decoder.hpp"
#include "reader/variant/variant_shredded_conversion.hpp"
#include "duckdb/function/scalar/variant_utils.hpp"

namespace duckdb {

static vector<VariantPathComponent> GetVariantExtractPath(const ColumnIndex &index) {
	vector<VariantPathComponent> result;
	if (!index.IsPushdownExtract()) {
		return result;
	}
	reference<const ColumnIndex> current(index.GetChildIndex(0));
	while (true) {
		if (current.get().HasPrimaryIndex()) {
			throw InternalException("VARIANT pushdown extract expected a field name path");
		}
		result.emplace_back(current.get().GetFieldName());
		if (!current.get().HasChildren()) {
			break;
		}
		current = current.get().GetChildIndex(0);
	}
	return result;
}

//===--------------------------------------------------------------------===//
// Variant Column Reader
//===--------------------------------------------------------------------===//
VariantColumnReader::VariantColumnReader(ClientContext &context, const ParquetReader &reader,
                                         const ParquetColumnSchema &schema,
                                         vector<unique_ptr<ColumnReader>> child_readers_p,
                                         const struct ColumnIndex &index)
    : ColumnReader(reader, schema), context(context), index(index), extract_path(GetVariantExtractPath(index)),
      child_readers(std::move(child_readers_p)) {
	D_ASSERT(Type().InternalType() == PhysicalType::STRUCT);

	if (child_readers[0]->Schema().name == "metadata" && child_readers[1]->Schema().name == "value") {
		metadata_reader_idx = 0;
		value_reader_idx = 1;
	} else if (child_readers[1]->Schema().name == "metadata" && child_readers[0]->Schema().name == "value") {
		metadata_reader_idx = 1;
		value_reader_idx = 0;
	} else {
		throw InternalException("The Variant column must have 'metadata' and 'value' as the first two columns");
	}
}

VariantColumnReader::VariantColumnReader(ClientContext &context, const ParquetReader &reader,
                                         const ParquetColumnSchema &schema,
                                         vector<unique_ptr<ColumnReader>> child_readers_p)
    : VariantColumnReader(context, reader, schema, std::move(child_readers_p), {}) {
}

ColumnReader &VariantColumnReader::GetChildReader(idx_t child_idx) {
	if (!child_readers[child_idx]) {
		throw InternalException("VariantColumnReader::GetChildReader(%d) - but this child reader is not set",
		                        child_idx);
	}
	return *child_readers[child_idx].get();
}

void VariantColumnReader::InitializeRead(idx_t row_group_idx_p, const vector<ColumnChunk> &columns,
                                         TProtocol &protocol_p) {
	for (auto &child : child_readers) {
		if (!child) {
			continue;
		}
		child->InitializeRead(row_group_idx_p, columns, protocol_p);
	}
}

unique_ptr<BaseStatistics> VariantColumnReader::Stats(idx_t row_group_idx_p, const vector<ColumnChunk> &columns) {
	auto result = ColumnReader::Stats(row_group_idx_p, columns);
	if (result && index.IsPushdownExtract()) {
		auto storage_index = StorageIndex::FromColumnIndex(index);
		return result->PushdownExtract(storage_index.GetChildIndexes()[0]);
	}
	return result;
}

static LogicalType GetIntermediateGroupType(optional_ptr<ColumnReader> typed_value) {
	child_list_t<LogicalType> children;
	children.emplace_back("value", LogicalType::BLOB);
	if (typed_value) {
		children.emplace_back("typed_value", typed_value->Type());
	}
	return LogicalType::STRUCT(std::move(children));
}


//===--------------------------------------------------------------------===//
// Targeted pushdown extract (1.5.5 port of posthog a9dcee6f96's navigation)
//
// For unshredded (2-child) variant columns, navigate the binary value along
// the pushed-down path and decode ONLY the addressed leaf, instead of
// converting every row's entire value (and re-decoding it in VariantExtract).
// The metadata dictionary is decoded once per distinct blob content — writers
// emit identical metadata per row — and key->field-id resolution is cached
// per dictionary.
//===--------------------------------------------------------------------===//

struct VariantColumnReader::VariantMetadataCacheEntry {
	explicit VariantMetadataCacheEntry(const string_t &blob) : metadata(blob) {
		for (idx_t id = 0; id < metadata.strings.size(); id++) {
			field_ids.emplace(metadata.strings[id], id);
		}
	}
	VariantMetadata metadata;
	//! key -> dictionary field id, resolved once per distinct dictionary
	unordered_map<string, idx_t> field_ids;
};

VariantColumnReader::VariantMetadataCacheEntry &VariantColumnReader::GetBinaryMetadata(const string_t &blob) {
	auto key = blob.GetString();
	auto entry = binary_metadata_cache.find(key);
	if (entry != binary_metadata_cache.end()) {
		return *entry->second;
	}
	//! Bound the cache — dictionaries are shared across a row group, so a handful of
	//! entries covers the working set; clear rather than grow unboundedly on pathological files.
	if (binary_metadata_cache.size() >= 1024) {
		binary_metadata_cache.clear();
	}
	auto res = binary_metadata_cache.emplace(key, make_shared_ptr<VariantMetadataCacheEntry>(blob));
	return *res.first->second;
}

namespace {

uint32_t VariantReadLE(uint8_t size, const_data_ptr_t data, idx_t offset, idx_t data_size) {
	if (offset + size > data_size) {
		throw IOException("Corrupted VARIANT 'value' buffer");
	}
	switch (size) {
	case 1:
		return Load<uint8_t>(data + offset);
	case 2:
		return Load<uint16_t>(data + offset);
	case 3: {
		uint32_t result = 0;
		memcpy(&result, data + offset, 3);
		return result;
	}
	case 4:
		return Load<uint32_t>(data + offset);
	default:
		throw IOException("Corrupted VARIANT field size (%d)", size);
	}
}

//! data_offset points at the value header of an OBJECT; on success it is advanced to the
//! value header of the child named 'key'. Returns false when the value is not an object or
//! the key is absent in this row.
bool NavigateVariantObject(const VariantColumnReader::VariantMetadataCacheEntry &cache, const_data_ptr_t data,
                           idx_t data_size, idx_t &data_offset, const string &key) {
	if (data_offset >= data_size) {
		throw IOException("Corrupted VARIANT 'value' buffer");
	}
	auto value_metadata = VariantValueMetadata::FromHeaderByte(data[data_offset]);
	if (value_metadata.basic_type != VariantBasicType::OBJECT) {
		return false;
	}
	data_offset++;

	idx_t num_elements;
	if (value_metadata.is_large) {
		num_elements = VariantReadLE(4, data, data_offset, data_size);
		data_offset += sizeof(uint32_t);
	} else {
		num_elements = VariantReadLE(1, data, data_offset, data_size);
		data_offset += sizeof(uint8_t);
	}

	auto wanted = cache.field_ids.find(key);
	if (wanted == cache.field_ids.end()) {
		return false; //! the dictionary has no such key — the row cannot contain it
	}

	auto field_ids_offset = data_offset;
	auto field_offsets_offset = field_ids_offset + (num_elements * value_metadata.field_id_size);
	auto values_offset = field_offsets_offset + ((num_elements + 1) * value_metadata.field_offset_size);

	for (idx_t i = 0; i < num_elements; i++) {
		auto field_id = VariantReadLE(value_metadata.field_id_size, data, field_ids_offset + (i * value_metadata.field_id_size), data_size);
		if (field_id != wanted->second) {
			continue;
		}
		auto value_offset = VariantReadLE(value_metadata.field_offset_size, data,
		                                  field_offsets_offset + (i * value_metadata.field_offset_size), data_size);
		data_offset = values_offset + value_offset;
		if (data_offset >= data_size) {
			throw IOException("Corrupted VARIANT 'value' buffer");
		}
		return true;
	}
	return false;
}

//! Same as NavigateVariantObject but for ARRAYs: advances data_offset to element 'index'.
bool NavigateVariantArray(const_data_ptr_t data, idx_t data_size, idx_t &data_offset, uint32_t index) {
	if (data_offset >= data_size) {
		throw IOException("Corrupted VARIANT 'value' buffer");
	}
	auto value_metadata = VariantValueMetadata::FromHeaderByte(data[data_offset]);
	if (value_metadata.basic_type != VariantBasicType::ARRAY) {
		return false;
	}
	data_offset++;

	idx_t num_elements;
	if (value_metadata.is_large) {
		num_elements = VariantReadLE(4, data, data_offset, data_size);
		data_offset += sizeof(uint32_t);
	} else {
		num_elements = VariantReadLE(1, data, data_offset, data_size);
		data_offset += sizeof(uint8_t);
	}
	if (index >= num_elements) {
		return false;
	}
	auto field_offsets_offset = data_offset;
	auto values_offset = field_offsets_offset + ((num_elements + 1) * value_metadata.field_offset_size);
	auto value_offset =
	    VariantReadLE(value_metadata.field_offset_size, data, field_offsets_offset + (index * value_metadata.field_offset_size), data_size);
	data_offset = values_offset + value_offset;
	if (data_offset >= data_size) {
		throw IOException("Corrupted VARIANT 'value' buffer");
	}
	return true;
}

} // anonymous namespace

vector<VariantValue> VariantColumnReader::NavigateBinaryExtract(Vector &metadata_col, Vector &value_col,
                                                                data_ptr_t define_out, idx_t num_values) {
	UnifiedVectorFormat metadata_format, value_format;
	metadata_col.ToUnifiedFormat(num_values, metadata_format);
	value_col.ToUnifiedFormat(num_values, value_format);
	auto metadata_strings = UnifiedVectorFormat::GetData<string_t>(metadata_format);
	auto value_strings = UnifiedVectorFormat::GetData<string_t>(value_format);

	vector<VariantValue> result;
	result.reserve(num_values);
	for (idx_t i = 0; i < num_values; i++) {
		if (define_out[i] < MaxDefine()) {
			//! NULL row
			result.emplace_back();
			continue;
		}
		auto midx = metadata_format.sel->get_index(i);
		auto vidx = value_format.sel->get_index(i);
		if (!metadata_format.validity.RowIsValid(midx) || !value_format.validity.RowIsValid(vidx)) {
			result.emplace_back();
			continue;
		}
		auto &value_blob = value_strings[vidx];
		auto data = const_data_ptr_cast(value_blob.GetData());
		auto data_size = value_blob.GetSize();

		idx_t data_offset = 0;
		bool found = true;
		for (auto &component : extract_path) {
			switch (component.lookup_mode) {
			case VariantChildLookupMode::BY_KEY:
				found = NavigateVariantObject(GetBinaryMetadata(metadata_strings[midx]), data, data_size, data_offset,
				                              component.key);
				break;
			case VariantChildLookupMode::BY_INDEX:
				found = NavigateVariantArray(data, data_size, data_offset, component.index);
				break;
			default:
				throw InternalException("Invalid variant path lookup mode in targeted extract");
			}
			if (!found) {
				break;
			}
		}
		if (!found) {
			result.emplace_back();
			continue;
		}
		auto &cache = GetBinaryMetadata(metadata_strings[midx]);
		result.push_back(VariantBinaryDecoder::Decode(cache.metadata, data, data_offset, data_size));
	}
	return result;
}

idx_t VariantColumnReader::Read(uint64_t num_values, data_ptr_t define_out, data_ptr_t repeat_out, Vector &result) {
	if (pending_skips > 0) {
		throw InternalException("VariantColumnReader cannot have pending skips");
	}
	optional_ptr<ColumnReader> typed_value_reader = child_readers.size() == 3 ? child_readers[2].get() : nullptr;

	// If the child reader values are all valid, "define_out" may not be initialized at all
	// So, we just initialize them to all be valid beforehand
	std::fill_n(define_out, num_values, MaxDefine());

	optional_idx read_count;

	Vector metadata_intermediate(LogicalType::BLOB, num_values);
	Vector intermediate_group(GetIntermediateGroupType(typed_value_reader), num_values);
	auto &group_entries = StructVector::GetEntries(intermediate_group);
	auto &value_intermediate = *group_entries[0];

	auto metadata_values =
	    child_readers[metadata_reader_idx]->Read(num_values, define_out, repeat_out, metadata_intermediate);
	auto value_values = child_readers[value_reader_idx]->Read(num_values, define_out, repeat_out, value_intermediate);

	D_ASSERT(child_readers[metadata_reader_idx]->Schema().name == "metadata");
	D_ASSERT(child_readers[value_reader_idx]->Schema().name == "value");

	if (metadata_values != value_values) {
		throw InvalidInputException(
		    "The Variant column did not contain the same amount of values for 'metadata' and 'value'");
	}

	vector<VariantValue> intermediate;
	if (typed_value_reader) {
		auto typed_values = typed_value_reader->Read(num_values, define_out, repeat_out, *group_entries[1]);
		if (typed_values != value_values) {
			throw InvalidInputException(
			    "The shredded Variant column did not contain the same amount of values for 'typed_value' and 'value'");
		}
	}
	if (index.IsPushdownExtract() && !typed_value_reader) {
		//! Unshredded column with a pushed-down extract: navigate the binary values directly,
		//! skipping the full Convert + ToVARIANT + re-decode round trip.
		D_ASSERT(!extract_path.empty());
		auto targeted = NavigateBinaryExtract(metadata_intermediate, value_intermediate, define_out, num_values);
		VariantValue::ToVARIANT(targeted, result);
		read_count = value_values;
		return read_count.GetIndex();
	}
	intermediate =
	    VariantShreddedConversion::Convert(metadata_intermediate, intermediate_group, 0, num_values, num_values);
	VariantValue::ToVARIANT(intermediate, result);
	if (index.IsPushdownExtract()) {
		D_ASSERT(!extract_path.empty());
		Vector extract_result(LogicalType::VARIANT(), num_values);
		VariantUtils::VariantExtract(result, extract_path, extract_result, num_values);
		result.Reference(extract_result);
	}

	read_count = value_values;
	return read_count.GetIndex();
}

void VariantColumnReader::Skip(idx_t num_values) {
	for (auto &child : child_readers) {
		if (!child) {
			continue;
		}
		child->Skip(num_values);
	}
}

void VariantColumnReader::RegisterPrefetch(ThriftFileTransport &transport, bool allow_merge) {
	for (auto &child : child_readers) {
		if (!child) {
			continue;
		}
		child->RegisterPrefetch(transport, allow_merge);
	}
}

uint64_t VariantColumnReader::TotalCompressedSize() {
	uint64_t size = 0;
	for (auto &child : child_readers) {
		if (!child) {
			continue;
		}
		size += child->TotalCompressedSize();
	}
	return size;
}

void VariantColumnReader::GetScannedColumnSizes(unordered_map<idx_t, uint64_t> &result) {
	for (auto &child : child_readers) {
		if (!child) {
			continue;
		}
		child->GetScannedColumnSizes(result);
	}
}

idx_t VariantColumnReader::GroupRowsAvailable() {
	for (auto &child : child_readers) {
		if (!child) {
			continue;
		}
		return child->GroupRowsAvailable();
	}
	throw InternalException("No projected columns in struct?");
}

bool VariantColumnReader::TypedValueLayoutToType(const LogicalType &typed_value, LogicalType &output) {
	if (!typed_value.IsNested()) {
		output = typed_value;
		return true;
	}
	auto type_id = typed_value.id();
	if (type_id == LogicalTypeId::STRUCT) {
		//! OBJECT (...)
		auto &object_fields = StructType::GetChildTypes(typed_value);
		child_list_t<LogicalType> children;
		for (auto &object_field : object_fields) {
			auto &name = object_field.first;
			auto &field = object_field.second;
			//! <name>: {
			//! 	value: BLOB,
			//! 	typed_value: <type>
			//! }
			auto &field_children = StructType::GetChildTypes(field);
			idx_t index = DConstants::INVALID_INDEX;
			for (idx_t i = 0; i < field_children.size(); i++) {
				if (field_children[i].first == "typed_value") {
					index = i;
					break;
				}
			}
			if (index == DConstants::INVALID_INDEX) {
				//! FIXME: we might be able to just omit this field from the OBJECT, instead of flat-out failing the
				//! conversion No 'typed_value' field, so we can't assign a structured type to this field at all
				return false;
			}
			LogicalType child_type;
			if (!TypedValueLayoutToType(field_children[index].second, child_type)) {
				return false;
			}
			children.emplace_back(name, child_type);
		}
		output = LogicalType::STRUCT(std::move(children));
		return true;
	}
	if (type_id == LogicalTypeId::LIST) {
		//! ARRAY
		auto &element = ListType::GetChildType(typed_value);
		//! element: {
		//! 	value: BLOB,
		//! 	typed_value: <type>
		//! }
		auto &element_children = StructType::GetChildTypes(element);
		idx_t index = DConstants::INVALID_INDEX;
		for (idx_t i = 0; i < element_children.size(); i++) {
			if (element_children[i].first == "typed_value") {
				index = i;
				break;
			}
		}
		if (index == DConstants::INVALID_INDEX) {
			//! This *might* be allowed by the spec, it's hard to reason about..
			return false;
		}
		LogicalType child_type;
		if (!TypedValueLayoutToType(element_children[index].second, child_type)) {
			return false;
		}
		output = LogicalType::LIST(child_type);
		return true;
	}
	throw InvalidInputException("VARIANT typed value has to be a primitive/struct/list, not %s",
	                            typed_value.ToString());
}

} // namespace duckdb
