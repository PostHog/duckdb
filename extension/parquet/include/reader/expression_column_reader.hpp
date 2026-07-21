//===----------------------------------------------------------------------===//
//                         DuckDB
//
// reader/expression_column_reader.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "column_reader.hpp"
#include "reader/templated_column_reader.hpp"

namespace duckdb {

//! A column reader that executes an expression over its child reader(s)
class ExpressionColumnReader : public ColumnReader {
public:
	static constexpr const PhysicalType TYPE = PhysicalType::INVALID;

public:
	ExpressionColumnReader(ClientContext &context, vector<unique_ptr<ColumnReader>> child_readers,
	                       unique_ptr<Expression> expr, const ParquetColumnSchema &schema);
	ExpressionColumnReader(ClientContext &context, vector<unique_ptr<ColumnReader>> child_readers,
	                       unique_ptr<Expression> expr, unique_ptr<ParquetColumnSchema> owned_schema);

	//! Reader(s) to produce the input(s) for the expression
	vector<unique_ptr<ColumnReader>> child_readers;
	unique_ptr<Expression> expr;
	ExpressionExecutor executor;
	DataChunk intermediate_chunk;
	// If this reader was created on top of a child reader, after-the-fact, the schema needs to live somewhere
	unique_ptr<ParquetColumnSchema> owned_schema;

public:
	void InitializeRead(idx_t row_group_idx_p, const vector<ColumnChunk> &columns, TProtocol &protocol_p) override;

	idx_t Read(uint64_t num_values, data_ptr_t define_out, data_ptr_t repeat_out, Vector &result) override;

	void Select(uint64_t num_values, data_ptr_t define_out, data_ptr_t repeat_out, Vector &result_out,
	            const SelectionVector &sel, idx_t approved_tuple_count) override;

	void Skip(idx_t num_values) override;
	uint64_t TotalCompressedSize() override {
		idx_t total_compressed_size = 0;
		for (auto &child_reader : child_readers) {
			total_compressed_size += child_reader->TotalCompressedSize();
		}
		return total_compressed_size;
	}
	void GetScannedColumnSizes(unordered_map<idx_t, uint64_t> &result) override {
		for (auto &child_reader : child_readers) {
			child_reader->GetScannedColumnSizes(result);
		}
	}
	idx_t FileOffset() const override {
		return child_readers[0]->FileOffset();
	}
	void RegisterPrefetch(ThriftFileTransport &transport, bool allow_merge) override {
		for (auto &child_reader : child_readers) {
			child_reader->RegisterPrefetch(transport, allow_merge);
		}
	}
	uint64_t GroupRowsAvailable() override;

private:
	void InitializeChunk();
};

} // namespace duckdb
