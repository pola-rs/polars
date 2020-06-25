// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Contains information about available Parquet metadata.
//!
//! The hierarchy of metadata is as follows:
//!
//! [`ParquetMetaData`](struct.ParquetMetaData.html) contains
//! [`FileMetaData`](struct.FileMetaData.html) and zero or more
//! [`RowGroupMetaData`](struct.RowGroupMetaData.html) for each row group.
//!
//! [`FileMetaData`](struct.FileMetaData.html) includes file version, application specific
//! metadata.
//!
//! Each [`RowGroupMetaData`](struct.RowGroupMetaData.html) contains information about row
//! group and one or more [`ColumnChunkMetaData`](struct.ColumnChunkMetaData.html) for
//! each column chunk.
//!
//! [`ColumnChunkMetaData`](struct.ColumnChunkMetaData.html) has information about column
//! chunk (primitive leaf column), including encoding/compression, number of values, etc.

use std::rc::Rc;

use parquet_format::{ColumnChunk, ColumnMetaData, RowGroup};

use crate::basic::{ColumnOrder, Compression, Encoding, Type};
use crate::errors::{ParquetError, Result};
use crate::file::statistics::{self, Statistics};
use crate::schema::types::{
    ColumnDescPtr, ColumnDescriptor, ColumnPath, SchemaDescPtr, SchemaDescriptor,
    Type as SchemaType, TypePtr,
};

/// Global Parquet metadata.
pub struct ParquetMetaData {
    file_metadata: FileMetaData,
    row_groups: Vec<RowGroupMetaData>,
}

impl ParquetMetaData {
    /// Creates Parquet metadata from file metadata and a list of row group metadata `Rc`s
    /// for each available row group.
    pub fn new(file_metadata: FileMetaData, row_groups: Vec<RowGroupMetaData>) -> Self {
        ParquetMetaData {
            file_metadata,
            row_groups,
        }
    }

    /// Returns file metadata as reference.
    pub fn file_metadata(&self) -> &FileMetaData {
        &self.file_metadata
    }

    /// Returns number of row groups in this file.
    pub fn num_row_groups(&self) -> usize {
        self.row_groups.len()
    }

    /// Returns row group metadata for `i`th position.
    /// Position should be less than number of row groups `num_row_groups`.
    pub fn row_group(&self, i: usize) -> &RowGroupMetaData {
        &self.row_groups[i]
    }

    /// Returns slice of row groups in this file.
    pub fn row_groups(&self) -> &[RowGroupMetaData] {
        &self.row_groups
    }
}

pub type KeyValue = parquet_format::KeyValue;

/// Reference counted pointer for [`FileMetaData`].
pub type FileMetaDataPtr = Rc<FileMetaData>;

/// Metadata for a Parquet file.
pub struct FileMetaData {
    version: i32,
    num_rows: i64,
    created_by: Option<String>,
    key_value_metadata: Option<Vec<KeyValue>>,
    schema: TypePtr,
    schema_descr: SchemaDescPtr,
    column_orders: Option<Vec<ColumnOrder>>,
}

impl FileMetaData {
    /// Creates new file metadata.
    pub fn new(
        version: i32,
        num_rows: i64,
        created_by: Option<String>,
        key_value_metadata: Option<Vec<KeyValue>>,
        schema: TypePtr,
        schema_descr: SchemaDescPtr,
        column_orders: Option<Vec<ColumnOrder>>,
    ) -> Self {
        FileMetaData {
            version,
            num_rows,
            created_by,
            key_value_metadata,
            schema,
            schema_descr,
            column_orders,
        }
    }

    /// Returns version of this file.
    pub fn version(&self) -> i32 {
        self.version
    }

    /// Returns number of rows in the file.
    pub fn num_rows(&self) -> i64 {
        self.num_rows
    }

    /// String message for application that wrote this file.
    ///
    /// This should have the following format:
    /// `<application> version <application version> (build <application build hash>)`.
    ///
    /// ```shell
    /// parquet-mr version 1.8.0 (build 0fda28af84b9746396014ad6a415b90592a98b3b)
    /// ```
    pub fn created_by(&self) -> &Option<String> {
        &self.created_by
    }

    /// Returns key_value_metadata of this file.
    pub fn key_value_metadata(&self) -> &Option<Vec<KeyValue>> {
        &self.key_value_metadata
    }

    /// Returns Parquet ['Type`] that describes schema in this file.
    pub fn schema(&self) -> &SchemaType {
        self.schema.as_ref()
    }

    /// Returns a reference to schema descriptor.
    pub fn schema_descr(&self) -> &SchemaDescriptor {
        &self.schema_descr
    }

    /// Returns reference counted clone for schema descriptor.
    pub fn schema_descr_ptr(&self) -> SchemaDescPtr {
        self.schema_descr.clone()
    }

    /// Column (sort) order used for `min` and `max` values of each column in this file.
    ///
    /// Each column order corresponds to one column, determined by its position in the
    /// list, matching the position of the column in the schema.
    ///
    /// When `None` is returned, there are no column orders available, and each column
    /// should be assumed to have undefined (legacy) column order.
    pub fn column_orders(&self) -> Option<&Vec<ColumnOrder>> {
        self.column_orders.as_ref()
    }

    /// Returns column order for `i`th column in this file.
    /// If column orders are not available, returns undefined (legacy) column order.
    pub fn column_order(&self, i: usize) -> ColumnOrder {
        self.column_orders
            .as_ref()
            .map(|data| data[i])
            .unwrap_or(ColumnOrder::UNDEFINED)
    }
}

/// Reference counted pointer for [`RowGroupMetaData`].
pub type RowGroupMetaDataPtr = Rc<RowGroupMetaData>;

/// Metadata for a row group.
pub struct RowGroupMetaData {
    columns: Vec<ColumnChunkMetaData>,
    num_rows: i64,
    total_byte_size: i64,
    schema_descr: SchemaDescPtr,
}

impl RowGroupMetaData {
    /// Returns builer for row group metadata.
    pub fn builder(schema_descr: SchemaDescPtr) -> RowGroupMetaDataBuilder {
        RowGroupMetaDataBuilder::new(schema_descr)
    }

    /// Number of columns in this row group.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Returns column chunk metadata for `i`th column.
    pub fn column(&self, i: usize) -> &ColumnChunkMetaData {
        &self.columns[i]
    }

    /// Returns slice of column chunk metadata.
    pub fn columns(&self) -> &[ColumnChunkMetaData] {
        &self.columns
    }

    /// Number of rows in this row group.
    pub fn num_rows(&self) -> i64 {
        self.num_rows
    }

    /// Total byte size of all uncompressed column data in this row group.
    pub fn total_byte_size(&self) -> i64 {
        self.total_byte_size
    }

    /// Returns reference to a schema descriptor.
    pub fn schema_descr(&self) -> &SchemaDescriptor {
        self.schema_descr.as_ref()
    }

    /// Returns reference counted clone of schema descriptor.
    pub fn schema_descr_ptr(&self) -> SchemaDescPtr {
        self.schema_descr.clone()
    }

    /// Method to convert from Thrift.
    pub fn from_thrift(
        schema_descr: SchemaDescPtr,
        mut rg: RowGroup,
    ) -> Result<RowGroupMetaData> {
        assert_eq!(schema_descr.num_columns(), rg.columns.len());
        let total_byte_size = rg.total_byte_size;
        let num_rows = rg.num_rows;
        let mut columns = vec![];
        for (c, d) in rg.columns.drain(0..).zip(schema_descr.columns()) {
            let cc = ColumnChunkMetaData::from_thrift(d.clone(), c)?;
            columns.push(cc);
        }
        Ok(RowGroupMetaData {
            columns,
            num_rows,
            total_byte_size,
            schema_descr,
        })
    }

    /// Method to convert to Thrift.
    pub fn to_thrift(&self) -> RowGroup {
        RowGroup {
            columns: self.columns().into_iter().map(|v| v.to_thrift()).collect(),
            total_byte_size: self.total_byte_size,
            num_rows: self.num_rows,
            sorting_columns: None,
        }
    }
}

/// Builder for row group metadata.
pub struct RowGroupMetaDataBuilder {
    columns: Vec<ColumnChunkMetaData>,
    schema_descr: SchemaDescPtr,
    num_rows: i64,
    total_byte_size: i64,
}

impl RowGroupMetaDataBuilder {
    /// Creates new builder from schema descriptor.
    fn new(schema_descr: SchemaDescPtr) -> Self {
        Self {
            columns: Vec::with_capacity(schema_descr.num_columns()),
            schema_descr,
            num_rows: 0,
            total_byte_size: 0,
        }
    }

    /// Sets number of rows in this row group.
    pub fn set_num_rows(mut self, value: i64) -> Self {
        self.num_rows = value;
        self
    }

    /// Sets total size in bytes for this row group.
    pub fn set_total_byte_size(mut self, value: i64) -> Self {
        self.total_byte_size = value;
        self
    }

    /// Sets column metadata for this row group.
    pub fn set_column_metadata(mut self, value: Vec<ColumnChunkMetaData>) -> Self {
        self.columns = value;
        self
    }

    /// Builds row group metadata.
    pub fn build(self) -> Result<RowGroupMetaData> {
        if self.schema_descr.num_columns() != self.columns.len() {
            return Err(general_err!(
                "Column length mismatch: {} != {}",
                self.schema_descr.num_columns(),
                self.columns.len()
            ));
        }

        Ok(RowGroupMetaData {
            columns: self.columns,
            num_rows: self.num_rows,
            total_byte_size: self.total_byte_size,
            schema_descr: self.schema_descr,
        })
    }
}

/// Metadata for a column chunk.
pub struct ColumnChunkMetaData {
    column_type: Type,
    column_path: ColumnPath,
    column_descr: ColumnDescPtr,
    encodings: Vec<Encoding>,
    file_path: Option<String>,
    file_offset: i64,
    num_values: i64,
    compression: Compression,
    total_compressed_size: i64,
    total_uncompressed_size: i64,
    data_page_offset: i64,
    index_page_offset: Option<i64>,
    dictionary_page_offset: Option<i64>,
    statistics: Option<Statistics>,
}

/// Represents common operations for a column chunk.
impl ColumnChunkMetaData {
    /// Returns builder for column chunk metadata.
    pub fn builder(column_descr: ColumnDescPtr) -> ColumnChunkMetaDataBuilder {
        ColumnChunkMetaDataBuilder::new(column_descr)
    }

    /// File where the column chunk is stored.
    ///
    /// If not set, assumed to belong to the same file as the metadata.
    /// This path is relative to the current file.
    pub fn file_path(&self) -> Option<&String> {
        self.file_path.as_ref()
    }

    /// Byte offset in `file_path()`.
    pub fn file_offset(&self) -> i64 {
        self.file_offset
    }

    /// Type of this column. Must be primitive.
    pub fn column_type(&self) -> Type {
        self.column_type
    }

    /// Path (or identifier) of this column.
    pub fn column_path(&self) -> &ColumnPath {
        &self.column_path
    }

    /// Descriptor for this column.
    pub fn column_descr(&self) -> &ColumnDescriptor {
        self.column_descr.as_ref()
    }

    /// Reference counted clone of descriptor for this column.
    pub fn column_descr_ptr(&self) -> ColumnDescPtr {
        self.column_descr.clone()
    }

    /// All encodings used for this column.
    pub fn encodings(&self) -> &Vec<Encoding> {
        &self.encodings
    }

    /// Total number of values in this column chunk.
    pub fn num_values(&self) -> i64 {
        self.num_values
    }

    /// Compression for this column.
    pub fn compression(&self) -> Compression {
        self.compression
    }

    /// Returns the total compressed data size of this column chunk.
    pub fn compressed_size(&self) -> i64 {
        self.total_compressed_size
    }

    /// Returns the total uncompressed data size of this column chunk.
    pub fn uncompressed_size(&self) -> i64 {
        self.total_uncompressed_size
    }

    /// Returns the offset for the column data.
    pub fn data_page_offset(&self) -> i64 {
        self.data_page_offset
    }

    /// Returns `true` if this column chunk contains a index page, `false` otherwise.
    pub fn has_index_page(&self) -> bool {
        self.index_page_offset.is_some()
    }

    /// Returns the offset for the index page.
    pub fn index_page_offset(&self) -> Option<i64> {
        self.index_page_offset
    }

    /// Returns `true` if this column chunk contains a dictionary page, `false` otherwise.
    pub fn has_dictionary_page(&self) -> bool {
        self.dictionary_page_offset.is_some()
    }

    /// Returns the offset for the dictionary page, if any.
    pub fn dictionary_page_offset(&self) -> Option<i64> {
        self.dictionary_page_offset
    }

    /// Returns statistics that are set for this column chunk,
    /// or `None` if no statistics are available.
    pub fn statistics(&self) -> Option<&Statistics> {
        self.statistics.as_ref()
    }

    /// Method to convert from Thrift.
    pub fn from_thrift(column_descr: ColumnDescPtr, cc: ColumnChunk) -> Result<Self> {
        if cc.meta_data.is_none() {
            return Err(general_err!("Expected to have column metadata"));
        }
        let mut col_metadata: ColumnMetaData = cc.meta_data.unwrap();
        let column_type = Type::from(col_metadata.type_);
        let column_path = ColumnPath::new(col_metadata.path_in_schema);
        let encodings = col_metadata
            .encodings
            .drain(0..)
            .map(Encoding::from)
            .collect();
        let compression = Compression::from(col_metadata.codec);
        let file_path = cc.file_path;
        let file_offset = cc.file_offset;
        let num_values = col_metadata.num_values;
        let total_compressed_size = col_metadata.total_compressed_size;
        let total_uncompressed_size = col_metadata.total_uncompressed_size;
        let data_page_offset = col_metadata.data_page_offset;
        let index_page_offset = col_metadata.index_page_offset;
        let dictionary_page_offset = col_metadata.dictionary_page_offset;
        let statistics = statistics::from_thrift(column_type, col_metadata.statistics);
        let result = ColumnChunkMetaData {
            column_type,
            column_path,
            column_descr,
            encodings,
            file_path,
            file_offset,
            num_values,
            compression,
            total_compressed_size,
            total_uncompressed_size,
            data_page_offset,
            index_page_offset,
            dictionary_page_offset,
            statistics,
        };
        Ok(result)
    }

    /// Method to convert to Thrift.
    pub fn to_thrift(&self) -> ColumnChunk {
        let column_metadata = ColumnMetaData {
            type_: self.column_type.into(),
            encodings: self.encodings().into_iter().map(|&v| v.into()).collect(),
            path_in_schema: Vec::from(self.column_path.as_ref()),
            codec: self.compression.into(),
            num_values: self.num_values,
            total_uncompressed_size: self.total_uncompressed_size,
            total_compressed_size: self.total_compressed_size,
            key_value_metadata: None,
            data_page_offset: self.data_page_offset,
            index_page_offset: self.index_page_offset,
            dictionary_page_offset: self.dictionary_page_offset,
            statistics: statistics::to_thrift(self.statistics.as_ref()),
            encoding_stats: None,
        };

        ColumnChunk {
            file_path: self.file_path().map(|v| v.clone()),
            file_offset: self.file_offset,
            meta_data: Some(column_metadata),
            offset_index_offset: None,
            offset_index_length: None,
            column_index_offset: None,
            column_index_length: None,
        }
    }
}

/// Builder for column chunk metadata.
pub struct ColumnChunkMetaDataBuilder {
    column_descr: ColumnDescPtr,
    encodings: Vec<Encoding>,
    file_path: Option<String>,
    file_offset: i64,
    num_values: i64,
    compression: Compression,
    total_compressed_size: i64,
    total_uncompressed_size: i64,
    data_page_offset: i64,
    index_page_offset: Option<i64>,
    dictionary_page_offset: Option<i64>,
    statistics: Option<Statistics>,
}

impl ColumnChunkMetaDataBuilder {
    /// Creates new column chunk metadata builder.
    fn new(column_descr: ColumnDescPtr) -> Self {
        Self {
            column_descr,
            encodings: Vec::new(),
            file_path: None,
            file_offset: 0,
            num_values: 0,
            compression: Compression::UNCOMPRESSED,
            total_compressed_size: 0,
            total_uncompressed_size: 0,
            data_page_offset: 0,
            index_page_offset: None,
            dictionary_page_offset: None,
            statistics: None,
        }
    }

    /// Sets list of encodings for this column chunk.
    pub fn set_encodings(mut self, encodings: Vec<Encoding>) -> Self {
        self.encodings = encodings;
        self
    }

    /// Sets optional file path for this column chunk.
    pub fn set_file_path(mut self, value: String) -> Self {
        self.file_path = Some(value);
        self
    }

    /// Sets file offset in bytes.
    pub fn set_file_offset(mut self, value: i64) -> Self {
        self.file_offset = value;
        self
    }

    /// Sets number of values.
    pub fn set_num_values(mut self, value: i64) -> Self {
        self.num_values = value;
        self
    }

    /// Sets compression.
    pub fn set_compression(mut self, value: Compression) -> Self {
        self.compression = value;
        self
    }

    /// Sets total compressed size in bytes.
    pub fn set_total_compressed_size(mut self, value: i64) -> Self {
        self.total_compressed_size = value;
        self
    }

    /// Sets total uncompressed size in bytes.
    pub fn set_total_uncompressed_size(mut self, value: i64) -> Self {
        self.total_uncompressed_size = value;
        self
    }

    /// Sets data page offset in bytes.
    pub fn set_data_page_offset(mut self, value: i64) -> Self {
        self.data_page_offset = value;
        self
    }

    /// Sets optional dictionary page ofset in bytes.
    pub fn set_dictionary_page_offset(mut self, value: Option<i64>) -> Self {
        self.dictionary_page_offset = value;
        self
    }

    /// Sets optional index page offset in bytes.
    pub fn set_index_page_offset(mut self, value: Option<i64>) -> Self {
        self.index_page_offset = value;
        self
    }

    /// Sets statistics for this column chunk.
    pub fn set_statistics(mut self, value: Statistics) -> Self {
        self.statistics = Some(value);
        self
    }

    /// Builds column chunk metadata.
    pub fn build(self) -> Result<ColumnChunkMetaData> {
        Ok(ColumnChunkMetaData {
            column_type: self.column_descr.physical_type(),
            column_path: self.column_descr.path().clone(),
            column_descr: self.column_descr,
            encodings: self.encodings,
            file_path: self.file_path,
            file_offset: self.file_offset,
            num_values: self.num_values,
            compression: self.compression,
            total_compressed_size: self.total_compressed_size,
            total_uncompressed_size: self.total_uncompressed_size,
            data_page_offset: self.data_page_offset,
            index_page_offset: self.index_page_offset,
            dictionary_page_offset: self.dictionary_page_offset,
            statistics: self.statistics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_group_metadata_thrift_conversion() {
        let schema_descr = get_test_schema_descr();

        let mut columns = vec![];
        for ptr in schema_descr.columns() {
            let column = ColumnChunkMetaData::builder(ptr.clone()).build().unwrap();
            columns.push(column);
        }
        let row_group_meta = RowGroupMetaData::builder(schema_descr.clone())
            .set_num_rows(1000)
            .set_total_byte_size(2000)
            .set_column_metadata(columns)
            .build()
            .unwrap();

        let row_group_exp = row_group_meta.to_thrift();
        let row_group_res =
            RowGroupMetaData::from_thrift(schema_descr.clone(), row_group_exp.clone())
                .unwrap()
                .to_thrift();

        assert_eq!(row_group_res, row_group_exp);
    }

    #[test]
    fn test_row_group_metadata_thrift_conversion_empty() {
        let schema_descr = get_test_schema_descr();

        let row_group_meta = RowGroupMetaData::builder(schema_descr.clone()).build();

        assert!(row_group_meta.is_err());
        if let Err(e) = row_group_meta {
            assert_eq!(
                format!("{}", e),
                "Parquet error: Column length mismatch: 2 != 0"
            );
        }
    }

    #[test]
    fn test_column_chunk_metadata_thrift_conversion() {
        let column_descr = get_test_schema_descr().column(0);

        let col_metadata = ColumnChunkMetaData::builder(column_descr.clone())
            .set_encodings(vec![Encoding::PLAIN, Encoding::RLE])
            .set_file_path("file_path".to_owned())
            .set_file_offset(100)
            .set_num_values(1000)
            .set_compression(Compression::SNAPPY)
            .set_total_compressed_size(2000)
            .set_total_uncompressed_size(3000)
            .set_data_page_offset(4000)
            .set_dictionary_page_offset(Some(5000))
            .build()
            .unwrap();

        let col_chunk_exp = col_metadata.to_thrift();

        let col_chunk_res =
            ColumnChunkMetaData::from_thrift(column_descr.clone(), col_chunk_exp.clone())
                .unwrap()
                .to_thrift();

        assert_eq!(col_chunk_res, col_chunk_exp);
    }

    #[test]
    fn test_column_chunk_metadata_thrift_conversion_empty() {
        let column_descr = get_test_schema_descr().column(0);

        let col_metadata = ColumnChunkMetaData::builder(column_descr.clone())
            .build()
            .unwrap();

        let col_chunk_exp = col_metadata.to_thrift();
        let col_chunk_res =
            ColumnChunkMetaData::from_thrift(column_descr.clone(), col_chunk_exp.clone())
                .unwrap()
                .to_thrift();

        assert_eq!(col_chunk_res, col_chunk_exp);
    }

    /// Returns sample schema descriptor so we can create column metadata.
    fn get_test_schema_descr() -> SchemaDescPtr {
        let schema = SchemaType::group_type_builder("schema")
            .with_fields(&mut vec![
                Rc::new(
                    SchemaType::primitive_type_builder("a", Type::INT32)
                        .build()
                        .unwrap(),
                ),
                Rc::new(
                    SchemaType::primitive_type_builder("b", Type::INT32)
                        .build()
                        .unwrap(),
                ),
            ])
            .build()
            .unwrap();

        Rc::new(SchemaDescriptor::new(Rc::new(schema)))
    }
}
