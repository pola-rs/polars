use parquet_format_safe::RowGroup;
#[cfg(feature = "serde_types")]
use serde::{Deserialize, Serialize};

use super::column_chunk_metadata::ColumnChunkMetaData;
use super::schema_descriptor::SchemaDescriptor;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::write::ColumnOffsetsMetadata;

/// Metadata for a row group.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub struct RowGroupMetaData {
    columns: Vec<ColumnChunkMetaData>,
    num_rows: usize,
    total_byte_size: usize,
}

impl RowGroupMetaData {
    /// Create a new [`RowGroupMetaData`]
    pub fn new(
        columns: Vec<ColumnChunkMetaData>,
        num_rows: usize,
        total_byte_size: usize,
    ) -> RowGroupMetaData {
        Self {
            columns,
            num_rows,
            total_byte_size,
        }
    }

    /// Returns slice of column chunk metadata.
    pub fn columns(&self) -> &[ColumnChunkMetaData] {
        &self.columns
    }

    /// Number of rows in this row group.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Total byte size of all uncompressed column data in this row group.
    pub fn total_byte_size(&self) -> usize {
        self.total_byte_size
    }

    /// Total size of all compressed column data in this row group.
    pub fn compressed_size(&self) -> usize {
        self.columns
            .iter()
            .map(|c| c.compressed_size() as usize)
            .sum::<usize>()
    }

    /// Method to convert from Thrift.
    pub(crate) fn try_from_thrift(
        schema_descr: &SchemaDescriptor,
        rg: RowGroup,
    ) -> ParquetResult<RowGroupMetaData> {
        if schema_descr.columns().len() != rg.columns.len() {
            return Err(ParquetError::oos(format!("The number of columns in the row group ({}) must be equal to the number of columns in the schema ({})", rg.columns.len(), schema_descr.columns().len())));
        }
        let total_byte_size = rg.total_byte_size.try_into()?;
        let num_rows = rg.num_rows.try_into()?;
        let columns = rg
            .columns
            .into_iter()
            .zip(schema_descr.columns())
            .map(|(column_chunk, descriptor)| {
                ColumnChunkMetaData::try_from_thrift(descriptor.clone(), column_chunk)
            })
            .collect::<ParquetResult<Vec<_>>>()?;

        Ok(RowGroupMetaData {
            columns,
            num_rows,
            total_byte_size,
        })
    }

    /// Method to convert to Thrift.
    pub(crate) fn into_thrift(self) -> RowGroup {
        let file_offset = self
            .columns
            .iter()
            .map(|c| {
                ColumnOffsetsMetadata::from_column_chunk_metadata(c).calc_row_group_file_offset()
            })
            .next()
            .unwrap_or(None);
        let total_compressed_size = Some(self.compressed_size() as i64);
        RowGroup {
            columns: self.columns.into_iter().map(|v| v.into_thrift()).collect(),
            total_byte_size: self.total_byte_size as i64,
            num_rows: self.num_rows as i64,
            sorting_columns: None,
            file_offset,
            total_compressed_size,
            ordinal: None,
        }
    }
}
