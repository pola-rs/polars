use std::sync::Arc;

use hashbrown::hash_map::RawEntryMut;
use parquet_format_safe::RowGroup;
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::idx_vec::UnitVec;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unitvec;

use super::column_chunk_metadata::{column_metadata_byte_range, ColumnChunkMetadata};
use super::schema_descriptor::SchemaDescriptor;
use crate::parquet::error::{ParquetError, ParquetResult};

type ColumnLookup = PlHashMap<PlSmallStr, UnitVec<usize>>;

trait InitColumnLookup {
    fn add_column(&mut self, index: usize, column: &ColumnChunkMetadata);
}

impl InitColumnLookup for ColumnLookup {
    #[inline(always)]
    fn add_column(&mut self, index: usize, column: &ColumnChunkMetadata) {
        let root_name = &column.descriptor().path_in_schema[0];

        match self.raw_entry_mut().from_key(root_name) {
            RawEntryMut::Vacant(slot) => {
                slot.insert(root_name.clone(), unitvec![index]);
            },
            RawEntryMut::Occupied(mut slot) => {
                slot.get_mut().push(index);
            },
        };
    }
}

/// Metadata for a row group.
#[derive(Debug, Clone, Default)]
pub struct RowGroupMetadata {
    columns: Arc<[ColumnChunkMetadata]>,
    column_lookup: PlHashMap<PlSmallStr, UnitVec<usize>>,
    num_rows: usize,
    total_byte_size: usize,
    full_byte_range: core::ops::Range<u64>,
}

impl RowGroupMetadata {
    #[inline(always)]
    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    /// Fetch all columns under this root name if it exists.
    pub fn columns_under_root_iter(
        &self,
        root_name: &str,
    ) -> Option<impl ExactSizeIterator<Item = &ColumnChunkMetadata> + DoubleEndedIterator> {
        self.column_lookup
            .get(root_name)
            .map(|x| x.iter().map(|&x| &self.columns[x]))
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

    pub fn full_byte_range(&self) -> core::ops::Range<u64> {
        self.full_byte_range.clone()
    }

    pub fn byte_ranges_iter(&self) -> impl '_ + ExactSizeIterator<Item = core::ops::Range<u64>> {
        self.columns.iter().map(|x| x.byte_range())
    }

    /// Method to convert from Thrift.
    pub(crate) fn try_from_thrift(
        schema_descr: &SchemaDescriptor,
        rg: RowGroup,
    ) -> ParquetResult<RowGroupMetadata> {
        if schema_descr.columns().len() != rg.columns.len() {
            return Err(ParquetError::oos(format!("The number of columns in the row group ({}) must be equal to the number of columns in the schema ({})", rg.columns.len(), schema_descr.columns().len())));
        }
        let total_byte_size = rg.total_byte_size.try_into()?;
        let num_rows = rg.num_rows.try_into()?;

        let mut column_lookup = ColumnLookup::with_capacity(rg.columns.len());
        let mut full_byte_range = if let Some(first_column_chunk) = rg.columns.first() {
            let Some(metadata) = &first_column_chunk.meta_data else {
                return Err(ParquetError::oos("Column chunk requires metadata"));
            };
            column_metadata_byte_range(metadata)
        } else {
            0..0
        };

        let columns = rg
            .columns
            .into_iter()
            .zip(schema_descr.columns())
            .enumerate()
            .map(|(i, (column_chunk, descriptor))| {
                let column =
                    ColumnChunkMetadata::try_from_thrift(descriptor.clone(), column_chunk)?;

                column_lookup.add_column(i, &column);

                let byte_range = column.byte_range();
                full_byte_range = full_byte_range.start.min(byte_range.start)
                    ..full_byte_range.end.max(byte_range.end);

                Ok(column)
            })
            .collect::<ParquetResult<Arc<[_]>>>()?;

        Ok(RowGroupMetadata {
            columns,
            column_lookup,
            num_rows,
            total_byte_size,
            full_byte_range,
        })
    }
}
