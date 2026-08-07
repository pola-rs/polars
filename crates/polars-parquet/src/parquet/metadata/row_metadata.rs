use std::sync::Arc;

use hashbrown::hash_map::RawEntryMut;
use polars_parquet_format::SortingColumn;
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::idx_vec::UnitVec;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unitvec;

use super::column_chunk_metadata::{ColumnChunkMetadata, column_metadata_byte_range_compact};
use super::column_descriptor::ColumnDescriptorRef;
use super::compact::CompactRowGroup;
use super::schema_descriptor::SchemaDescriptor;
use crate::parquet::error::{ParquetError, ParquetResult};

type ColumnLookup = PlHashMap<PlSmallStr, UnitVec<usize>>;

#[inline(always)]
fn add_column(lookup: &mut ColumnLookup, index: usize, column: &ColumnChunkMetadata) {
    let root_name = &column.descriptor().path_in_schema[0];

    match lookup.raw_entry_mut().from_key(root_name) {
        RawEntryMut::Vacant(slot) => {
            slot.insert(root_name.clone(), unitvec![index]);
        },
        RawEntryMut::Occupied(mut slot) => {
            slot.get_mut().push(index);
        },
    }
}

/// Metadata for a row group.
#[derive(Debug, Clone, Default)]
pub struct RowGroupMetadata {
    // `ColumnChunkMetadata` is large, so we use `Arc<Vec<_>>` instead of `Arc<[_]>` to avoid
    // moving every value into a fresh Arc allocation when collecting. The `Arc<Vec<...>>`
    // form just wraps the existing Vec buffer: one Arc bump, zero element moves.
    columns: Arc<Vec<ColumnChunkMetadata>>,
    column_lookup: ColumnLookup,
    num_rows: usize,
    total_byte_size: usize,
    full_byte_range: core::ops::Range<u64>,
    sorting_columns: Option<Vec<SortingColumn>>,
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

    /// Fetch all columns under this root name if it exists.
    pub fn columns_idxs_under_root_iter<'a>(&'a self, root_name: &str) -> Option<&'a [usize]> {
        self.column_lookup.get(root_name).map(|x| x.as_slice())
    }

    pub fn parquet_columns(&self) -> &[ColumnChunkMetadata] {
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
    ///
    /// Per-chunk sizes are clamped at zero before summing so a malformed
    /// file with a negative `compressed_size` cannot underflow into a huge
    /// `usize`.
    pub fn compressed_size(&self) -> usize {
        self.columns
            .iter()
            .map(|c| c.compressed_size().max(0) as usize)
            .sum::<usize>()
    }

    pub fn full_byte_range(&self) -> core::ops::Range<u64> {
        self.full_byte_range.clone()
    }

    pub fn byte_ranges_iter(&self) -> impl ExactSizeIterator<Item = core::ops::Range<u64>> + '_ {
        self.columns.iter().map(|x| x.byte_range())
    }

    pub fn sorting_columns(&self) -> Option<&[SortingColumn]> {
        self.sorting_columns.as_deref()
    }

    /// Build a `RowGroupMetadata` from a [`CompactRowGroup`], joining each
    /// chunk to its descriptor in the schema.
    pub(crate) fn from_compact(
        schema_descr: &SchemaDescriptor,
        rg: CompactRowGroup,
    ) -> ParquetResult<RowGroupMetadata> {
        if schema_descr.columns().len() != rg.columns.len() {
            return Err(ParquetError::oos(format!(
                "The number of columns in the row group ({}) must be equal to the number of columns in the schema ({})",
                rg.columns.len(),
                schema_descr.columns().len()
            )));
        }
        let total_byte_size = rg.total_byte_size.try_into()?;
        let num_rows = rg.num_rows.try_into()?;

        let mut column_lookup = ColumnLookup::with_capacity(rg.columns.len());
        let mut full_byte_range = match rg.columns.first() {
            Some(first) => column_metadata_byte_range_compact(&first.meta_data),
            None => 0..0,
        };

        let sorting_columns = rg.sorting_columns;

        // Refcount-bump the schema's `Arc<Vec<ColumnDescriptor>>` once; each
        // chunk holds a [`ColumnDescriptorRef`] that bumps the refcount again
        // (cheap), instead of deep-cloning the descriptor.
        let column_descrs = Arc::clone(schema_descr.columns_arc());

        let columns = rg
            .columns
            .into_iter()
            .enumerate()
            .map(|(i, column_chunk)| {
                let column = ColumnChunkMetadata::from_compact(
                    ColumnDescriptorRef::new(Arc::clone(&column_descrs), i),
                    column_chunk,
                );
                add_column(&mut column_lookup, i, &column);
                let byte_range = column.byte_range();
                full_byte_range = full_byte_range.start.min(byte_range.start)
                    ..full_byte_range.end.max(byte_range.end);
                column
            })
            .collect::<Vec<_>>();
        let columns = Arc::new(columns);

        Ok(RowGroupMetadata {
            columns,
            column_lookup,
            num_rows,
            total_byte_size,
            full_byte_range,
            sorting_columns,
        })
    }
}
