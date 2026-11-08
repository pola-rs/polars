use std::sync::Arc;

use hashbrown::hash_map::RawEntryMut;
use polars_parquet_format::SortingColumn;
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::idx_vec::UnitVec;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unitvec;

use super::column_chunk_metadata::ColumnChunkMetadata;
use super::column_descriptor::ColumnDescriptorRef;
use super::compact::CompactRowGroup;
use super::schema_descriptor::SchemaDescriptor;
use crate::parquet::encryption::decrypt::{ColumnChunkDecryption, FileDecryptor};
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

    #[inline]
    pub fn parquet_columns(&self) -> &[ColumnChunkMetadata] {
        &self.columns
    }

    /// Number of rows in this row group.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Total byte size of all uncompressed column data in this row group.
    #[inline]
    pub fn total_byte_size(&self) -> usize {
        self.total_byte_size
    }

    /// Total size of all compressed column data in this row group.
    ///
    /// Per-chunk sizes are clamped at zero before summing so a malformed
    /// file with a negative `compressed_size` cannot underflow into a huge
    /// `usize`.
    ///
    /// Columns without metadata (encrypted columns whose key is unavailable) are excluded.
    pub fn compressed_size(&self) -> usize {
        self.columns
            .iter()
            .filter_map(|c| c.compressed_size().ok())
            .map(|size| size.max(0) as usize)
            .sum::<usize>()
    }

    /// The byte range covering all columns that have metadata.
    pub fn full_byte_range(&self) -> core::ops::Range<u64> {
        self.full_byte_range.clone()
    }

    /// The byte ranges of all columns that have metadata. Columns without metadata
    /// (encrypted columns whose key is unavailable) are excluded.
    pub fn byte_ranges_iter(&self) -> impl Iterator<Item = core::ops::Range<u64>> + '_ {
        self.columns.iter().filter_map(|c| c.byte_range().ok())
    }

    pub fn sorting_columns(&self) -> Option<&[SortingColumn]> {
        self.sorting_columns.as_deref()
    }

    /// Build a `RowGroupMetadata` from a [`CompactRowGroup`], joining each
    /// chunk to its descriptor in the schema.
    ///
    /// For encrypted files, `decryption` holds the file decryptor and the index of this row
    /// group within the file, which are needed to decrypt encrypted column chunks.
    pub(crate) fn from_compact(
        schema_descr: &SchemaDescriptor,
        rg: CompactRowGroup,
        decryption: Option<(&Arc<FileDecryptor>, usize)>,
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
        // Chunks may lack metadata if they're encrypted with an unavailable column key.
        // These can't be read, so are excluded from the full byte range.
        let mut full_byte_range: Option<core::ops::Range<u64>> = None;

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
                let chunk_decryption = decryption.filter(|_| column_chunk.crypto.is_some()).map(
                    |(file_decryptor, row_group_idx)| {
                        Box::new(ColumnChunkDecryption {
                            file_decryptor: Arc::clone(file_decryptor),
                            row_group_idx,
                            column_ordinal: i,
                        })
                    },
                );
                let column = ColumnChunkMetadata::from_compact(
                    ColumnDescriptorRef::new(Arc::clone(&column_descrs), i),
                    column_chunk,
                    chunk_decryption,
                );
                add_column(&mut column_lookup, i, &column);
                if let Ok(byte_range) = column.byte_range() {
                    full_byte_range = Some(match full_byte_range.take() {
                        Some(range) => {
                            range.start.min(byte_range.start)..range.end.max(byte_range.end)
                        },
                        None => byte_range,
                    });
                }
                column
            })
            .collect::<Vec<_>>();
        let full_byte_range = full_byte_range.unwrap_or(0..0);
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
