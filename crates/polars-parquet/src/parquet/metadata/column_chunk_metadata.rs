use polars_buffer::Buffer;
use polars_parquet_format::Statistics as ParquetStatistics;

use super::column_descriptor::{ColumnDescriptor, ColumnDescriptorRef};
use super::compact::{CompactColumnChunk, CompactColumnMetaData, CompactStatistics};
use crate::parquet::compression::Compression;
use crate::parquet::error::ParquetResult;
use crate::parquet::schema::types::PhysicalType;
use crate::parquet::statistics::Statistics;

/// Metadata for a column chunk.
///
/// Wraps a [`CompactColumnChunk`] (drops fields with no in-tree consumer).
/// Stats `min_value` / `max_value` are stored as `(offset, len)` ranges
/// into the file's footer buffer; resolve them by passing
/// `&FileMetadata::footer_buf` to [`Self::statistics`].
///
/// `column_descr` is a [`ColumnDescriptorRef`]: refcount-bump on clone, no
/// deep `ColumnDescriptor` copy. All chunks of one file share a single
/// underlying `Vec<ColumnDescriptor>` allocation.
///
/// This struct is intentionally not `Clone`, as it is a huge struct.
#[derive(Debug)]
pub struct ColumnChunkMetadata {
    column_chunk: CompactColumnChunk,
    column_descr: ColumnDescriptorRef,
}

// Represents common operations for a column chunk.
impl ColumnChunkMetadata {
    /// The compact column metadata for this chunk. Always present;
    /// encrypted columns are rejected at footer-decode time.
    ///
    /// Crate-internal: callers outside `polars-parquet` should use the
    /// typed accessors below (`compression()`, `num_values()`, etc.)
    /// rather than reaching into the compact representation directly.
    #[inline]
    pub(crate) fn compact_metadata(&self) -> &CompactColumnMetaData {
        &self.column_chunk.meta_data
    }

    /// The [`ColumnDescriptor`] for this column. This descriptor contains
    /// the physical and logical type of the pages.
    pub fn descriptor(&self) -> &ColumnDescriptor {
        &self.column_descr
    }

    /// The [`PhysicalType`] of this column.
    pub fn physical_type(&self) -> PhysicalType {
        self.descriptor().descriptor.primitive_type.physical_type
    }

    /// Decodes the raw statistics into [`Statistics`].
    ///
    /// `CompactStatistics` stores min/max as `ByteRange`s into the footer
    /// buffer to avoid per-stat allocations during decode. This accessor
    /// materialises the bytes back into a `ParquetStatistics` (one alloc
    /// per side present) and runs the existing per-type deserializer.
    ///
    /// `footer_buf` must be the same buffer this chunk was decoded from,
    /// typically [`super::FileMetadata::footer_buf`].
    pub fn statistics(&self, footer_buf: &Buffer<u8>) -> Option<ParquetResult<Statistics>> {
        let stats = self.compact_metadata().statistics.as_ref()?;
        let parquet_stats = compact_stats_to_parquet(stats, footer_buf);
        Some(Statistics::deserialize(
            &parquet_stats,
            self.descriptor().descriptor.primitive_type.clone(),
        ))
    }

    /// Total number of values in this column chunk. Note that this is not
    /// necessarily the number of rows. E.g. the (nested) array `[[1, 2], [3]]`
    /// has 2 rows and 3 values.
    pub fn num_values(&self) -> i64 {
        self.compact_metadata().num_values
    }

    /// [`Compression`] for this column.
    pub fn compression(&self) -> Compression {
        self.compact_metadata().codec
    }

    /// Returns the total compressed data size of this column chunk.
    pub fn compressed_size(&self) -> i64 {
        self.compact_metadata().total_compressed_size
    }

    /// Returns the total uncompressed data size of this column chunk.
    pub fn uncompressed_size(&self) -> i64 {
        self.compact_metadata().total_uncompressed_size
    }

    /// Returns the offset for the column data.
    pub fn data_page_offset(&self) -> i64 {
        self.compact_metadata().data_page_offset
    }

    /// Returns `true` if this column chunk contains an index page, `false` otherwise.
    pub fn has_index_page(&self) -> bool {
        self.compact_metadata().index_page_offset.is_some()
    }

    /// Returns the offset for the index page.
    pub fn index_page_offset(&self) -> Option<i64> {
        self.compact_metadata().index_page_offset
    }

    /// Returns the offset for the dictionary page, if any.
    pub fn dictionary_page_offset(&self) -> Option<i64> {
        self.compact_metadata().dictionary_page_offset
    }

    /// Bloom filter byte offset, if present.
    pub fn bloom_filter_offset(&self) -> Option<i64> {
        self.compact_metadata().bloom_filter_offset
    }

    /// Bloom filter byte length, if present.
    pub fn bloom_filter_length(&self) -> Option<i32> {
        self.compact_metadata().bloom_filter_length
    }

    /// PageIndex `OffsetIndex` byte offset, if present.
    pub fn offset_index_offset(&self) -> Option<i64> {
        self.column_chunk.offset_index_offset
    }

    /// PageIndex `OffsetIndex` byte length, if present.
    pub fn offset_index_length(&self) -> Option<i32> {
        self.column_chunk.offset_index_length
    }

    /// PageIndex `ColumnIndex` byte offset, if present.
    pub fn column_index_offset(&self) -> Option<i64> {
        self.column_chunk.column_index_offset
    }

    /// PageIndex `ColumnIndex` byte length, if present.
    pub fn column_index_length(&self) -> Option<i32> {
        self.column_chunk.column_index_length
    }

    /// Returns the offset and length in bytes of the column chunk within the file.
    pub fn byte_range(&self) -> core::ops::Range<u64> {
        column_metadata_byte_range_compact(self.compact_metadata())
    }

    /// Build from a [`CompactColumnChunk`] + descriptor handle.
    /// Infallible: the decoder rejects malformed chunks (missing
    /// `meta_data`) up front, so by here the invariant is type-enforced.
    pub(crate) fn from_compact(
        column_descr: ColumnDescriptorRef,
        column_chunk: CompactColumnChunk,
    ) -> Self {
        Self {
            column_chunk,
            column_descr,
        }
    }
}

/// Materialise a `polars_parquet_format::Statistics` from a `CompactStatistics`
/// by resolving the `ByteRange`s against `footer_buf`. Allocates 0-2 `Vec<u8>`s
/// (one per side present). Used by [`ColumnChunkMetadata::statistics`].
#[inline]
fn compact_stats_to_parquet(s: &CompactStatistics, footer_buf: &[u8]) -> ParquetStatistics {
    ParquetStatistics {
        max: None,
        min: None,
        null_count: s.null_count,
        distinct_count: s.distinct_count,
        max_value: s.max_value.map(|r| r.resolve(footer_buf).to_vec()),
        min_value: s.min_value.map(|r| r.resolve(footer_buf).to_vec()),
        is_max_value_exact: s.is_max_value_exact,
        is_min_value_exact: s.is_min_value_exact,
    }
}

pub(super) fn column_metadata_byte_range_compact(
    column_metadata: &CompactColumnMetaData,
) -> core::ops::Range<u64> {
    let offset = if let Some(dict_page_offset) = column_metadata.dictionary_page_offset {
        dict_page_offset as u64
    } else {
        column_metadata.data_page_offset as u64
    };
    let len = column_metadata.total_compressed_size as u64;
    offset..offset.checked_add(len).unwrap()
}
