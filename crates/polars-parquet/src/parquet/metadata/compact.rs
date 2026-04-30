//! Compact replacements for the format-crate read-path types: `FileMetaData`,
//! `RowGroup`, `ColumnChunk`, `ColumnMetaData`, `Statistics`. Produced by the
//! hand-written Thrift decoder in [`crate::parquet::handwritten_thrift`] and
//! consumed by [`super::FileMetadata::from_compact`].
//!
//! Two wins over the format-crate types:
//!
//! 1. **Smaller resident footprint.** Fields with no in-tree consumer
//!    (`encodings`, `path_in_schema`, `key_value_metadata`,
//!    `encoding_stats`, `size_statistics`, `type_`) are omitted entirely.
//!    Saves ~110 bytes per `ColumnChunk` × 200k chunks on a wide file.
//!
//! 2. **Shared-buffer statistics.** `min_value` / `max_value` are stored as
//!    `(offset, len)` ranges into the footer [`Buffer<u8>`] rather than
//!    per-stat `Vec<u8>` allocations. On a 10k-col × 20-rg file with stats
//!    this drops ~400k heap allocs to zero. The buffer itself is held once
//!    on [`super::FileMetadata::footer_buf`].

use polars_buffer::Buffer;
use polars_parquet_format::{ColumnOrder, KeyValue, SchemaElement, SortingColumn};

use crate::parquet::compression::Compression;

/// `(offset, len)` into a shared [`Buffer<u8>`] holding the footer bytes.
/// Used by [`CompactStatistics`] to reference `min_value` / `max_value`
/// payloads in the footer without per-value heap allocation.
///
/// Both fields are `u32`; footers larger than 4 GiB cannot be represented.
/// Parquet footer size is dominated by per-column metadata and in practice
/// sits well under this limit even for 100k-column tables. Construction
/// sites cast from `usize` with `as u32` (silent truncation). Audit these
/// if the 4 GiB invariant is ever in doubt.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ByteRange {
    pub offset: u32,
    pub len: u32,
}

impl ByteRange {
    /// Resolve this range against the shared `buf`. Caller must pass the
    /// same buffer the range was produced from (typically the parquet footer).
    #[inline]
    pub(crate) fn resolve<'a>(&self, buf: &'a [u8]) -> &'a [u8] {
        &buf[self.offset as usize..(self.offset as usize + self.len as usize)]
    }
}

/// Compact replacement for `polars_parquet_format::Statistics`.
///
/// Deprecated `max` / `min` fields skipped (polars reads `max_value` /
/// `min_value`). Modern values are stored as offsets into the file's
/// footer buffer, available via [`super::FileMetadata::footer_buf`].
#[derive(Debug, Clone)]
pub(crate) struct CompactStatistics {
    pub null_count: Option<i64>,
    pub distinct_count: Option<i64>,
    pub max_value: Option<ByteRange>,
    pub min_value: Option<ByteRange>,
    pub is_max_value_exact: Option<bool>,
    pub is_min_value_exact: Option<bool>,
}

/// Compact replacement for `polars_parquet_format::ColumnMetaData`.
///
/// Drops fields with no read-path consumer:
/// - `type_`: `ColumnDescriptor::primitive_type.physical_type` covers it
/// - `encodings`: no in-workspace caller
/// - `path_in_schema`: `ColumnDescriptor::path_in_schema` covers it
/// - `key_value_metadata`: only file-level KV is exposed
/// - `encoding_stats`, `size_statistics`: no public accessors
#[derive(Debug, Clone)]
pub(crate) struct CompactColumnMetaData {
    pub codec: Compression,
    pub num_values: i64,
    pub total_uncompressed_size: i64,
    pub total_compressed_size: i64,
    pub data_page_offset: i64,
    pub index_page_offset: Option<i64>,
    pub dictionary_page_offset: Option<i64>,
    pub statistics: Option<CompactStatistics>,
    pub bloom_filter_offset: Option<i64>,
    pub bloom_filter_length: Option<i32>,
}

/// Compact replacement for `polars_parquet_format::ColumnChunk`.
///
/// Drops `file_path`, `file_offset`, encryption fields. None have read-path
/// consumers in this build (the write path constructs format-crate types).
///
/// `meta_data` is non-`Option` because polars has no encryption support: a
/// chunk without unencrypted metadata is unrepresentable here. The decoder
/// rejects such chunks at footer-decode time with `"ColumnChunk.meta_data
/// missing"`, so by the time a `CompactColumnChunk` exists the field is
/// guaranteed present.
#[derive(Debug, Clone)]
pub(crate) struct CompactColumnChunk {
    pub meta_data: CompactColumnMetaData,
    pub offset_index_offset: Option<i64>,
    pub offset_index_length: Option<i32>,
    pub column_index_offset: Option<i64>,
    pub column_index_length: Option<i32>,
}

/// Compact replacement for `polars_parquet_format::RowGroup`.
///
/// `file_offset`, `total_compressed_size`, `ordinal` are dropped (no
/// in-workspace consumers).
#[derive(Debug, Clone)]
pub(crate) struct CompactRowGroup {
    pub columns: Vec<CompactColumnChunk>,
    pub total_byte_size: i64,
    pub num_rows: i64,
    pub sorting_columns: Option<Vec<SortingColumn>>,
}

/// Compact replacement for `polars_parquet_format::FileMetaData`.
///
/// `schema`, `key_value_metadata`, `column_orders` stay as lists of
/// format-crate structs because downstream consumers
/// (`SchemaDescriptor::try_from_thrift`, `FileMetadata::key_value_metadata`,
/// `parse_column_orders`) expect that shape. Refactoring those would cross
/// the `polars-parquet-format` boundary. (Note: `sorting_columns` lives on
/// `CompactRowGroup`, not here, and is also kept as the format-crate type.)
#[derive(Debug, Clone)]
pub(crate) struct CompactFileMetaData {
    pub version: i32,
    pub schema: Vec<SchemaElement>,
    pub num_rows: i64,
    pub row_groups: Vec<CompactRowGroup>,
    pub key_value_metadata: Option<Vec<KeyValue>>,
    pub created_by: Option<String>,
    pub column_orders: Option<Vec<ColumnOrder>>,
    /// The footer buffer the [`CompactStatistics`] `ByteRange`s point into.
    /// `from_compact` stores it on [`super::FileMetadata::footer_buf`] so
    /// stats payloads remain resolvable for the lifetime of the metadata.
    pub footer_buf: Buffer<u8>,
}
