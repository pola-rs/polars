use arrow::datatypes::ArrowSchemaRef;
use polars_parquet::write::{
    BrotliLevel, CompressionOptions, GzipLevel, StatisticsOptions, ZstdLevel,
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::KeyValueMetadata;

#[derive(Default, Clone, Debug, PartialEq, Hash, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct ParquetWriteOptions {
    /// Data page compression
    pub compression: ParquetCompression,
    /// Compute and write column statistics.
    pub statistics: StatisticsOptions,
    /// If `None` will be all written to a single row group.
    pub row_group_size: Option<usize>,
    /// if `None` will be 1024^2 bytes
    pub data_page_size: Option<usize>,
    /// Custom file-level key value metadata
    pub key_value_metadata: Option<KeyValueMetadata>,
    pub arrow_schema: Option<ArrowSchemaRef>,
}

/// The compression strategy to use for writing Parquet files.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ParquetCompression {
    Uncompressed,
    Snappy,
    Gzip(Option<GzipLevel>),
    Brotli(Option<BrotliLevel>),
    Zstd(Option<ZstdLevel>),
    Lz4Raw,
}

impl Default for ParquetCompression {
    fn default() -> Self {
        Self::Zstd(None)
    }
}

impl From<ParquetCompression> for CompressionOptions {
    fn from(value: ParquetCompression) -> Self {
        use ParquetCompression::*;
        match value {
            Uncompressed => CompressionOptions::Uncompressed,
            Snappy => CompressionOptions::Snappy,
            Gzip(level) => CompressionOptions::Gzip(level),
            Brotli(level) => CompressionOptions::Brotli(level),
            Lz4Raw => CompressionOptions::Lz4Raw,
            Zstd(level) => CompressionOptions::Zstd(level),
        }
    }
}
