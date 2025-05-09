use polars_error::PolarsResult;
use polars_parquet::write::{
    BrotliLevel as BrotliLevelParquet, CompressionOptions, GzipLevel as GzipLevelParquet,
    StatisticsOptions, ZstdLevel as ZstdLevelParquet,
};
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::KeyValueMetadata;

#[derive(Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

    /// Per-field overwrites for writing properties.
    pub field_overwrites: Vec<ParquetFieldOverwrites>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ChildFieldOverwrites {
    /// Flat datatypes
    None,
    /// List / Array
    ListLike(Box<ParquetFieldOverwrites>),
    Struct(Vec<ParquetFieldOverwrites>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MetadataKeyValue {
    pub key: PlSmallStr,
    pub value: Option<PlSmallStr>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParquetFieldOverwrites {
    pub name: Option<PlSmallStr>,
    pub children: ChildFieldOverwrites,
    pub field_id: Option<i32>,
    pub metadata: Option<Vec<MetadataKeyValue>>,
}

/// The compression strategy to use for writing Parquet files.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ParquetCompression {
    Uncompressed,
    Snappy,
    Gzip(Option<GzipLevel>),
    Lzo,
    Brotli(Option<BrotliLevel>),
    Zstd(Option<ZstdLevel>),
    Lz4Raw,
}

impl Default for ParquetCompression {
    fn default() -> Self {
        Self::Zstd(None)
    }
}

/// A valid Gzip compression level.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GzipLevel(u8);

impl GzipLevel {
    pub fn try_new(level: u8) -> PolarsResult<Self> {
        GzipLevelParquet::try_new(level)?;
        Ok(GzipLevel(level))
    }
}

/// A valid Brotli compression level.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BrotliLevel(u32);

impl BrotliLevel {
    pub fn try_new(level: u32) -> PolarsResult<Self> {
        BrotliLevelParquet::try_new(level)?;
        Ok(BrotliLevel(level))
    }
}

/// A valid Zstandard compression level.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ZstdLevel(i32);

impl ZstdLevel {
    pub fn try_new(level: i32) -> PolarsResult<Self> {
        ZstdLevelParquet::try_new(level)?;
        Ok(ZstdLevel(level))
    }
}

impl From<ParquetCompression> for CompressionOptions {
    fn from(value: ParquetCompression) -> Self {
        use ParquetCompression::*;
        match value {
            Uncompressed => CompressionOptions::Uncompressed,
            Snappy => CompressionOptions::Snappy,
            Gzip(level) => {
                CompressionOptions::Gzip(level.map(|v| GzipLevelParquet::try_new(v.0).unwrap()))
            },
            Lzo => CompressionOptions::Lzo,
            Brotli(level) => {
                CompressionOptions::Brotli(level.map(|v| BrotliLevelParquet::try_new(v.0).unwrap()))
            },
            Lz4Raw => CompressionOptions::Lz4Raw,
            Zstd(level) => {
                CompressionOptions::Zstd(level.map(|v| ZstdLevelParquet::try_new(v.0).unwrap()))
            },
        }
    }
}
