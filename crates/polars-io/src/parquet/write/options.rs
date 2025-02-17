use polars_error::PolarsResult;
use polars_parquet::write::{
    BrotliLevel as BrotliLevelParquet, CompressionOptions, GzipLevel as GzipLevelParquet,
    StatisticsOptions, ZstdLevel as ZstdLevelParquet,
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Hash)]
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
    /// maintain the order the data was processed
    pub maintain_order: bool,
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
