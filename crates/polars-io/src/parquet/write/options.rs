use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use polars_arrow::datatypes::ArrowSchemaRef;
use polars_core::prelude::CompatLevel;
use polars_parquet::parquet::encryption::encrypt::FileEncryptionProperties;
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
    #[cfg_attr(feature = "serde", serde(default))]
    pub compat_level: Option<CompatLevel>,
    /// Properties for writing files encrypted with Parquet modular encryption
    #[cfg_attr(feature = "serde", serde(default))]
    pub encryption_properties: Option<PlFileEncryptionProperties>,
}

impl ParquetWriteOptions {
    pub fn compat_level(&self) -> CompatLevel {
        self.compat_level.unwrap_or(CompatLevel::oldest())
    }
}

/// Properties for writing files encrypted with Parquet modular encryption.
///
/// These hold secret keys, so can't be serialized, and are compared and hashed
/// by pointer.
#[derive(Clone)]
pub struct PlFileEncryptionProperties(pub Arc<FileEncryptionProperties>);

impl Debug for PlFileEncryptionProperties {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // FileEncryptionProperties doesn't output keys.
        self.0.fmt(f)
    }
}

impl Eq for PlFileEncryptionProperties {}

impl PartialEq for PlFileEncryptionProperties {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Hash for PlFileEncryptionProperties {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(Arc::as_ptr(&self.0) as usize)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for PlFileEncryptionProperties {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        Err(D::Error::custom(
            "cannot deserialize parquet encryption properties",
        ))
    }
}

#[cfg(feature = "serde")]
impl Serialize for PlFileEncryptionProperties {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;
        Err(S::Error::custom(
            "cannot serialize parquet encryption properties",
        ))
    }
}

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for PlFileEncryptionProperties {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "PlFileEncryptionProperties".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "PlFileEncryptionProperties"))
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Vec::<u8>::json_schema(generator)
    }
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
