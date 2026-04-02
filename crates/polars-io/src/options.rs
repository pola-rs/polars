use polars_core::schema::SchemaRef;
use polars_error::{PolarsError, PolarsResult};
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct RowIndex {
    pub name: PlSmallStr,
    pub offset: IdxSize,
}

/// Options for Hive partitioning.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct HiveOptions {
    /// This can be `None` to automatically enable for single directory scans
    /// and disable otherwise. However it should be initialized if it is inside
    /// a DSL / IR plan.
    pub enabled: Option<bool>,
    pub hive_start_idx: usize,
    pub schema: Option<SchemaRef>,
    pub try_parse_dates: bool,
}

impl HiveOptions {
    pub fn new_enabled() -> Self {
        Self {
            enabled: Some(true),
            hive_start_idx: 0,
            schema: None,
            try_parse_dates: true,
        }
    }

    pub fn new_disabled() -> Self {
        Self {
            enabled: Some(false),
            hive_start_idx: 0,
            schema: None,
            try_parse_dates: false,
        }
    }
}

impl Default for HiveOptions {
    fn default() -> Self {
        Self::new_enabled()
    }
}

/// Compression options for file that are expressed externally like CSV and NDJSON. Externally does
/// not mean by an external tool, more that it doesn't happen internally like it does for Parquet
/// and IPC.
///
/// Compared to other formats like IPC and Parquet, compression is external.
#[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive()]
pub enum ExternalCompression {
    #[default]
    Uncompressed,
    Gzip {
        level: Option<u32>,
    },
    Zstd {
        level: Option<u32>,
    },
}

impl ExternalCompression {
    /// Returns the expected file suffix associated with the compression format.
    pub fn file_suffix(self) -> Option<&'static str> {
        match self {
            Self::Uncompressed => None,
            Self::Gzip { .. } => Some(".gz"),
            Self::Zstd { .. } => Some(".zst"),
        }
    }

    pub fn try_from(value: &str, level: Option<u32>) -> PolarsResult<Self> {
        match value {
            "uncompressed" => Ok(Self::Uncompressed),
            "gzip" => Ok(Self::Gzip { level }),
            "zstd" => Ok(Self::Zstd { level }),
            _ => Err(PolarsError::InvalidOperation(
                format!("Invalid compression format: ({value})").into(),
            )),
        }
    }
}
