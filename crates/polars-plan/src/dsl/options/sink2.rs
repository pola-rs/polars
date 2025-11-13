use polars_io::cloud::CloudOptions;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_utils::IdxSize;
use polars_utils::plpath::{CloudScheme, PlPath};

use super::Expr;
use super::sink::*;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub enum SinkOutputType {
    Single(SinkTarget),
    Multiple(DirectorySinkOptions),
}

impl SinkOutputType {
    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        match self {
            Self::Single(x) => match x {
                SinkTarget::Path(p) => CloudScheme::from_uri(p.to_str()),
                SinkTarget::Dyn(_) => None,
            },
            Self::Multiple(x) => x.cloud_scheme(),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub struct UnifiedSinkArgs {
    pub mkdir: bool,
    pub maintain_order: bool,
    pub sync_on_close: SyncOnCloseType,
    pub cloud_options: Option<CloudOptions>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub struct DirectorySinkOptions {
    pub base_path: PlPath,
    pub file_path_provider: Option<PartitionTargetCallback>,
    pub sink_type: PartitionVariant2,
    /// TODO: Move this to UnifiedSinkArgs
    pub finish_callback: Option<SinkFinishCallback>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub enum PartitionVariant2 {
    PartitionBy {
        keys: Vec<Expr>,
        include_keys: bool,
        keys_sorted: bool,
        per_partition_sort_by: Vec<SortColumn>,
        // TODO: `max_rows_per_file` parameter.
    },
    MaxSize {
        max_rows_per_file: IdxSize,
        per_file_sort_by: Vec<SortColumn>,
    },
}

impl DirectorySinkOptions {
    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        CloudScheme::from_uri(self.base_path.to_str())
    }
}
