use std::hash::{Hash, Hasher};

use polars_io::cloud::CloudOptions;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_utils::IdxSize;
use polars_utils::arena::Arena;
use polars_utils::plpath::{CloudScheme, PlPath};

use super::Expr;
use super::sink::*;
use crate::plans::{AExpr, ExprIR};

#[derive(Clone, Debug, PartialEq)]
pub enum SinkDestination {
    File {
        target: SinkTarget,
    },
    Partitioned {
        base_path: PlPath,
        file_path_provider: Option<PartitionTargetCallback>,
        partition_strategy: PartitionStrategy,
        finish_callback: Option<SinkFinishCallback>,
    },
}

impl SinkDestination {
    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        match self {
            Self::File { target } => target.cloud_scheme(),
            Self::Partitioned { base_path, .. } => base_path.cloud_scheme(),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct UnifiedSinkArgs {
    pub mkdir: bool,
    pub maintain_order: bool,
    pub sync_on_close: SyncOnCloseType,
    pub cloud_options: Option<CloudOptions>,
}

impl Default for UnifiedSinkArgs {
    fn default() -> Self {
        Self {
            mkdir: false,
            maintain_order: true,
            sync_on_close: SyncOnCloseType::None,
            cloud_options: None,
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub enum PartitionStrategy {
    Keyed {
        keys: Vec<Expr>,
        include_keys: bool,
        keys_pre_grouped: bool,
        per_partition_sort_by: Vec<SortColumn>,
        // TODO: `max_rows_per_file` parameter.
    },
    MaxRowsPerFile {
        max_rows_per_file: IdxSize,
        per_file_sort_by: Vec<SortColumn>,
    },
}

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum PartitionStrategyIR {
    Keyed {
        keys: Vec<ExprIR>,
        include_keys: bool,
        keys_pre_grouped: bool,
        per_partition_sort_by: Vec<SortColumnIR>,
    },
    MaxRowsPerFile {
        max_rows_per_file: IdxSize,
        per_file_sort_by: Vec<SortColumnIR>,
    },
}

#[cfg(feature = "cse")]
impl PartitionStrategyIR {
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Keyed {
                keys,
                include_keys,
                keys_pre_grouped,
                per_partition_sort_by,
            } => {
                for k in keys {
                    k.traverse_and_hash(expr_arena, state);
                }

                include_keys.hash(state);
                keys_pre_grouped.hash(state);

                for x in per_partition_sort_by {
                    x.traverse_and_hash(expr_arena, state);
                }
            },
            Self::MaxRowsPerFile {
                max_rows_per_file,
                per_file_sort_by,
            } => {
                max_rows_per_file.hash(state);

                for x in per_file_sort_by {
                    x.traverse_and_hash(expr_arena, state);
                }
            },
        }
    }
}
