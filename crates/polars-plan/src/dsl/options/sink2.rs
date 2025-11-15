use std::hash::{Hash, Hasher};

use polars_io::cloud::CloudOptions;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_utils::IdxSize;
use polars_utils::arena::Arena;
use polars_utils::plpath::{CloudScheme, PlPath};

use super::Expr;
use super::sink::*;
use crate::plans::{AExpr, ExprIR};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct UnifiedSinkArgs {
    pub mkdir: bool,
    pub maintain_order: bool,
    pub sync_on_close: SyncOnCloseType,
    pub cloud_options: Option<CloudOptions>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FileSinkType {
    File(SinkTarget),
    Partitioned(PartitionedSinkOptions),
}

impl FileSinkType {
    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        match self {
            Self::File(x) => x.cloud_scheme(),
            Self::Partitioned(x) => x.cloud_scheme(),
        }
    }
}

impl PartitionedSinkOptionsIR {
    pub fn expr_irs_iter(&self) -> impl ExactSizeIterator<Item = &ExprIR> {
        let mut partition_key_exprs: &[ExprIR] = &[];
        let sort_exprs: &[SortColumnIR];

        match &self.partition_strategy {
            PartitionStrategyIR::Keyed {
                keys,
                include_keys: _,
                keys_sorted: _,
                per_partition_sort_by,
            } => {
                partition_key_exprs = keys.as_slice();
                sort_exprs = per_partition_sort_by.as_slice();
            },
            PartitionStrategyIR::MaxRowsPerFile {
                max_rows_per_file: _,
                per_file_sort_by,
            } => {
                sort_exprs = per_file_sort_by.as_slice();
            },
        };

        (0..partition_key_exprs.len() + sort_exprs.len()).map(|i| {
            if i < partition_key_exprs.len() {
                &partition_key_exprs[i]
            } else {
                &sort_exprs[i - partition_key_exprs.len()].expr
            }
        })
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionedSinkOptions {
    pub base_path: PlPath,
    pub file_path_provider: Option<PartitionTargetCallback>,
    pub partition_strategy: PartitionStrategy,
    /// TODO: Move this to UnifiedSinkArgs
    pub finish_callback: Option<SinkFinishCallback>,
}

impl PartitionedSinkOptions {
    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        CloudScheme::from_uri(self.base_path.to_str())
    }
}

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionedSinkOptionsIR {
    pub base_path: PlPath,
    pub file_path_provider: Option<PartitionTargetCallback>,
    pub partition_strategy: PartitionStrategyIR,
    /// TODO: Move this to UnifiedSinkArgs
    pub finish_callback: Option<SinkFinishCallback>,
}

impl PartitionedSinkOptionsIR {
    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        CloudScheme::from_uri(self.base_path.to_str())
    }

    #[cfg(feature = "cse")]
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        let PartitionedSinkOptionsIR {
            base_path,
            file_path_provider,
            partition_strategy,
            finish_callback,
        } = self;

        base_path.hash(state);
        file_path_provider.hash(state);
        partition_strategy.traverse_and_hash(expr_arena, state);
        finish_callback.hash(state);
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub enum PartitionStrategy {
    Keyed {
        keys: Vec<Expr>,
        include_keys: bool,
        keys_sorted: bool,
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
        keys_sorted: bool,
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
                keys_sorted,
                per_partition_sort_by,
            } => {
                for k in keys {
                    k.traverse_and_hash(expr_arena, state);
                }

                include_keys.hash(state);
                keys_sorted.hash(state);

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
