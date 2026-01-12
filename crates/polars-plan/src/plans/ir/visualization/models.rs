use std::sync::Arc;

use polars_core::frame::UniqueKeepStrategy;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_ops::frame::{JoinCoalesce, JoinValidation, MaintainOrderJoin};
use polars_utils::arena::Arena;
use polars_utils::pl_path::PlRefPath;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unique_id::UniqueId;
use polars_utils::{IdxSize, format_pl_smallstr};

use crate::dsl::{PartitionStrategyIR, PredicateFileSkip, SinkTypeIR, SortColumnIR};
use crate::plans::{AExpr, ExprIR};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "ir_visualization_schema", derive(schemars::JsonSchema))]
#[derive(Debug)]
pub struct IRVisualizationData {
    pub title: PlSmallStr,
    /// Number of nodes from the start of `nodes` that are root nodes.
    pub num_roots: u64,
    pub nodes: Vec<IRNodeInfo>,
    pub edges: Vec<Edge>,
}

impl IRVisualizationData {
    pub fn to_json(&self) -> polars_error::PolarsResult<String> {
        serde_json::to_string(self).map_err(polars_error::to_compute_err)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "ir_visualization_schema", derive(schemars::JsonSchema))]
#[derive(Debug, Default)]
pub struct IRNodeInfo {
    pub id: u64,
    pub title: PlSmallStr,
    pub properties: IRNodeProperties,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "ir_visualization_schema", derive(schemars::JsonSchema))]
#[derive(Debug, Default)]
pub struct Edge {
    pub source: u64,
    pub target: u64,
}

impl Edge {
    pub fn new<T, U>(source: T, target: U) -> Self
    where
        u64: TryFrom<T> + TryFrom<U>,
        <u64 as TryFrom<T>>::Error: std::fmt::Debug,
        <u64 as TryFrom<U>>::Error: std::fmt::Debug,
    {
        Self {
            source: source.try_into().unwrap(),
            target: target.try_into().unwrap(),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type"))]
#[cfg_attr(feature = "ir_visualization_schema", derive(schemars::JsonSchema))]
#[derive(Debug, Default, strum_macros::IntoStaticStr)]
pub enum IRNodeProperties {
    Cache {
        id: UniqueId,
    },
    DataFrameScan {
        n_rows: u64,
        schema_names: Vec<PlSmallStr>,
    },
    Distinct {
        subset: Option<Vec<PlSmallStr>>,
        maintain_order: bool,
        keep_strategy: UniqueKeepStrategy,
        slice: Option<(i64, u64)>,
    },
    ExtContext {
        num_contexts: u64,
        schema_names: Vec<PlSmallStr>,
    },
    Filter {
        predicate: PlSmallStr,
    },
    GroupBy {
        keys: Vec<PlSmallStr>,
        aggs: Vec<PlSmallStr>,
        maintain_order: bool,
        slice: Option<(i64, u64)>,
        plan_callback: Option<PlSmallStr>,
    },
    HConcat {
        num_inputs: u64,
        schema_names: Vec<PlSmallStr>,
        parallel: bool,
        strict: bool,
    },
    HStack {
        exprs: Vec<PlSmallStr>,
        run_parallel: bool,
        duplicate_check: bool,
        should_broadcast: bool,
    },
    #[default]
    Invalid,
    Join {
        how: PlSmallStr,
        left_on: Vec<PlSmallStr>,
        right_on: Vec<PlSmallStr>,
        nulls_equal: bool,
        coalesce: JoinCoalesce,
        maintain_order: MaintainOrderJoin,
        validation: JoinValidation,
        suffix: Option<PlSmallStr>,
        slice: Option<(i64, u64)>,
        allow_parallel: bool,
        force_parallel: bool,
    },
    CrossJoin {
        maintain_order: MaintainOrderJoin,
        slice: Option<(i64, u64)>,
        predicate: Option<PlSmallStr>,
        suffix: Option<PlSmallStr>,
    },
    MapFunction {
        function: PlSmallStr,
    },
    Scan {
        scan_type: PlSmallStr,
        num_sources: u64,
        first_source: Option<PlSmallStr>,
        file_columns: Option<Vec<PlSmallStr>>,
        projection: Option<Vec<PlSmallStr>>,
        row_index_name: Option<PlSmallStr>,
        row_index_offset: Option<u64>,
        pre_slice: Option<(i64, u64)>,
        predicate: Option<PlSmallStr>,
        predicate_file_skip_applied: Option<PredicateFileSkip>,
        has_table_statistics: bool,
        include_file_paths: Option<PlSmallStr>,
        column_mapping_type: Option<PlSmallStr>,
        default_values_type: Option<PlSmallStr>,
        deletion_files_type: Option<PlSmallStr>,
        rechunk: bool,
        hive_columns: Option<Vec<PlSmallStr>>,
    },
    Select {
        exprs: Vec<PlSmallStr>,
        run_parallel: bool,
        duplicate_check: bool,
        should_broadcast: bool,
    },
    SimpleProjection {
        columns: Vec<PlSmallStr>,
    },
    Sink {
        payload: SinkType,
    },
    SinkMultiple {
        num_inputs: u64,
    },
    Slice {
        offset: i64,
        len: u64,
    },
    Sort {
        sort_columns: Vec<SortColumn>,
        slice: Option<(i64, u64)>,
        multithreaded: bool,
        maintain_order: bool,
        limit: Option<u64>,
    },
    Union {
        maintain_order: bool,
        parallel: bool,
        rechunk: bool,
        slice: Option<(i64, u64)>,
        from_partitioned_ds: bool,
        flattened_by_opt: bool,
    },
    //
    // Feature gated
    //
    #[cfg(feature = "asof_join")]
    AsOfJoin {
        left_on: PlSmallStr,
        right_on: PlSmallStr,
        left_by: Option<Vec<PlSmallStr>>,
        right_by: Option<Vec<PlSmallStr>>,
        strategy: polars_ops::frame::AsofStrategy,
        /// [value, dtype_str]
        tolerance: Option<[PlSmallStr; 2]>,
        suffix: Option<PlSmallStr>,
        slice: Option<(i64, u64)>,
        coalesce: JoinCoalesce,
        allow_eq: bool,
        check_sortedness: bool,
    },
    #[cfg(feature = "iejoin")]
    IEJoin {
        left_on: Vec<PlSmallStr>,
        right_on: Vec<PlSmallStr>,
        inequality_operators: Vec<polars_ops::frame::InequalityOperator>,
        suffix: Option<PlSmallStr>,
        slice: Option<(i64, u64)>,
    },
    #[cfg(feature = "dynamic_group_by")]
    DynamicGroupBy {
        index_column: PlSmallStr,
        every: PlSmallStr,
        period: PlSmallStr,
        offset: PlSmallStr,
        label: polars_time::prelude::Label,
        include_boundaries: bool,
        closed_window: polars_time::ClosedWindow,
        group_by: Vec<PlSmallStr>,
        start_by: polars_time::prelude::StartBy,
        plan_callback: Option<PlSmallStr>,
    },
    #[cfg(feature = "dynamic_group_by")]
    RollingGroupBy {
        keys: Vec<PlSmallStr>,
        aggs: Vec<PlSmallStr>,
        index_column: PlSmallStr,
        period: PlSmallStr,
        offset: PlSmallStr,
        closed_window: polars_time::ClosedWindow,
        slice: Option<(i64, u64)>,
        plan_callback: Option<PlSmallStr>,
    },
    #[cfg(feature = "merge_sorted")]
    MergeSorted {
        key: PlSmallStr,
    },
    #[cfg(feature = "python")]
    PythonScan {
        scan_source_type: crate::prelude::python_dsl::PythonScanSource,
        n_rows: Option<u64>,
        projection: Option<Vec<PlSmallStr>>,
        predicate: Option<PlSmallStr>,
        schema_names: Vec<PlSmallStr>,
        is_pure: bool,
        validate_schema: bool,
    },
}

pub trait FromWithArena<T>: Sized {
    fn from_with_arena(value: T, expr_arena: &Arena<AExpr>) -> Self;
}

pub trait IntoWithArena<T> {
    fn into_with_arena(self, expr_arena: &Arena<AExpr>) -> T;
}

impl<T, U> IntoWithArena<U> for T
where
    U: FromWithArena<T>,
{
    fn into_with_arena(self, expr_arena: &Arena<AExpr>) -> U {
        U::from_with_arena(self, expr_arena)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "ir_visualization_schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub enum SinkType {
    Memory,
    Callback(CallbackSinkType),
    File(FileSinkOptions),
    #[cfg_attr(all(feature = "serde", not(feature = "ir_serde")), serde(skip))]
    Partitioned(PartitionedSinkOptions),
}

impl FromWithArena<&SinkTypeIR> for SinkType {
    fn from_with_arena(value: &SinkTypeIR, expr_arena: &Arena<AExpr>) -> Self {
        match value {
            SinkTypeIR::Memory => Self::Memory,
            SinkTypeIR::Callback(c) => Self::Callback(CallbackSinkType {
                maintain_order: c.maintain_order,
                chunk_size: c.chunk_size,
            }),
            SinkTypeIR::File(f) => Self::File(FileSinkOptions {
                target: (&f.target).into(),
                file_format: (&f.file_format).into(),
                mkdir: f.unified_sink_args.mkdir,
                maintain_order: f.unified_sink_args.maintain_order,
                sync_on_close: f.unified_sink_args.sync_on_close,
            }),
            SinkTypeIR::Partitioned(p) => Self::Partitioned(PartitionedSinkOptions {
                base_path: p.base_path.clone(),
                file_path_provider: (&p.file_path_provider).into(),
                partition_strategy: (&p.partition_strategy).into_with_arena(expr_arena),
                file_format: (&p.file_format).into(),
                mkdir: p.unified_sink_args.mkdir,
                maintain_order: p.unified_sink_args.maintain_order,
                sync_on_close: p.unified_sink_args.sync_on_close,
                max_rows_per_file: p.max_rows_per_file,
                approximate_bytes_per_file: p.approximate_bytes_per_file,
            }),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "ir_visualization_schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct CallbackSinkType {
    pub maintain_order: bool,
    pub chunk_size: Option<std::num::NonZeroUsize>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "ir_visualization_schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct FileSinkOptions {
    pub target: SinkTarget,
    pub file_format: FileFormat,
    pub mkdir: bool,
    pub maintain_order: bool,
    pub sync_on_close: SyncOnCloseType,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "ir_visualization_schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum SinkTarget {
    Path(PlRefPath),
    Dyn,
}

impl From<&crate::dsl::SinkTarget> for SinkTarget {
    fn from(value: &crate::dsl::SinkTarget) -> Self {
        match value {
            crate::dsl::SinkTarget::Path(path) => Self::Path(path.to_owned()),
            crate::dsl::SinkTarget::Dyn(_) => Self::Dyn,
        }
    }
}

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "ir_visualization_schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionedSinkOptions {
    pub base_path: PlRefPath,
    pub file_path_provider: FileProviderType,
    pub partition_strategy: PartitionStrategy,
    pub file_format: FileFormat,
    pub mkdir: bool,
    pub maintain_order: bool,
    pub sync_on_close: SyncOnCloseType,
    pub max_rows_per_file: IdxSize,
    pub approximate_bytes_per_file: u64,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "ir_visualization_schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum FileProviderType {
    Hive { extension: PlSmallStr },
    Function,
    Legacy,
}

impl From<&crate::dsl::sink2::FileProviderType> for FileProviderType {
    fn from(value: &crate::dsl::sink2::FileProviderType) -> Self {
        match value {
            crate::dsl::sink2::FileProviderType::Hive { extension } => Self::Hive {
                extension: extension.to_owned(),
            },
            crate::dsl::sink2::FileProviderType::Function(_) => Self::Function,
            crate::dsl::sink2::FileProviderType::Legacy(_) => Self::Legacy,
        }
    }
}

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "ir_visualization_schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub enum PartitionStrategy {
    Keyed {
        keys: Vec<PlSmallStr>,
        include_keys: bool,
        keys_pregrouped: bool,
        pre_partition_sort_by: Vec<SortColumn>,
    },
    FileSize,
}

impl FromWithArena<&PartitionStrategyIR> for PartitionStrategy {
    fn from_with_arena(value: &PartitionStrategyIR, expr_arena: &Arena<AExpr>) -> Self {
        match value {
            PartitionStrategyIR::Keyed {
                keys,
                include_keys,
                keys_pre_grouped,
                per_partition_sort_by,
            } => Self::Keyed {
                keys: expr_list(keys, expr_arena),
                include_keys: *include_keys,
                keys_pregrouped: *keys_pre_grouped,
                pre_partition_sort_by: per_partition_sort_by
                    .iter()
                    .map(
                        |SortColumnIR {
                             expr,
                             descending,
                             nulls_last,
                         }| SortColumn {
                            expr: format_pl_smallstr!("{}", expr.display(expr_arena)),
                            descending: *descending,
                            nulls_last: *nulls_last,
                        },
                    )
                    .collect(),
            },
            PartitionStrategyIR::FileSize => Self::FileSize,
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "ir_visualization_schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum FileFormat {
    Parquet,
    Ipc,
    Csv,
    NDJson,
}

impl From<&crate::dsl::FileWriteFormat> for FileFormat {
    fn from(value: &crate::dsl::FileWriteFormat) -> Self {
        match value {
            crate::dsl::FileWriteFormat::Parquet(_) => FileFormat::Parquet,
            crate::dsl::FileWriteFormat::Ipc(_) => FileFormat::Ipc,
            crate::dsl::FileWriteFormat::Csv(_) => FileFormat::Csv,
            crate::dsl::FileWriteFormat::NDJson(_) => FileFormat::NDJson,
        }
    }
}

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "ir_visualization_schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub struct SortColumn {
    pub expr: PlSmallStr,
    pub descending: bool,
    pub nulls_last: bool,
}

pub fn expr_list(exprs: &[ExprIR], expr_arena: &Arena<AExpr>) -> Vec<PlSmallStr> {
    exprs
        .iter()
        .map(|e| format_pl_smallstr!("{}", e.display(expr_arena)))
        .collect()
}
