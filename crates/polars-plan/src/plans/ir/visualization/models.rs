use polars_core::frame::UniqueKeepStrategy;
use polars_ops::frame::{JoinCoalesce, JoinValidation, MaintainOrderJoin};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unique_id::UniqueId;

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
        slice: Option<[i128; 2]>,
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
        slice: Option<[i128; 2]>,
        plan_callback: Option<PlSmallStr>,
    },
    HConcat {
        num_inputs: u64,
        schema_names: Vec<PlSmallStr>,
        parallel: bool,
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
        slice: Option<[i128; 2]>,
        allow_parallel: bool,
        force_parallel: bool,
    },
    CrossJoin {
        maintain_order: MaintainOrderJoin,
        slice: Option<[i128; 2]>,
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
        pre_slice: Option<[i128; 2]>,
        predicate: Option<PlSmallStr>,
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
        payload: PlSmallStr,
    },
    SinkMultiple {
        num_inputs: u64,
    },
    Slice {
        offset: i128,
        len: u64,
    },
    Sort {
        by_exprs: Vec<PlSmallStr>,
        slice: Option<[i128; 2]>,
        descending: Vec<bool>,
        nulls_last: Vec<bool>,
        multithreaded: bool,
        maintain_order: bool,
        limit: Option<u64>,
    },
    Union {
        maintain_order: bool,
        parallel: bool,
        rechunk: bool,
        slice: Option<[i128; 2]>,
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
        slice: Option<[i128; 2]>,
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
        slice: Option<[i128; 2]>,
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
        slice: Option<[i128; 2]>,
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
    PlaceholderScan {
        id: usize,
        schema_names: Vec<PlSmallStr>,
    },
}
