use std::num::NonZeroUsize;

use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_ops::frame::MaintainOrderJoin;
use polars_ops::prelude::{JoinCoalesce, JoinValidation};
use polars_utils::pl_str::PlSmallStr;

#[derive(serde::Serialize, serde::Deserialize, Debug)]
#[cfg_attr(
    feature = "physical_plan_visualization_schema",
    derive(schemars::JsonSchema)
)]
pub struct PhysicalPlanVisualizationData {
    pub title: PlSmallStr,
    /// Number of nodes from the start of `nodes` that are root nodes.
    pub num_roots: u64,
    pub nodes: Vec<PhysNodeInfo>,
    pub edges: Vec<Edge>,
}

impl PhysicalPlanVisualizationData {
    pub fn to_json(&self) -> polars_error::PolarsResult<String> {
        serde_json::to_string(self).map_err(polars_error::to_compute_err)
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Default)]
#[cfg_attr(
    feature = "physical_plan_visualization_schema",
    derive(schemars::JsonSchema)
)]
pub struct PhysNodeInfo {
    pub id: u64,
    pub title: PlSmallStr,
    pub properties: PhysNodeProperties,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Default)]
#[cfg_attr(
    feature = "physical_plan_visualization_schema",
    derive(schemars::JsonSchema)
)]
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

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
#[derive(Default, Debug, strum_macros::IntoStaticStr)]
#[cfg_attr(
    feature = "physical_plan_visualization_schema",
    derive(schemars::JsonSchema)
)]
pub enum PhysNodeProperties {
    #[default]
    Default,
    CallbackSink {
        callback_function: PlSmallStr,
        maintain_order: bool,
        chunk_size: Option<NonZeroUsize>,
    },
    DynamicSlice,
    FileSink {
        target: PlSmallStr,
        sync_on_close: SyncOnCloseType,
        maintain_order: bool,
        mkdir: bool,
        file_type: PlSmallStr,
    },
    Filter {
        predicate: PlSmallStr,
    },
    GatherEvery {
        n: u64,
        offset: u64,
    },
    GroupBy {
        keys: Vec<PlSmallStr>,
        aggs: Vec<PlSmallStr>,
    },
    #[cfg(feature = "dynamic_group_by")]
    RollingGroupBy {
        index_column: PlSmallStr,
        period: PlSmallStr,
        offset: PlSmallStr,
        closed_window: PlSmallStr,
        aggs: Vec<PlSmallStr>,
    },
    InMemoryMap {
        format_str: PlSmallStr,
    },
    InMemorySink,
    InMemorySource {
        n_rows: u64,
        schema_names: Vec<PlSmallStr>,
    },
    InputIndependentSelect {
        selectors: Vec<PlSmallStr>,
    },
    // Joins
    CrossJoin {
        maintain_order: MaintainOrderJoin,
        suffix: Option<PlSmallStr>,
    },
    EquiJoin {
        how: PlSmallStr,
        left_on: Vec<PlSmallStr>,
        right_on: Vec<PlSmallStr>,
        nulls_equal: bool,
        coalesce: JoinCoalesce,
        maintain_order: MaintainOrderJoin,
        validation: JoinValidation,
        suffix: Option<PlSmallStr>,
    },
    InMemoryJoin {
        how: PlSmallStr,
        left_on: Vec<PlSmallStr>,
        right_on: Vec<PlSmallStr>,
        nulls_equal: bool,
        coalesce: JoinCoalesce,
        maintain_order: MaintainOrderJoin,
        validation: JoinValidation,
        suffix: Option<PlSmallStr>,
        slice: Option<(i64, u64)>,
    },
    InMemoryAsOfJoin {
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
    InMemoryIEJoin {
        left_on: Vec<PlSmallStr>,
        right_on: Vec<PlSmallStr>,
        inequality_operators: Vec<polars_ops::frame::InequalityOperator>,
        suffix: Option<PlSmallStr>,
        slice: Option<(i64, u64)>,
    },
    Map {
        display_str: PlSmallStr,
    },
    MultiScan {
        scan_type: PlSmallStr,
        num_sources: u64,
        first_source: Option<PlSmallStr>,
        projected_file_columns: Vec<PlSmallStr>,
        file_projection_builder_type: PlSmallStr,
        row_index_name: Option<PlSmallStr>,
        row_index_offset: Option<u64>,
        pre_slice: Option<(i64, u64)>,
        predicate: Option<PlSmallStr>,
        has_table_statistics: bool,
        include_file_paths: Option<PlSmallStr>,
        deletion_files_type: Option<PlSmallStr>,
        hive_columns: Option<Vec<PlSmallStr>>,
    },
    Multiplexer,
    NegativeSlice {
        offset: i64,
        length: u64,
    },
    OrderedUnion {
        num_inputs: u64,
    },
    PartitionSink {
        base_path: PlSmallStr,
        file_path_callback: Option<PlSmallStr>,
        partition_variant: PlSmallStr,
        partition_variant_max_size: Option<u64>,
        partition_variant_key_exprs: Option<Vec<PlSmallStr>>,
        partition_variant_include_key: Option<bool>,
        file_type: PlSmallStr,
        per_partition_sort_exprs: Option<Vec<PlSmallStr>>,
        per_partition_sort_descending: Option<Vec<bool>>,
        per_partition_sort_nulls_last: Option<Vec<bool>>,
        finish_callback: Option<PlSmallStr>,
        sync_on_close: SyncOnCloseType,
        maintain_order: bool,
        mkdir: bool,
    },
    PeakMin,
    PeakMax,
    Reduce {
        exprs: Vec<PlSmallStr>,
    },
    Repeat,
    Rle,
    RleId,
    Select {
        selectors: Vec<PlSmallStr>,
        extend_original: bool,
    },
    Shift {
        has_fill: bool,
    },
    SimpleProjection {
        columns: Vec<PlSmallStr>,
    },
    SinkMultiple {
        num_sinks: u64,
    },
    Sort {
        by_exprs: Vec<PlSmallStr>,
        slice: Option<(i64, u64)>,
        descending: Vec<bool>,
        nulls_last: Vec<bool>,
        multithreaded: bool,
        maintain_order: bool,
        limit: Option<u64>,
    },
    Slice {
        offset: i64,
        length: u64,
    },
    TopK {
        by_exprs: Vec<PlSmallStr>,
        reverse: Vec<bool>,
        nulls_last: Vec<bool>,
    },
    WithRowIndex {
        name: PlSmallStr,
        offset: Option<u64>,
    },
    Zip {
        num_inputs: u64,
        null_extend: bool,
    },
    //
    // Feature gated
    //
    #[cfg(feature = "cum_agg")]
    CumAgg {
        kind: PlSmallStr,
    },
    #[cfg(feature = "ewma")]
    Ewm {
        variant: PlSmallStr,
        alpha: f64,
        adjust: bool,
        bias: bool,
        min_periods: usize,
        ignore_nulls: bool,
    },
    #[cfg(feature = "semi_anti_join")]
    SemiAntiJoin {
        left_on: Vec<PlSmallStr>,
        right_on: Vec<PlSmallStr>,
        nulls_equal: bool,
        output_as_bool: bool,
    },
    #[cfg(feature = "merge_sorted")]
    MergeSorted,
    #[cfg(feature = "python")]
    PythonScan {
        scan_source_type: polars_plan::prelude::python_dsl::PythonScanSource,
        n_rows: Option<u64>,
        projection: Option<Vec<PlSmallStr>>,
        predicate: Option<PlSmallStr>,
        schema_names: Vec<PlSmallStr>,
        is_pure: bool,
        validate_schema: bool,
    },
}
