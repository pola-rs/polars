use serde::{Deserialize, Serialize};

use crate::{PredicateFileSkipDescription, PythonPredicateDescription, SortColumnDescription};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct IrNodeDescription {
    pub id: usize,
    pub input_ids: Vec<usize>,
    pub properties: IrPropsDescription,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, strum_macros::IntoStaticStr)]
#[serde(tag = "type")]
pub enum IrPropsDescription {
    Cache {
        id: String,
    },
    DataFrameScan {
        n_rows: usize,
        schema_names: Vec<String>,
    },
    Distinct {
        subset: Option<Vec<String>>,
        maintain_order: bool,
        keep_strategy: String,
        slice: Option<(i64, usize)>,
    },
    ExtContext {
        num_contexts: usize,
        schema_names: Vec<String>,
    },
    Filter {
        predicate: Vec<String>,
    },
    Gather {
        null_on_oob: bool,
    },
    GroupBy {
        keys: Vec<String>,
        aggs: Vec<String>,
        maintain_order: bool,
        slice: Option<(i64, usize)>,
    },
    HConcat {
        num_inputs: usize,
        schema_names: Vec<String>,
        strict: bool,
    },
    HStack {
        exprs: Vec<String>,
        should_broadcast: bool,
    },
    Invalid,
    Join {
        how: String,
        left_on: Vec<String>,
        right_on: Vec<String>,
        nulls_equal: bool,
        coalesce: String,
        maintain_order: String,
        validation: String,
        suffix: Option<String>,
        slice: Option<(i64, usize)>,
    },
    CrossJoin {
        maintain_order: String,
        slice: Option<(i64, usize)>,
        predicate: Option<Vec<String>>,
        suffix: Option<String>,
    },
    MapFunction {
        function: String,
    },
    Scan {
        scan_type: String,
        num_sources: usize,
        first_source: Option<String>,
        file_columns: Option<Vec<String>>,
        projection: Option<Vec<String>>,
        row_index_name: Option<String>,
        row_index_offset: Option<u64>,
        pre_slice: Option<(i64, u64)>,
        predicate: Option<Vec<String>>,
        predicate_file_skip_applied: Option<PredicateFileSkipDescription>,
        has_table_statistics: bool,
        include_file_paths: Option<String>,
        column_mapping_type: Option<String>,
        hive_columns: Option<Vec<String>>,
    },
    Select {
        exprs: Vec<String>,
    },
    SimpleProjection {
        columns: Vec<String>,
    },
    Sink {
        dest: SinkDestDescription,
    },
    SinkMultiple {
        num_inputs: usize,
    },
    Slice {
        offset: i64,
        len: u64,
    },
    Sort {
        sort_columns: Vec<SortColumnDescription>,
        slice: Option<(i64, usize, Option<String>)>,
        maintain_order: bool,
        limit: Option<u64>,
    },
    Union {
        num_inputs: usize,
        maintain_order: bool,
        slice: Option<(i64, usize)>,
    },
    AsOfJoin {
        left_on: Vec<String>,
        right_on: Vec<String>,
        left_by: Option<Vec<String>>,
        right_by: Option<Vec<String>>,
        strategy: String,
        /// [value, dtype_str]
        tolerance: Option<[String; 2]>,
        suffix: Option<String>,
        slice: Option<(i64, usize)>,
        coalesce: String,
        allow_eq: bool,
        check_sortedness: bool,
    },
    IEJoin {
        left_on: Vec<String>,
        right_on: Vec<String>,
        inequality_operators: Vec<String>,
        suffix: Option<String>,
        slice: Option<(i64, usize)>,
    },
    DynamicGroupBy {
        index_column: String,
        aggs: Vec<String>,
        every: String,
        period: String,
        offset: String,
        label: String,
        include_boundaries: bool,
        closed_window: String,
        group_by: Vec<String>,
        start_by: String,
    },
    RollingGroupBy {
        keys: Vec<String>,
        aggs: Vec<String>,
        index_column: String,
        period: String,
        offset: String,
        closed_window: String,
        slice: Option<(i64, usize)>,
    },
    MergeSorted {
        keys: Vec<String>,
        maintain_order: bool,
    },
    PythonScan {
        scan_source_type: String,
        n_rows: Option<usize>,
        projection: Option<Vec<String>>,
        predicate: PythonPredicateDescription,
        schema_names: Vec<String>,
        is_pure: bool,
        validate_schema: bool,
    },
    UnoptimizedDispatch {
        num_inputs: usize,
        operation: String,
    },

    #[default]
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, strum_macros::IntoStaticStr)]
#[serde(tag = "partition_type")]
pub enum PartitioningModelDescription {
    RoundRobin,
    Local,
    Single,
    Broadcast,
    Hash { by: String },
    Range,
}

#[derive(Debug, Clone, Serialize, Deserialize, strum_macros::IntoStaticStr)]
pub enum SinkDestDescription {
    Memory,
    Callback,
    File {
        file_format: String,
        target: String,
    },
    Partitioned {
        file_format: String,
        base_path: String,
    },
}
