use polars_utils::unique_id::UniqueId;
use serde::{Deserialize, Serialize};

use super::*;

type DataFrameKey = usize;
type DslPlanKey = usize;

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub(crate) enum SerializableDslPlanNode {
    #[cfg(feature = "python")]
    PythonScan {
        options: crate::dsl::python_dsl::PythonOptionsDsl,
    },
    Filter {
        input: DslPlanKey,
        predicate: Expr,
    },
    Cache {
        input: DslPlanKey,
        id: UniqueId,
    },
    Scan {
        sources: ScanSources,
        unified_scan_args: Box<UnifiedScanArgs>,
        scan_type: Box<FileScanDsl>,
    },
    DataFrameScan {
        df: DataFrameKey,
        schema: SchemaRef,
    },
    Select {
        expr: Vec<Expr>,
        input: DslPlanKey,
        options: ProjectionOptions,
    },
    GroupBy {
        input: DslPlanKey,
        keys: Vec<Expr>,
        aggs: Vec<Expr>,
        maintain_order: bool,
        options: Arc<GroupbyOptions>,
        apply: Option<(PlanCallback<DataFrame, DataFrame>, SchemaRef)>,
    },
    Join {
        input_left: DslPlanKey,
        input_right: DslPlanKey,
        left_on: Vec<Expr>,
        right_on: Vec<Expr>,
        predicates: Vec<Expr>,
        options: Arc<JoinOptions>,
    },
    HStack {
        input: DslPlanKey,
        exprs: Vec<Expr>,
        options: ProjectionOptions,
    },
    MatchToSchema {
        input: DslPlanKey,
        match_schema: SchemaRef,
        per_column: Arc<[MatchToSchemaPerColumn]>,
        extra_columns: ExtraColumnsPolicy,
    },
    PipeWithSchema {
        input: DslPlanKey,
        callback: PlanCallback<(DslPlan, Schema), DslPlan>,
    },
    Distinct {
        input: DslPlanKey,
        options: DistinctOptionsDSL,
    },
    Sort {
        input: DslPlanKey,
        by_column: Vec<Expr>,
        slice: Option<(i64, usize)>,
        sort_options: SortMultipleOptions,
    },
    Slice {
        input: DslPlanKey,
        offset: i64,
        len: IdxSize,
    },
    MapFunction {
        input: DslPlanKey,
        function: DslFunction,
    },
    Union {
        inputs: Vec<DslPlan>,
        args: UnionArgs,
    },
    HConcat {
        inputs: Vec<DslPlan>,
        options: HConcatOptions,
    },
    ExtContext {
        input: DslPlanKey,
        contexts: Vec<DslPlan>,
    },
    Sink {
        input: DslPlanKey,
        payload: SinkType,
    },
    SinkMultiple {
        inputs: Vec<DslPlan>,
    },
    #[cfg(feature = "merge_sorted")]
    MergeSorted {
        input_left: DslPlanKey,
        input_right: DslPlanKey,
        key: PlSmallStr,
    },
    IR {
        dsl: DslPlanKey,
        version: u32,
    },
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub(crate) struct SerializableDslPlan {
    pub(crate) dataframes: Vec<DataFrame>,
    pub(crate) dsl_plans: Vec<SerializableDslPlanNode>,
}

struct Arenas {
    dataframe_arena: Vec<DataFrame>,
    dataframe_key_table: PlHashMap<*const DataFrame, DataFrameKey>,
    dsl_plan_arena: Vec<SerializableDslPlanNode>,
    dsl_plan_key_table: PlHashMap<*const DslPlan, DslPlanKey>,
}

impl From<&DslPlan> for SerializableDslPlan {
    fn from(plan: &DslPlan) -> Self {
        let mut arenas = Arenas {
            dataframe_arena: Vec::new(),
            dataframe_key_table: PlHashMap::default(),
            dsl_plan_arena: Vec::new(),
            dsl_plan_key_table: PlHashMap::default(),
        };
        let root_dsl_plan = convert_dsl_plan_to_serializable_plan(plan, &mut arenas);
        arenas.dsl_plan_arena.push(root_dsl_plan);
        SerializableDslPlan {
            dsl_plans: arenas.dsl_plan_arena,
            dataframes: arenas.dataframe_arena,
        }
    }
}

fn convert_dsl_plan_to_serializable_plan(
    plan: &DslPlan,
    arenas: &mut Arenas,
) -> SerializableDslPlanNode {
    use {DslPlan as DP, SerializableDslPlanNode as SP};

    match plan {
        #[cfg(feature = "python")]
        DP::PythonScan { options } => SP::PythonScan {
            options: options.clone(),
        },
        DP::Filter { input, predicate } => SP::Filter {
            input: dsl_plan_key(input, arenas),
            predicate: predicate.clone(),
        },
        DP::Cache { input, id } => SP::Cache {
            input: dsl_plan_key(input, arenas),
            id: *id,
        },
        DP::Scan {
            sources,
            unified_scan_args,
            scan_type,
            cached_ir: _,
        } => SP::Scan {
            sources: sources.clone(),
            unified_scan_args: unified_scan_args.clone(),
            scan_type: scan_type.clone(),
        },
        DP::DataFrameScan { df, schema } => SP::DataFrameScan {
            df: dataframe_key(df, arenas),
            schema: schema.clone(),
        },
        DP::Select {
            expr,
            input,
            options,
        } => SP::Select {
            expr: expr.clone(),
            input: dsl_plan_key(input, arenas),
            options: *options,
        },
        DP::GroupBy {
            input,
            keys,
            aggs,
            maintain_order,
            options,
            apply,
        } => SP::GroupBy {
            input: dsl_plan_key(input, arenas),
            keys: keys.clone(),
            aggs: aggs.clone(),
            maintain_order: *maintain_order,
            options: options.clone(),
            apply: apply.clone(),
        },
        DP::Join {
            input_left,
            input_right,
            left_on,
            right_on,
            predicates,
            options,
        } => SP::Join {
            input_left: dsl_plan_key(input_left, arenas),
            input_right: dsl_plan_key(input_right, arenas),
            left_on: left_on.clone(),
            right_on: right_on.clone(),
            predicates: predicates.clone(),
            options: options.clone(),
        },
        DP::HStack {
            input,
            exprs,
            options,
        } => SP::HStack {
            input: dsl_plan_key(input, arenas),
            exprs: exprs.clone(),
            options: *options,
        },
        DP::MatchToSchema {
            input,
            match_schema,
            per_column,
            extra_columns,
        } => SP::MatchToSchema {
            input: dsl_plan_key(input, arenas),
            match_schema: match_schema.clone(),
            per_column: per_column.clone(),
            extra_columns: *extra_columns,
        },
        DP::PipeWithSchema { input, callback } => SP::PipeWithSchema {
            input: dsl_plan_key(input, arenas),
            callback: callback.clone(),
        },
        DP::Distinct { input, options } => SP::Distinct {
            input: dsl_plan_key(input, arenas),
            options: options.clone(),
        },
        DP::Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => SP::Sort {
            input: dsl_plan_key(input, arenas),
            by_column: by_column.clone(),
            slice: *slice,
            sort_options: sort_options.clone(),
        },
        DP::Slice { input, offset, len } => SP::Slice {
            input: dsl_plan_key(input, arenas),
            offset: *offset,
            len: *len,
        },
        DP::MapFunction { input, function } => SP::MapFunction {
            input: dsl_plan_key(input, arenas),
            function: function.clone(),
        },
        DP::Union { inputs, args } => SP::Union {
            inputs: inputs.clone(),
            args: *args,
        },
        DP::HConcat { inputs, options } => SP::HConcat {
            inputs: inputs.clone(),
            options: *options,
        },
        DP::ExtContext { input, contexts } => SP::ExtContext {
            input: dsl_plan_key(input, arenas),
            contexts: contexts.clone(),
        },
        DP::Sink { input, payload } => SP::Sink {
            input: dsl_plan_key(input, arenas),
            payload: payload.clone(),
        },
        DP::SinkMultiple { inputs } => SP::SinkMultiple {
            inputs: inputs.clone(),
        },
        #[cfg(feature = "merge_sorted")]
        DP::MergeSorted {
            input_left,
            input_right,
            key,
        } => SP::MergeSorted {
            input_left: dsl_plan_key(input_left, arenas),
            input_right: dsl_plan_key(input_right, arenas),
            key: key.clone(),
        },
        DP::IR {
            dsl,
            version,
            node: _,
        } => SP::IR {
            dsl: dsl_plan_key(dsl, arenas),
            version: *version,
        },
    }
}

fn dataframe_key(df: &Arc<DataFrame>, arenas: &mut Arenas) -> DataFrameKey {
    let ptr = Arc::as_ptr(df);
    if let Some(key) = arenas.dataframe_key_table.get(&ptr) {
        *key
    } else {
        let key = arenas.dataframe_arena.len();
        arenas.dataframe_arena.push((**df).clone());
        arenas.dataframe_key_table.insert(ptr, key);
        key
    }
}

fn dsl_plan_key(plan: &Arc<DslPlan>, arenas: &mut Arenas) -> DslPlanKey {
    let ptr = Arc::as_ptr(plan);
    if let Some(key) = arenas.dsl_plan_key_table.get(&ptr) {
        *key
    } else {
        let serializable = convert_dsl_plan_to_serializable_plan(plan, arenas);
        let key = arenas.dsl_plan_arena.len();
        arenas.dsl_plan_arena.push(serializable);
        arenas.dsl_plan_key_table.insert(ptr, key);
        key
    }
}

impl TryFrom<SerializableDslPlan> for DslPlan {
    type Error = PolarsError;

    fn try_from(ser_dsl_plan: SerializableDslPlan) -> Result<Self, Self::Error> {
        let dataframes = ser_dsl_plan
            .dataframes
            .into_iter()
            .map(Arc::new)
            .collect::<Vec<_>>();
        let mut dsl_plans = ser_dsl_plan.dsl_plans;
        let dsl_plan_root = dsl_plans.pop().ok_or(polars_err!(
            ComputeError: "Serialized DSL plan contains no nodes"
        ))?;
        let mut de_dsl_plans: Vec<Arc<DslPlan>> = Vec::with_capacity(dsl_plans.len());
        for node in dsl_plans.iter() {
            de_dsl_plans.push(Arc::new(try_convert_serializable_plan_to_dsl_plan(
                node,
                &dataframes,
                &de_dsl_plans,
            )?));
        }
        try_convert_serializable_plan_to_dsl_plan(&dsl_plan_root, &dataframes, &de_dsl_plans)
    }
}

fn try_convert_serializable_plan_to_dsl_plan(
    node: &SerializableDslPlanNode,
    dataframes: &[Arc<DataFrame>],
    dsl_plans: &[Arc<DslPlan>],
) -> Result<DslPlan, PolarsError> {
    use {DslPlan as DP, SerializableDslPlanNode as SP};

    match node {
        #[cfg(feature = "python")]
        SP::PythonScan { options } => Ok(DP::PythonScan {
            options: options.clone(),
        }),
        SP::Filter { input, predicate } => Ok(DP::Filter {
            input: get_dsl_plan(*input, dsl_plans)?,
            predicate: predicate.clone(),
        }),
        SP::Cache { input, id } => Ok(DP::Cache {
            input: get_dsl_plan(*input, dsl_plans)?,
            id: *id,
        }),
        SP::Scan {
            sources,
            unified_scan_args,
            scan_type,
        } => Ok(DP::Scan {
            sources: sources.clone(),
            unified_scan_args: unified_scan_args.clone(),
            scan_type: scan_type.clone(),
            cached_ir: Default::default(),
        }),
        SP::DataFrameScan { df, schema } => Ok(DP::DataFrameScan {
            df: get_dataframe(*df, dataframes)?,
            schema: schema.clone(),
        }),
        SP::Select {
            expr,
            input,
            options,
        } => Ok(DP::Select {
            expr: expr.clone(),
            input: get_dsl_plan(*input, dsl_plans)?,
            options: *options,
        }),
        SP::GroupBy {
            input,
            keys,
            aggs,
            maintain_order,
            options,
            apply,
        } => Ok(DP::GroupBy {
            input: get_dsl_plan(*input, dsl_plans)?,
            keys: keys.clone(),
            aggs: aggs.clone(),
            maintain_order: *maintain_order,
            options: options.clone(),
            apply: apply.clone(),
        }),
        SP::Join {
            input_left,
            input_right,
            left_on,
            right_on,
            predicates,
            options,
        } => Ok(DP::Join {
            input_left: get_dsl_plan(*input_left, dsl_plans)?,
            input_right: get_dsl_plan(*input_right, dsl_plans)?,
            left_on: left_on.clone(),
            right_on: right_on.clone(),
            predicates: predicates.clone(),
            options: options.clone(),
        }),
        SP::HStack {
            input,
            exprs,
            options,
        } => Ok(DP::HStack {
            input: get_dsl_plan(*input, dsl_plans)?,
            exprs: exprs.clone(),
            options: *options,
        }),
        SP::MatchToSchema {
            input,
            match_schema,
            per_column,
            extra_columns,
        } => Ok(DP::MatchToSchema {
            input: get_dsl_plan(*input, dsl_plans)?,
            match_schema: match_schema.clone(),
            per_column: per_column.clone(),
            extra_columns: *extra_columns,
        }),
        SP::PipeWithSchema { input, callback } => Ok(DP::PipeWithSchema {
            input: get_dsl_plan(*input, dsl_plans)?,
            callback: callback.clone(),
        }),
        SP::Distinct { input, options } => Ok(DP::Distinct {
            input: get_dsl_plan(*input, dsl_plans)?,
            options: options.clone(),
        }),
        SP::Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => Ok(DP::Sort {
            input: get_dsl_plan(*input, dsl_plans)?,
            by_column: by_column.clone(),
            slice: *slice,
            sort_options: sort_options.clone(),
        }),
        SP::Slice { input, offset, len } => Ok(DP::Slice {
            input: get_dsl_plan(*input, dsl_plans)?,
            offset: *offset,
            len: *len,
        }),
        SP::MapFunction { input, function } => Ok(DP::MapFunction {
            input: get_dsl_plan(*input, dsl_plans)?,
            function: function.clone(),
        }),
        SP::Union { inputs, args } => Ok(DP::Union {
            inputs: inputs.clone(),
            args: *args,
        }),
        SP::HConcat { inputs, options } => Ok(DP::HConcat {
            inputs: inputs.clone(),
            options: *options,
        }),
        SP::ExtContext { input, contexts } => Ok(DP::ExtContext {
            input: get_dsl_plan(*input, dsl_plans)?,
            contexts: contexts.clone(),
        }),
        SP::Sink { input, payload } => Ok(DP::Sink {
            input: get_dsl_plan(*input, dsl_plans)?,
            payload: payload.clone(),
        }),
        SP::SinkMultiple { inputs } => Ok(DP::SinkMultiple {
            inputs: inputs.clone(),
        }),
        #[cfg(feature = "merge_sorted")]
        SP::MergeSorted {
            input_left,
            input_right,
            key,
        } => Ok(DP::MergeSorted {
            input_left: get_dsl_plan(*input_left, dsl_plans)?,
            input_right: get_dsl_plan(*input_right, dsl_plans)?,
            key: key.clone(),
        }),
        SP::IR {
            dsl: dsl_key,
            version,
        } => Ok(DP::IR {
            dsl: get_dsl_plan(*dsl_key, dsl_plans)?,
            version: *version,
            node: Default::default(),
        }),
    }
}

fn get_dataframe(key: usize, dataframes: &[Arc<DataFrame>]) -> Result<Arc<DataFrame>, PolarsError> {
    Ok(dataframes
        .get(key)
        .ok_or(polars_err!(
            ComputeError: "Could not find DataFrame node at index {} in serialized plan", key
        ))?
        .clone())
}

fn get_dsl_plan(key: usize, dsl_plans: &[Arc<DslPlan>]) -> Result<Arc<DslPlan>, PolarsError> {
    Ok(dsl_plans
        .get(key)
        .ok_or(polars_err!(
            ComputeError: "Could not find DslPlan node at index {} in serialized plan", key
        ))?
        .clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsl_plan_serialization() {
        let name = || "a".into();
        let df = Arc::new(
            DataFrame::new(vec![Column::new(name(), Series::new(name(), &[1, 2, 3]))]).unwrap(),
        );
        let dfscan = Arc::new(DslPlan::DataFrameScan {
            df: df.clone(),
            schema: df.schema().clone(),
        });
        let join_options = JoinOptions {
            allow_parallel: true,
            force_parallel: false,
            ..Default::default()
        };
        let lf = DslPlan::Join {
            input_left: dfscan.clone(),
            input_right: dfscan,
            left_on: vec![Expr::Column(name())],
            right_on: vec![Expr::Column(name())],
            predicates: Default::default(),
            options: Arc::new(join_options),
        };
        let mut buffer: Vec<u8> = Vec::new();
        lf.serialize_versioned(&mut buffer, Default::default())
            .unwrap();
        let mut reader: &[u8] = &buffer;
        let deserialized = DslPlan::deserialize_versioned(&mut reader).unwrap();
        assert_eq!(format!("{:?}", lf), format!("{:?}", deserialized));
    }
}
