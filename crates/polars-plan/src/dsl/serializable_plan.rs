use polars_utils::unique_id::UniqueId;
use recursive::recursive;
use serde::{Deserialize, Serialize};
use slotmap::{SecondaryMap, SlotMap, new_key_type};

use super::*;

new_key_type! {
    /// A key type for identifying DataFrame nodes in a serialized DSL plan.
    pub(crate) struct DataFrameKey;

    /// A key type for identifying DslPlan nodes in a serialized DSL plan.
    pub(crate) struct DslPlanKey;
}

/// A representation of DslPlan that does not contain any `Arc` pointers, and
/// instead uses indices to refer to DataFrames and other DslPlan nodes.
///
/// This data structure mirrors the `DslPlan` enum, but uses `DataFrameKey` and
/// `DslPlanKey` to refer to DataFrames and other DslPlan nodes, respectively.
/// We it like this, because serde does not support the keeping of a global
/// state during (de)serialization.  Instead, we do a manual conversion to a
/// serde-compatible representation, and then let serde handle the rest.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct SerializableDslPlan {
    pub(crate) root: DslPlanKey,
    pub(crate) dataframes: SlotMap<DataFrameKey, DataFrameSerdeWrap>,
    pub(crate) dsl_plans: SlotMap<DslPlanKey, SerializableDslPlanNode>,
}

#[derive(Debug, Serialize, Deserialize)]
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
        callback: PlanCallback<(DslPlan, SchemaRef), DslPlan>,
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
        inputs: Vec<SerializableDslPlanNode>,
        args: UnionArgs,
    },
    HConcat {
        inputs: Vec<SerializableDslPlanNode>,
        options: HConcatOptions,
    },
    ExtContext {
        input: DslPlanKey,
        contexts: Vec<SerializableDslPlanNode>,
    },
    Sink {
        input: DslPlanKey,
        payload: SinkType,
    },
    SinkMultiple {
        inputs: Vec<SerializableDslPlanNode>,
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

#[derive(Debug, Default)]
struct SerializeArenas {
    dataframes: SlotMap<DataFrameKey, DataFrameSerdeWrap>,
    dataframes_keys_table: PlIndexMap<*const DataFrame, DataFrameKey>,
    dsl_plans: SlotMap<DslPlanKey, SerializableDslPlanNode>,
    dsl_plans_keys_table: PlIndexMap<*const DslPlan, DslPlanKey>,
}

impl From<&DslPlan> for SerializableDslPlan {
    fn from(plan: &DslPlan) -> Self {
        let mut arenas = SerializeArenas::default();
        let root_dsl_plan = convert_dsl_plan_to_serializable_plan(plan, &mut arenas);

        let root_key = arenas.dsl_plans.insert(root_dsl_plan);
        SerializableDslPlan {
            root: root_key,
            dataframes: arenas.dataframes,
            dsl_plans: arenas.dsl_plans,
        }
    }
}

#[recursive]
fn convert_dsl_plan_to_serializable_plan(
    plan: &DslPlan,
    arenas: &mut SerializeArenas,
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
            inputs: inputs
                .iter()
                .map(|p| convert_dsl_plan_to_serializable_plan(p, arenas))
                .collect(),
            args: *args,
        },
        DP::HConcat { inputs, options } => SP::HConcat {
            inputs: inputs
                .iter()
                .map(|p| convert_dsl_plan_to_serializable_plan(p, arenas))
                .collect(),
            options: *options,
        },
        DP::ExtContext { input, contexts } => SP::ExtContext {
            input: dsl_plan_key(input, arenas),
            contexts: contexts
                .iter()
                .map(|p| convert_dsl_plan_to_serializable_plan(p, arenas))
                .collect(),
        },
        DP::Sink { input, payload } => SP::Sink {
            input: dsl_plan_key(input, arenas),
            payload: payload.clone(),
        },
        DP::SinkMultiple { inputs } => SP::SinkMultiple {
            inputs: inputs
                .iter()
                .map(|p| convert_dsl_plan_to_serializable_plan(p, arenas))
                .collect(),
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

fn dataframe_key(df: &Arc<DataFrame>, arenas: &mut SerializeArenas) -> DataFrameKey {
    let ptr = Arc::as_ptr(df);
    if let Some(key) = arenas.dataframes_keys_table.get(&ptr) {
        *key
    } else {
        let key = arenas.dataframes.insert(DataFrameSerdeWrap(df.clone()));
        arenas.dataframes_keys_table.insert(ptr, key);
        key
    }
}

fn dsl_plan_key(plan: &Arc<DslPlan>, arenas: &mut SerializeArenas) -> DslPlanKey {
    let ptr = Arc::as_ptr(plan);
    if let Some(key) = arenas.dsl_plans_keys_table.get(&ptr) {
        *key
    } else {
        let ser_plan = convert_dsl_plan_to_serializable_plan(plan, arenas);
        let key = arenas.dsl_plans.insert(ser_plan);
        arenas.dsl_plans_keys_table.insert(ptr, key);
        key
    }
}

#[derive(Debug, Default)]
struct DeserializeArenas {
    dataframes: SecondaryMap<DataFrameKey, DataFrameSerdeWrap>,
    dsl_plans: SecondaryMap<DslPlanKey, Arc<DslPlan>>,
}

impl TryFrom<&SerializableDslPlan> for DslPlan {
    type Error = PolarsError;

    fn try_from(ser_dsl_plan: &SerializableDslPlan) -> Result<Self, Self::Error> {
        let mut arenas = DeserializeArenas::default();
        let root = ser_dsl_plan
            .dsl_plans
            .get(ser_dsl_plan.root)
            .ok_or(polars_err!(ComputeError: "Could not find root DslPlan in serialized plan"))?;
        try_convert_serializable_plan_to_dsl_plan(root, ser_dsl_plan, &mut arenas)
    }
}

#[recursive]
fn try_convert_serializable_plan_to_dsl_plan(
    node: &SerializableDslPlanNode,
    ser_dsl_plan: &SerializableDslPlan,
    arenas: &mut DeserializeArenas,
) -> Result<DslPlan, PolarsError> {
    use {DslPlan as DP, SerializableDslPlanNode as SP};

    match node {
        #[cfg(feature = "python")]
        SP::PythonScan { options } => Ok(DP::PythonScan {
            options: options.clone(),
        }),
        SP::Filter { input, predicate } => Ok(DP::Filter {
            input: get_dsl_plan(*input, ser_dsl_plan, arenas)?,
            predicate: predicate.clone(),
        }),
        SP::Cache { input, id } => Ok(DP::Cache {
            input: get_dsl_plan(*input, ser_dsl_plan, arenas)?,
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
            df: get_dataframe(*df, ser_dsl_plan, arenas)?,
            schema: schema.clone(),
        }),
        SP::Select {
            expr,
            input,
            options,
        } => Ok(DP::Select {
            expr: expr.clone(),
            input: get_dsl_plan(*input, ser_dsl_plan, arenas)?,
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
            input: get_dsl_plan(*input, ser_dsl_plan, arenas)?,
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
            input_left: get_dsl_plan(*input_left, ser_dsl_plan, arenas)?,
            input_right: get_dsl_plan(*input_right, ser_dsl_plan, arenas)?,
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
            input: get_dsl_plan(*input, ser_dsl_plan, arenas)?,
            exprs: exprs.clone(),
            options: *options,
        }),
        SP::MatchToSchema {
            input,
            match_schema,
            per_column,
            extra_columns,
        } => Ok(DP::MatchToSchema {
            input: get_dsl_plan(*input, ser_dsl_plan, arenas)?,
            match_schema: match_schema.clone(),
            per_column: per_column.clone(),
            extra_columns: *extra_columns,
        }),
        SP::PipeWithSchema { input, callback } => Ok(DP::PipeWithSchema {
            input: get_dsl_plan(*input, ser_dsl_plan, arenas)?,
            callback: callback.clone(),
        }),
        SP::Distinct { input, options } => Ok(DP::Distinct {
            input: get_dsl_plan(*input, ser_dsl_plan, arenas)?,
            options: options.clone(),
        }),
        SP::Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => Ok(DP::Sort {
            input: get_dsl_plan(*input, ser_dsl_plan, arenas)?,
            by_column: by_column.clone(),
            slice: *slice,
            sort_options: sort_options.clone(),
        }),
        SP::Slice { input, offset, len } => Ok(DP::Slice {
            input: get_dsl_plan(*input, ser_dsl_plan, arenas)?,
            offset: *offset,
            len: *len,
        }),
        SP::MapFunction { input, function } => Ok(DP::MapFunction {
            input: get_dsl_plan(*input, ser_dsl_plan, arenas)?,
            function: function.clone(),
        }),
        SP::Union { inputs, args } => Ok(DP::Union {
            inputs: inputs
                .iter()
                .map(|node| try_convert_serializable_plan_to_dsl_plan(node, ser_dsl_plan, arenas))
                .collect::<Result<Vec<_>, _>>()?,
            args: *args,
        }),
        SP::HConcat { inputs, options } => Ok(DP::HConcat {
            inputs: inputs
                .iter()
                .map(|node| try_convert_serializable_plan_to_dsl_plan(node, ser_dsl_plan, arenas))
                .collect::<Result<Vec<_>, _>>()?,
            options: *options,
        }),
        SP::ExtContext { input, contexts } => Ok(DP::ExtContext {
            input: get_dsl_plan(*input, ser_dsl_plan, arenas)?,
            contexts: contexts
                .iter()
                .map(|node| try_convert_serializable_plan_to_dsl_plan(node, ser_dsl_plan, arenas))
                .collect::<Result<Vec<_>, _>>()?,
        }),
        SP::Sink { input, payload } => Ok(DP::Sink {
            input: get_dsl_plan(*input, ser_dsl_plan, arenas)?,
            payload: payload.clone(),
        }),
        SP::SinkMultiple { inputs } => Ok(DP::SinkMultiple {
            inputs: inputs
                .iter()
                .map(|node| try_convert_serializable_plan_to_dsl_plan(node, ser_dsl_plan, arenas))
                .collect::<Result<Vec<_>, _>>()?,
        }),
        #[cfg(feature = "merge_sorted")]
        SP::MergeSorted {
            input_left,
            input_right,
            key,
        } => Ok(DP::MergeSorted {
            input_left: get_dsl_plan(*input_left, ser_dsl_plan, arenas)?,
            input_right: get_dsl_plan(*input_right, ser_dsl_plan, arenas)?,
            key: key.clone(),
        }),
        SP::IR {
            dsl: dsl_key,
            version,
        } => Ok(DP::IR {
            dsl: get_dsl_plan(*dsl_key, ser_dsl_plan, arenas)?,
            version: *version,
            node: Default::default(),
        }),
    }
}

fn get_dataframe(
    key: DataFrameKey,
    ser_dsl_plan: &SerializableDslPlan,
    arenas: &mut DeserializeArenas,
) -> Result<Arc<DataFrame>, PolarsError> {
    if let Some(df) = arenas.dataframes.get(key) {
        Ok(df.0.clone())
    } else {
        let df = ser_dsl_plan.dataframes.get(key).ok_or(polars_err!(
            ComputeError: "Could not find DataFrame at index {:?} in serialized plan", key
        ))?;
        arenas.dataframes.insert(key, df.clone());
        Ok(df.0.clone())
    }
}

fn get_dsl_plan(
    key: DslPlanKey,
    ser_dsl_plan: &SerializableDslPlan,
    arenas: &mut DeserializeArenas,
) -> Result<Arc<DslPlan>, PolarsError> {
    if let Some(dsl_plan) = arenas.dsl_plans.get(key) {
        Ok(dsl_plan.clone())
    } else {
        let node = ser_dsl_plan.dsl_plans.get(key).ok_or(polars_err!(
            ComputeError: "Could not find DslPlan node at index {:?} in serialized plan", key
        ))?;
        let dsl_plan = try_convert_serializable_plan_to_dsl_plan(node, ser_dsl_plan, arenas)?;
        let arc_dsl_plan = Arc::new(dsl_plan);
        arenas.dsl_plans.insert(key, arc_dsl_plan.clone());
        Ok(arc_dsl_plan)
    }
}

/// Serialization wrapper that splits large serialized byte values into chunks.
#[derive(Debug, Clone)]
pub(crate) struct DataFrameSerdeWrap(Arc<DataFrame>);

#[cfg(feature = "serde")]
mod _serde_impl {
    use std::sync::Arc;

    use polars_core::frame::DataFrame;
    use polars_utils::chunked_bytes_cursor::FixedSizeChunkedBytesCursor;
    use serde::de::Error;
    use serde::{Deserialize, Serialize};

    use super::DataFrameSerdeWrap;

    fn max_byte_slice_len() -> usize {
        std::env::var("POLARS_SERIALIZE_LAZYFRAME_MAX_BYTE_SLICE_LEN")
            .as_deref()
            .map_or(
                usize::try_from(u32::MAX).unwrap(), // Limit for rmp_serde
                |x| x.parse().unwrap(),
            )
    }

    impl Serialize for DataFrameSerdeWrap {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            use serde::ser::Error;

            let mut bytes: Vec<u8> = vec![];
            self.0
                .as_ref()
                .clone()
                .serialize_into_writer(&mut bytes)
                .map_err(S::Error::custom)?;

            serializer.collect_seq(bytes.chunks(max_byte_slice_len()))
        }
    }

    impl<'de> Deserialize<'de> for DataFrameSerdeWrap {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let bytes: Vec<Vec<u8>> = Vec::deserialize(deserializer)?;

            let result = match bytes.as_slice() {
                [v] => DataFrame::deserialize_from_reader(&mut std::io::Cursor::new(v.as_slice())),
                _ => DataFrame::deserialize_from_reader(
                    &mut FixedSizeChunkedBytesCursor::try_new(bytes.as_slice()).unwrap(),
                ),
            };

            result
                .map(|x| DataFrameSerdeWrap(Arc::new(x)))
                .map_err(D::Error::custom)
        }
    }
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
        assert_eq!(format!("{lf:?}"), format!("{deserialized:?}"));
    }
}
