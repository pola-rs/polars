use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use hashbrown::hash_map::Entry;
use polars_core::prelude::*;
use polars_core::with_match_physical_integer_polars_type;
#[cfg(feature = "parquet")]
use polars_io::predicates::{PhysicalIoExpr, StatsEvaluator};
use polars_ops::prelude::JoinType;
use polars_plan::prelude::*;

use crate::executors::operators::HstackOperator;
use crate::executors::sinks::group_by::aggregates::convert_to_hash_agg;
use crate::executors::sinks::group_by::GenericGroupby2;
use crate::executors::sinks::*;
use crate::executors::{operators, sources};
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{Operator, Sink as SinkTrait, Source};
use crate::pipeline::PipeLine;

fn exprs_to_physical<F>(
    exprs: &[Node],
    expr_arena: &Arena<AExpr>,
    to_physical: &F,
    schema: Option<&SchemaRef>,
) -> PolarsResult<Vec<Arc<dyn PhysicalPipedExpr>>>
where
    F: Fn(Node, &Arena<AExpr>, Option<&SchemaRef>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    exprs
        .iter()
        .map(|node| to_physical(*node, expr_arena, schema))
        .collect()
}

#[allow(unused_variables)]
fn get_source<F>(
    source: ALogicalPlan,
    operator_objects: &mut Vec<Box<dyn Operator>>,
    expr_arena: &Arena<AExpr>,
    to_physical: &F,
    push_predicate: bool,
    verbose: bool,
) -> PolarsResult<Box<dyn Source>>
where
    F: Fn(Node, &Arena<AExpr>, Option<&SchemaRef>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    use ALogicalPlan::*;
    match source {
        DataFrameScan {
            df,
            projection,
            selection,
            output_schema,
            ..
        } => {
            let mut df = (*df).clone();
            if push_predicate {
                if let Some(predicate) = selection {
                    let predicate = to_physical(predicate, expr_arena, output_schema.as_ref())?;
                    let op = operators::FilterOperator { predicate };
                    let op = Box::new(op) as Box<dyn Operator>;
                    operator_objects.push(op)
                }
                // projection is free
                if let Some(projection) = projection {
                    df = df.select(projection.as_slice())?;
                }
            }
            Ok(Box::new(sources::DataFrameSource::from_df(df)) as Box<dyn Source>)
        },
        Scan {
            paths,
            file_info,
            file_options,
            predicate,
            output_schema,
            scan_type,
        } => {
            // Add predicate to operators.
            // Except for parquet, as that format can use statistics to prune file/row-groups.
            #[cfg(feature = "parquet")]
            let is_parquet = matches!(scan_type, FileScan::Parquet { .. });
            #[cfg(not(feature = "parquet"))]
            let is_parquet = false;

            if let (false, true, Some(predicate)) = (is_parquet, push_predicate, predicate) {
                #[cfg(feature = "parquet")]
                debug_assert!(!matches!(scan_type, FileScan::Parquet { .. }));
                let predicate = to_physical(predicate, expr_arena, output_schema.as_ref())?;
                let op = operators::FilterOperator { predicate };
                let op = Box::new(op) as Box<dyn Operator>;
                operator_objects.push(op)
            }
            match scan_type {
                #[cfg(feature = "csv")]
                FileScan::Csv {
                    options: csv_options,
                } => {
                    assert_eq!(paths.len(), 1);
                    let src = sources::CsvSource::new(
                        paths[0].clone(),
                        file_info.schema,
                        csv_options,
                        file_options,
                        verbose,
                    )?;
                    Ok(Box::new(src) as Box<dyn Source>)
                },
                #[cfg(feature = "parquet")]
                FileScan::Parquet {
                    options: parquet_options,
                    cloud_options,
                    metadata,
                } => {
                    let predicate = predicate
                        .map(|predicate| {
                            let p = to_physical(predicate, expr_arena, output_schema.as_ref())?;
                            // Arc's all the way down. :(
                            // Temporarily until: https://github.com/rust-lang/rust/issues/65991
                            // stabilizes
                            struct Wrap {
                                p: Arc<dyn PhysicalPipedExpr>,
                            }
                            impl PhysicalIoExpr for Wrap {
                                fn evaluate_io(&self, df: &DataFrame) -> PolarsResult<Series> {
                                    self.p.evaluate_io(df)
                                }
                                fn as_stats_evaluator(&self) -> Option<&dyn StatsEvaluator> {
                                    self.p.as_stats_evaluator()
                                }
                            }

                            PolarsResult::Ok(Arc::new(Wrap { p }) as Arc<dyn PhysicalIoExpr>)
                        })
                        .transpose()?;
                    let src = sources::ParquetSource::new(
                        paths,
                        parquet_options,
                        cloud_options,
                        metadata,
                        file_options,
                        file_info,
                        verbose,
                        predicate,
                    )?;
                    Ok(Box::new(src) as Box<dyn Source>)
                },
                _ => todo!(),
            }
        },
        _ => unreachable!(),
    }
}

pub fn get_sink<F>(
    node: Node,
    lp_arena: &Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    to_physical: &F,
) -> PolarsResult<Box<dyn SinkTrait>>
where
    F: Fn(Node, &Arena<AExpr>, Option<&SchemaRef>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    use ALogicalPlan::*;
    let out = match lp_arena.get(node) {
        Sink { input, payload } => {
            let input_schema = lp_arena.get(*input).schema(lp_arena);
            match payload {
                SinkType::Memory => {
                    Box::new(OrderedSink::new(input_schema.into_owned())) as Box<dyn SinkTrait>
                },
                #[allow(unused_variables)]
                SinkType::File {
                    path, file_type, ..
                } => {
                    let path = path.as_ref().as_path();
                    match &file_type {
                        #[cfg(feature = "parquet")]
                        FileType::Parquet(options) => {
                            Box::new(ParquetSink::new(path, *options, input_schema.as_ref())?)
                                as Box<dyn SinkTrait>
                        },
                        #[cfg(feature = "ipc")]
                        FileType::Ipc(options) => {
                            Box::new(IpcSink::new(path, *options, input_schema.as_ref())?)
                                as Box<dyn SinkTrait>
                        },
                        #[cfg(feature = "csv")]
                        FileType::Csv(options) => {
                            Box::new(CsvSink::new(path, options.clone(), input_schema.as_ref())?)
                                as Box<dyn SinkTrait>
                        },
                        #[cfg(feature = "json")]
                        FileType::Json(options) => {
                            Box::new(JsonSink::new(path, *options, input_schema.as_ref())?)
                                as Box<dyn SinkTrait>
                        },
                        #[allow(unreachable_patterns)]
                        _ => unreachable!(),
                    }
                },
                #[cfg(feature = "cloud")]
                SinkType::Cloud {
                    uri,
                    file_type,
                    cloud_options,
                } => {
                    let uri = uri.as_ref().as_str();
                    let input_schema = lp_arena.get(*input).schema(lp_arena);
                    let cloud_options = &cloud_options;
                    match &file_type {
                        #[cfg(feature = "parquet")]
                        FileType::Parquet(parquet_options) => Box::new(ParquetCloudSink::new(
                            uri,
                            cloud_options.as_ref(),
                            *parquet_options,
                            input_schema.as_ref(),
                        )?)
                            as Box<dyn SinkTrait>,
                        #[cfg(feature = "ipc")]
                        FileType::Ipc(_ipc_options) => {
                            // TODO: support Ipc as well
                            todo!("For now, only parquet cloud files are supported");
                        },
                        #[allow(unreachable_patterns)]
                        _ => unreachable!(),
                    }
                },
            }
        },
        Join {
            input_left,
            input_right,
            options,
            left_on,
            right_on,
            ..
        } => {
            // slice pushdown optimization should not set this one in a streaming query.
            assert!(options.args.slice.is_none());

            match &options.args.how {
                #[cfg(feature = "cross_join")]
                JoinType::Cross => {
                    Box::new(CrossJoin::new(options.args.suffix().into())) as Box<dyn SinkTrait>
                },
                join_type @ JoinType::Inner | join_type @ JoinType::Left => {
                    let input_schema_left = lp_arena.get(*input_left).schema(lp_arena);
                    let join_columns_left = Arc::new(exprs_to_physical(
                        left_on,
                        expr_arena,
                        to_physical,
                        Some(input_schema_left.as_ref()),
                    )?);
                    let input_schema_right = lp_arena.get(*input_right).schema(lp_arena);
                    let join_columns_right = Arc::new(exprs_to_physical(
                        right_on,
                        expr_arena,
                        to_physical,
                        Some(input_schema_right.as_ref()),
                    )?);

                    let swapped = swap_join_order(options);

                    let (join_columns_left, join_columns_right) = if swapped {
                        (join_columns_right, join_columns_left)
                    } else {
                        (join_columns_left, join_columns_right)
                    };

                    Box::new(GenericBuild::new(
                        Arc::from(options.args.suffix()),
                        join_type.clone(),
                        swapped,
                        join_columns_left,
                        join_columns_right,
                    )) as Box<dyn SinkTrait>
                },
                _ => unimplemented!(),
            }
        },
        Slice { offset, len, .. } => {
            let slice = SliceSink::new(*offset as u64, *len as usize);
            Box::new(slice) as Box<dyn SinkTrait>
        },
        Sort {
            input,
            by_column,
            args,
        } => {
            let input_schema = lp_arena.get(*input).schema(lp_arena).into_owned();

            if by_column.len() == 1 {
                let by_column = aexpr_to_leaf_names_iter(by_column[0], expr_arena)
                    .next()
                    .unwrap();
                let index = input_schema.try_index_of(by_column.as_ref())?;

                let sort_sink = SortSink::new(index, args.clone(), input_schema);
                Box::new(sort_sink) as Box<dyn SinkTrait>
            } else {
                let sort_idx = by_column
                    .iter()
                    .map(|node| {
                        let name = aexpr_to_leaf_names_iter(*node, expr_arena).next().unwrap();
                        input_schema.try_index_of(name.as_ref())
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;

                let sort_sink = SortSinkMultiple::new(args.clone(), input_schema, sort_idx)?;
                Box::new(sort_sink) as Box<dyn SinkTrait>
            }
        },
        Distinct { input, options } => {
            // We create a Groupby.agg_first()/agg_last (depending on the keep strategy
            let input_schema = lp_arena.get(*input).schema(lp_arena).into_owned();

            let (keys, aggs, output_schema) = match &options.subset {
                None => {
                    let keys = input_schema
                        .iter_names()
                        .map(|name| expr_arena.add(AExpr::Column(Arc::from(name.as_str()))))
                        .collect::<Vec<_>>();
                    let aggs = vec![];
                    (keys, aggs, input_schema.clone())
                },
                Some(keys) => {
                    let mut group_by_out_schema = Schema::with_capacity(input_schema.len());
                    let key_names = PlHashSet::from_iter(keys.iter().map(|s| s.as_ref()));
                    let keys = keys
                        .iter()
                        .map(|key| {
                            let (_, name, dtype) = input_schema.get_full(key.as_str()).unwrap();
                            group_by_out_schema.with_column(name.clone(), dtype.clone());
                            expr_arena.add(AExpr::Column(Arc::from(key.as_str())))
                        })
                        .collect();

                    let aggs = input_schema
                        .iter_names()
                        .flat_map(|name| {
                            if key_names.contains(name.as_str()) {
                                None
                            } else {
                                let (_, name, dtype) =
                                    input_schema.get_full(name.as_str()).unwrap();
                                group_by_out_schema.with_column(name.clone(), dtype.clone());
                                let col = expr_arena.add(AExpr::Column(Arc::from(name.as_str())));
                                Some(match options.keep_strategy {
                                    UniqueKeepStrategy::First | UniqueKeepStrategy::Any => {
                                        expr_arena.add(AExpr::Agg(AAggExpr::First(col)))
                                    },
                                    UniqueKeepStrategy::Last => {
                                        expr_arena.add(AExpr::Agg(AAggExpr::Last(col)))
                                    },
                                    UniqueKeepStrategy::None => {
                                        unreachable!()
                                    },
                                })
                            }
                        })
                        .collect();
                    (keys, aggs, group_by_out_schema.into())
                },
            };

            let key_columns = Arc::new(exprs_to_physical(
                &keys,
                expr_arena,
                to_physical,
                Some(&input_schema),
            )?);

            let mut aggregation_columns = Vec::with_capacity(aggs.len());
            let mut agg_fns = Vec::with_capacity(aggs.len());
            let mut input_agg_dtypes = Vec::with_capacity(aggs.len());

            for node in &aggs {
                let (input_dtype, index, agg_fn) =
                    convert_to_hash_agg(*node, expr_arena, &input_schema, &to_physical);
                aggregation_columns.push(index);
                agg_fns.push(agg_fn);
                input_agg_dtypes.push(input_dtype);
            }
            let aggregation_columns = Arc::new(aggregation_columns);

            let group_by_sink = Box::new(GenericGroupby2::new(
                key_columns,
                aggregation_columns,
                Arc::from(agg_fns),
                output_schema,
                input_agg_dtypes,
                options.slice,
            ));

            Box::new(ReProjectSink::new(input_schema, group_by_sink))
        },
        Aggregate {
            input,
            keys,
            aggs,
            schema: output_schema,
            options,
            ..
        } => {
            let input_schema = lp_arena.get(*input).schema(lp_arena).as_ref().clone();
            let key_columns = Arc::new(exprs_to_physical(
                keys,
                expr_arena,
                to_physical,
                Some(&input_schema),
            )?);

            let mut aggregation_columns = Vec::with_capacity(aggs.len());
            let mut agg_fns = Vec::with_capacity(aggs.len());
            let mut input_agg_dtypes = Vec::with_capacity(aggs.len());

            for node in aggs {
                let (input_dtype, index, agg_fn) =
                    convert_to_hash_agg(*node, expr_arena, &input_schema, &to_physical);
                aggregation_columns.push(index);
                agg_fns.push(agg_fn);
                input_agg_dtypes.push(input_dtype);
            }
            let aggregation_columns = Arc::new(aggregation_columns);

            if std::env::var("POLARS_STREAMING_GB2").as_deref() == Ok("1") {
                Box::new(GenericGroupby2::new(
                    key_columns,
                    aggregation_columns,
                    Arc::from(agg_fns),
                    output_schema.clone(),
                    input_agg_dtypes,
                    options.slice,
                ))
            } else {
                match (
                    output_schema.get_at_index(0).unwrap().1.to_physical(),
                    keys.len(),
                ) {
                    (dt, 1) if dt.is_integer() => {
                        with_match_physical_integer_polars_type!(dt, |$T| {
                            Box::new(group_by::PrimitiveGroupbySink::<$T>::new(
                                key_columns[0].clone(),
                                aggregation_columns,
                                agg_fns,
                                input_schema,
                                output_schema.clone(),
                                options.slice,
                            )) as Box<dyn SinkTrait>
                        })
                    },
                    (DataType::Utf8, 1) => Box::new(group_by::Utf8GroupbySink::new(
                        key_columns[0].clone(),
                        aggregation_columns,
                        agg_fns,
                        input_schema,
                        output_schema.clone(),
                        options.slice,
                    )) as Box<dyn SinkTrait>,
                    _ => Box::new(GenericGroupby2::new(
                        key_columns,
                        aggregation_columns,
                        Arc::from(agg_fns),
                        output_schema.clone(),
                        input_agg_dtypes,
                        options.slice,
                    )),
                }
            }
        },
        lp => {
            panic!("{lp:?} not implemented")
        },
    };
    Ok(out)
}

pub fn get_dummy_operator() -> Box<dyn Operator> {
    Box::new(operators::PlaceHolder {})
}

fn get_hstack<F>(
    exprs: &[Node],
    expr_arena: &Arena<AExpr>,
    to_physical: &F,
    input_schema: SchemaRef,
    cse_exprs: Option<Box<HstackOperator>>,
    unchecked: bool,
) -> PolarsResult<HstackOperator>
where
    F: Fn(Node, &Arena<AExpr>, Option<&SchemaRef>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    Ok(operators::HstackOperator {
        exprs: exprs_to_physical(exprs, expr_arena, &to_physical, Some(&input_schema))?,
        input_schema,
        cse_exprs,
        unchecked,
    })
}

pub fn get_operator<F>(
    node: Node,
    lp_arena: &Arena<ALogicalPlan>,
    expr_arena: &Arena<AExpr>,
    to_physical: &F,
) -> PolarsResult<Box<dyn Operator>>
where
    F: Fn(Node, &Arena<AExpr>, Option<&SchemaRef>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    use ALogicalPlan::*;
    let op = match lp_arena.get(node) {
        Projection { expr, input, .. } => {
            let input_schema = lp_arena.get(*input).schema(lp_arena);

            let cse_exprs = expr.cse_exprs();
            let cse_exprs = if cse_exprs.is_empty() {
                None
            } else {
                Some(get_hstack(
                    cse_exprs,
                    expr_arena,
                    to_physical,
                    (*input_schema).clone(),
                    None,
                    true,
                )?)
            };

            let op = operators::ProjectionOperator {
                exprs: exprs_to_physical(
                    expr.default_exprs(),
                    expr_arena,
                    &to_physical,
                    Some(&input_schema),
                )?,
                cse_exprs,
            };
            Box::new(op) as Box<dyn Operator>
        },
        HStack { exprs, input, .. } => {
            let input_schema = lp_arena.get(*input).schema(lp_arena);

            let cse_exprs = exprs.cse_exprs();
            let cse_exprs = if cse_exprs.is_empty() {
                None
            } else {
                Some(Box::new(get_hstack(
                    cse_exprs,
                    expr_arena,
                    to_physical,
                    (*input_schema).clone(),
                    None,
                    true,
                )?))
            };
            let op = get_hstack(
                exprs.default_exprs(),
                expr_arena,
                to_physical,
                (*input_schema).clone(),
                cse_exprs,
                false,
            )?;

            Box::new(op) as Box<dyn Operator>
        },
        Selection { predicate, input } => {
            let input_schema = lp_arena.get(*input).schema(lp_arena);
            let predicate = to_physical(*predicate, expr_arena, Some(input_schema.as_ref()))?;
            let op = operators::FilterOperator { predicate };
            Box::new(op) as Box<dyn Operator>
        },
        MapFunction {
            function: FunctionNode::FastProjection { columns, .. },
            input,
        } => {
            let input_schema = lp_arena.get(*input).schema(lp_arena);
            let op =
                operators::FastProjectionOperator::new(columns.clone(), input_schema.into_owned());
            Box::new(op) as Box<dyn Operator>
        },
        MapFunction { function, .. } => {
            let op = operators::FunctionOperator::new(function.clone());
            Box::new(op) as Box<dyn Operator>
        },
        Union { .. } => {
            let op = operators::Pass::new("union");
            Box::new(op) as Box<dyn Operator>
        },

        lp => {
            panic!("operator {lp:?} not (yet) supported")
        },
    };
    Ok(op)
}

#[allow(clippy::too_many_arguments)]
pub fn create_pipeline<F>(
    sources: &[Node],
    operators: Vec<Box<dyn Operator>>,
    operator_nodes: Vec<Node>,
    sink_nodes: Vec<(usize, Node, Rc<RefCell<u32>>)>,
    lp_arena: &Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
    to_physical: F,
    verbose: bool,
    sink_cache: &mut PlHashMap<usize, Box<dyn SinkTrait>>,
) -> PolarsResult<PipeLine>
where
    F: Fn(Node, &Arena<AExpr>, Option<&SchemaRef>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    use ALogicalPlan::*;

    let mut source_objects = Vec::with_capacity(sources.len());
    let mut operator_objects = Vec::with_capacity(operators.len() + 1);

    for node in sources {
        let src = match lp_arena.get(*node) {
            lp @ DataFrameScan { .. } => get_source(
                lp.clone(),
                &mut operator_objects,
                expr_arena,
                &to_physical,
                true,
                verbose,
            )?,
            lp @ Scan { .. } => get_source(
                lp.clone(),
                &mut operator_objects,
                expr_arena,
                &to_physical,
                true,
                verbose,
            )?,
            Union { inputs, .. } => {
                let sources = inputs
                    .iter()
                    .enumerate()
                    .map(|(i, node)| {
                        let lp = lp_arena.get(*node);
                        // only push predicate of first source
                        get_source(
                            lp.clone(),
                            &mut operator_objects,
                            expr_arena,
                            &to_physical,
                            i == 0,
                            verbose && i == 0,
                        )
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;
                Box::new(sources::UnionSource::new(sources)) as Box<dyn Source>
            },
            lp => {
                panic!("source {lp:?} not (yet) supported")
            },
        };
        source_objects.push(src)
    }

    // this offset is because the source might have inserted operators
    let operator_offset = operator_objects.len();
    operator_objects.extend(operators);

    let sink_nodes = sink_nodes
        .into_iter()
        .map(|(offset, node, shared_count)| {
            // ensure that shared sinks are really shared
            // to achieve this we store/fetch them in a cache
            let sink = if *shared_count.borrow() == 1 {
                get_sink(node, lp_arena, expr_arena, &to_physical)?
            } else {
                match sink_cache.entry(node.0) {
                    Entry::Vacant(entry) => {
                        let sink = get_sink(node, lp_arena, expr_arena, &to_physical)?;
                        entry.insert(sink.split(0));
                        sink
                    },
                    Entry::Occupied(entry) => entry.get().split(0),
                }
            };

            Ok((offset + operator_offset, node, sink, shared_count))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    Ok(PipeLine::new(
        source_objects,
        operator_objects,
        operator_nodes,
        sink_nodes,
        operator_offset,
        verbose,
    ))
}

pub fn swap_join_order(options: &JoinOptions) -> bool {
    matches!(options.args.how, JoinType::Left)
        || match (options.rows_left, options.rows_right) {
            ((Some(left), _), (Some(right), _)) => left > right,
            ((_, left), (_, right)) => left > right,
        }
}
