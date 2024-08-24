use polars_core::prelude::*;
use polars_core::POOL;
use polars_plan::global::_set_n_rows_for_scan;
use polars_plan::plans::expr_ir::ExprIR;

use super::super::executors::{self, Executor};
use super::*;
use crate::utils::*;

fn partitionable_gb(
    keys: &[ExprIR],
    aggs: &[ExprIR],
    _input_schema: &Schema,
    expr_arena: &Arena<AExpr>,
    apply: &Option<Arc<dyn DataFrameUdf>>,
) -> bool {
    // We first check if we can partition the group_by on the latest moment.
    let mut partitionable = true;

    // checks:
    //      1. complex expressions in the group_by itself are also not partitionable
    //          in this case anything more than col("foo")
    //      2. a custom function cannot be partitioned
    //      3. we don't bother with more than 2 keys, as the cardinality likely explodes
    //         by the combinations
    if !keys.is_empty() && keys.len() < 3 && apply.is_none() {
        // complex expressions in the group_by itself are also not partitionable
        // in this case anything more than col("foo")
        for key in keys {
            if (expr_arena).iter(key.node()).count() > 1 {
                partitionable = false;
                break;
            }
        }

        if partitionable {
            for agg in aggs {
                let agg = agg.node();
                let aexpr = expr_arena.get(agg);
                let depth = (expr_arena).iter(agg).count();

                // These single expressions are partitionable
                if matches!(aexpr, AExpr::Len) {
                    continue;
                }
                // col()
                // lit() etc.
                if depth == 1 {
                    partitionable = false;
                    break;
                }

                let has_aggregation =
                    |node: Node| has_aexpr(node, expr_arena, |ae| matches!(ae, AExpr::Agg(_)));

                // check if the aggregation type is partitionable
                // only simple aggregation like col().sum
                // that can be divided in to the aggregation of their partitions are allowed
                if !((expr_arena).iter(agg).all(|(_, ae)| {
                    use AExpr::*;
                    match ae {
                        // struct is needed to keep both states
                        #[cfg(feature = "dtype-struct")]
                        Agg(IRAggExpr::Mean(_)) => {
                            // only numeric means for now.
                            // logical types seem to break because of casts to float.
                            matches!(expr_arena.get(agg).get_type(_input_schema, Context::Default, expr_arena).map(|dt| {
                                        dt.is_numeric()}), Ok(true))
                        },
                        // only allowed expressions
                        Agg(agg_e) => {
                            matches!(
                                            agg_e,
                                            IRAggExpr::Min{..}
                                                | IRAggExpr::Max{..}
                                                | IRAggExpr::Sum(_)
                                                | IRAggExpr::Last(_)
                                                | IRAggExpr::First(_)
                                                | IRAggExpr::Count(_, true)
                                        )
                        },
                        Function {input, options, ..} => {
                            matches!(options.collect_groups, ApplyOptions::ElementWise) && input.len() == 1 &&
                                !has_aggregation(input[0].node())
                        }
                        BinaryExpr {left, right, ..} => {
                            !has_aggregation(*left) && !has_aggregation(*right)
                        }
                        Ternary {truthy, falsy, predicate,..} => {
                            !has_aggregation(*truthy) && !has_aggregation(*falsy) && !has_aggregation(*predicate)
                        }
                        Column(_) | Len | Literal(_) | Cast {..} => {
                            true
                        }
                        _ => {
                            false
                        },
                    }
                }) &&
                    // we only allow expressions that end with an aggregation
                    matches!(aexpr, AExpr::Agg(_)))
                {
                    partitionable = false;
                    break;
                }

                #[cfg(feature = "object")]
                {
                    for name in aexpr_to_leaf_names(agg, expr_arena) {
                        let dtype = _input_schema.get(&name).unwrap();

                        if let DataType::Object(_, _) = dtype {
                            partitionable = false;
                            break;
                        }
                    }
                    if !partitionable {
                        break;
                    }
                }
            }
        }
    } else {
        partitionable = false;
    }
    partitionable
}

struct ConversionState {
    expr_depth: u16,
}

impl ConversionState {
    fn new() -> PolarsResult<Self> {
        Ok(ConversionState {
            expr_depth: get_expr_depth_limit()?,
        })
    }
}

pub fn create_physical_plan(
    root: Node,
    lp_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<Box<dyn Executor>> {
    let state = ConversionState::new()?;
    create_physical_plan_impl(root, lp_arena, expr_arena, &state)
}

fn create_physical_plan_impl(
    root: Node,
    lp_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
    state: &ConversionState,
) -> PolarsResult<Box<dyn Executor>> {
    use IR::*;

    let logical_plan = lp_arena.take(root);
    match logical_plan {
        #[cfg(feature = "python")]
        PythonScan { mut options } => {
            let mut predicate_serialized = None;

            let predicate = if let PythonPredicate::Polars(e) = &options.predicate {
                let phys_expr = || {
                    let mut state = ExpressionConversionState::new(true, state.expr_depth);
                    create_physical_expr(
                        e,
                        Context::Default,
                        expr_arena,
                        Some(&options.schema),
                        &mut state,
                    )
                };

                // Convert to a pyarrow eval string.
                if matches!(options.python_source, PythonScanSource::Pyarrow) {
                    if let Some(eval_str) = polars_plan::plans::python::pyarrow::predicate_to_pa(
                        e.node(),
                        expr_arena,
                        Default::default(),
                    ) {
                        options.predicate = PythonPredicate::PyArrow(eval_str);
                        // We don't have to use a physical expression as pyarrow deals with the filter.
                        None
                    } else {
                        Some(phys_expr()?)
                    }
                }
                // Convert to physical expression for the case the reader cannot consume the predicate.
                else {
                    let dsl_expr = e.to_expr(expr_arena);
                    predicate_serialized =
                        polars_plan::plans::python::predicate::serialize(&dsl_expr)?;

                    Some(phys_expr()?)
                }
            } else {
                None
            };
            Ok(Box::new(executors::PythonScanExec {
                options,
                predicate,
                predicate_serialized,
            }))
        },
        Sink { payload, .. } => match payload {
            SinkType::Memory => {
                polars_bail!(InvalidOperation: "memory sink not supported in the standard engine")
            },
            SinkType::File { file_type, .. } => {
                polars_bail!(InvalidOperation:
                    "sink_{file_type:?} not yet supported in standard engine. Use 'collect().write_parquet()'"
                )
            },
            #[cfg(feature = "cloud")]
            SinkType::Cloud { .. } => {
                polars_bail!(InvalidOperation: "cloud sink not supported in standard engine.")
            },
        },
        Union { inputs, options } => {
            let inputs = inputs
                .into_iter()
                .map(|node| create_physical_plan_impl(node, lp_arena, expr_arena, state))
                .collect::<PolarsResult<Vec<_>>>()?;
            Ok(Box::new(executors::UnionExec { inputs, options }))
        },
        HConcat {
            inputs, options, ..
        } => {
            let inputs = inputs
                .into_iter()
                .map(|node| create_physical_plan_impl(node, lp_arena, expr_arena, state))
                .collect::<PolarsResult<Vec<_>>>()?;
            Ok(Box::new(executors::HConcatExec { inputs, options }))
        },
        Slice { input, offset, len } => {
            let input = create_physical_plan_impl(input, lp_arena, expr_arena, state)?;
            Ok(Box::new(executors::SliceExec { input, offset, len }))
        },
        Filter { input, predicate } => {
            let mut streamable = is_streamable(predicate.node(), expr_arena, Context::Default);
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
            if streamable {
                // This can cause problems with string caches
                streamable = !input_schema
                    .iter_dtypes()
                    .any(|dt| dt.contains_categoricals())
                    || {
                        #[cfg(feature = "dtype-categorical")]
                        {
                            polars_core::using_string_cache()
                        }

                        #[cfg(not(feature = "dtype-categorical"))]
                        {
                            false
                        }
                    }
            }
            let input = create_physical_plan_impl(input, lp_arena, expr_arena, state)?;
            let mut state = ExpressionConversionState::new(true, state.expr_depth);
            let predicate = create_physical_expr(
                &predicate,
                Context::Default,
                expr_arena,
                Some(&input_schema),
                &mut state,
            )?;
            Ok(Box::new(executors::FilterExec::new(
                predicate,
                input,
                state.has_windows,
                streamable,
            )))
        },
        #[allow(unused_variables)]
        Scan {
            paths,
            file_info,
            hive_parts,
            output_schema,
            scan_type,
            predicate,
            mut file_options,
        } => {
            file_options.slice = if let Some((offset, len)) = file_options.slice {
                Some((offset, _set_n_rows_for_scan(Some(len)).unwrap()))
            } else {
                _set_n_rows_for_scan(None).map(|x| (0, x))
            };

            let mut state = ExpressionConversionState::new(true, state.expr_depth);
            let predicate = predicate
                .map(|pred| {
                    create_physical_expr(
                        &pred,
                        Context::Default,
                        expr_arena,
                        output_schema.as_ref(),
                        &mut state,
                    )
                })
                .map_or(Ok(None), |v| v.map(Some))?;

            match scan_type {
                #[cfg(feature = "csv")]
                FileScan::Csv { options, .. } => Ok(Box::new(executors::CsvExec {
                    paths,
                    file_info,
                    options,
                    predicate,
                    file_options,
                })),
                #[cfg(feature = "ipc")]
                FileScan::Ipc {
                    options,
                    cloud_options,
                    metadata,
                } => Ok(Box::new(executors::IpcExec {
                    paths,
                    file_info,
                    predicate,
                    options,
                    file_options,
                    hive_parts,
                    cloud_options,
                })),
                #[cfg(feature = "parquet")]
                FileScan::Parquet {
                    options,
                    cloud_options,
                    metadata,
                } => Ok(Box::new(executors::ParquetExec::new(
                    paths,
                    file_info,
                    hive_parts,
                    predicate,
                    options,
                    cloud_options,
                    file_options,
                    metadata,
                ))),
                #[cfg(feature = "json")]
                FileScan::NDJson { options, .. } => Ok(Box::new(executors::JsonExec::new(
                    paths,
                    options,
                    file_options,
                    file_info,
                    predicate,
                ))),
                FileScan::Anonymous { function, .. } => {
                    Ok(Box::new(executors::AnonymousScanExec {
                        function,
                        predicate,
                        file_options,
                        file_info,
                        output_schema,
                        predicate_has_windows: state.has_windows,
                    }))
                },
            }
        },
        Select {
            expr,
            input,
            schema: _schema,
            options,
            ..
        } => {
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
            let input = create_physical_plan_impl(input, lp_arena, expr_arena, state)?;
            let mut state = ExpressionConversionState::new(
                POOL.current_num_threads() > expr.len(),
                state.expr_depth,
            );

            let streamable =
                options.should_broadcast && all_streamable(&expr, expr_arena, Context::Default);
            let phys_expr = create_physical_expressions_from_irs(
                &expr,
                Context::Default,
                expr_arena,
                Some(&input_schema),
                &mut state,
            )?;
            Ok(Box::new(executors::ProjectionExec {
                input,
                expr: phys_expr,
                has_windows: state.has_windows,
                input_schema,
                #[cfg(test)]
                schema: _schema,
                options,
                streamable,
            }))
        },
        Reduce {
            exprs,
            input,
            schema,
        } => {
            let select = Select {
                input,
                expr: exprs,
                schema,
                options: Default::default(),
            };
            let node = lp_arena.add(select);
            create_physical_plan(node, lp_arena, expr_arena)
        },
        DataFrameScan {
            df,
            filter: predicate,
            schema,
            output_schema,
            ..
        } => {
            let mut state = ExpressionConversionState::new(true, state.expr_depth);
            let selection = predicate
                .map(|pred| {
                    create_physical_expr(
                        &pred,
                        Context::Default,
                        expr_arena,
                        Some(&schema),
                        &mut state,
                    )
                })
                .transpose()?;
            Ok(Box::new(executors::DataFrameExec {
                df,
                projection: output_schema.map(|s| s.iter_names().cloned().collect()),
                filter: selection,
                predicate_has_windows: state.has_windows,
            }))
        },
        Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => {
            let input_schema = lp_arena.get(input).schema(lp_arena);
            let by_column = create_physical_expressions_from_irs(
                &by_column,
                Context::Default,
                expr_arena,
                Some(input_schema.as_ref()),
                &mut ExpressionConversionState::new(true, state.expr_depth),
            )?;
            let input = create_physical_plan_impl(input, lp_arena, expr_arena, state)?;
            Ok(Box::new(executors::SortExec {
                input,
                by_column,
                slice,
                sort_options,
            }))
        },
        Cache {
            input,
            id,
            cache_hits,
        } => {
            let input = create_physical_plan_impl(input, lp_arena, expr_arena, state)?;
            Ok(Box::new(executors::CacheExec {
                id,
                input,
                count: cache_hits,
            }))
        },
        Distinct { input, options } => {
            let input = create_physical_plan_impl(input, lp_arena, expr_arena, state)?;
            Ok(Box::new(executors::UniqueExec { input, options }))
        },
        GroupBy {
            input,
            keys,
            aggs,
            apply,
            schema,
            maintain_order,
            options,
        } => {
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
            let options = Arc::try_unwrap(options).unwrap_or_else(|options| (*options).clone());
            let phys_keys = create_physical_expressions_from_irs(
                &keys,
                Context::Default,
                expr_arena,
                Some(&input_schema),
                &mut ExpressionConversionState::new(true, state.expr_depth),
            )?;
            let phys_aggs = create_physical_expressions_from_irs(
                &aggs,
                Context::Aggregation,
                expr_arena,
                Some(&input_schema),
                &mut ExpressionConversionState::new(true, state.expr_depth),
            )?;

            let _slice = options.slice;
            #[cfg(feature = "dynamic_group_by")]
            if let Some(options) = options.dynamic {
                let input = create_physical_plan_impl(input, lp_arena, expr_arena, state)?;
                return Ok(Box::new(executors::GroupByDynamicExec {
                    input,
                    keys: phys_keys,
                    aggs: phys_aggs,
                    options,
                    input_schema,
                    slice: _slice,
                    apply,
                }));
            }

            #[cfg(feature = "dynamic_group_by")]
            if let Some(options) = options.rolling {
                let input = create_physical_plan_impl(input, lp_arena, expr_arena, state)?;
                return Ok(Box::new(executors::GroupByRollingExec {
                    input,
                    keys: phys_keys,
                    aggs: phys_aggs,
                    options,
                    input_schema,
                    slice: _slice,
                    apply,
                }));
            }

            // We first check if we can partition the group_by on the latest moment.
            let partitionable = partitionable_gb(&keys, &aggs, &input_schema, expr_arena, &apply);
            if partitionable {
                let from_partitioned_ds = (&*lp_arena).iter(input).any(|(_, lp)| {
                    if let Union { options, .. } = lp {
                        options.from_partitioned_ds
                    } else {
                        false
                    }
                });
                let input = create_physical_plan_impl(input, lp_arena, expr_arena, state)?;
                let keys = keys
                    .iter()
                    .map(|e| e.to_expr(expr_arena))
                    .collect::<Vec<_>>();
                let aggs = aggs
                    .iter()
                    .map(|e| e.to_expr(expr_arena))
                    .collect::<Vec<_>>();
                Ok(Box::new(executors::PartitionGroupByExec::new(
                    input,
                    phys_keys,
                    phys_aggs,
                    maintain_order,
                    options.slice,
                    input_schema,
                    schema,
                    from_partitioned_ds,
                    keys,
                    aggs,
                )))
            } else {
                let input = create_physical_plan_impl(input, lp_arena, expr_arena, state)?;
                Ok(Box::new(executors::GroupByExec::new(
                    input,
                    phys_keys,
                    phys_aggs,
                    apply,
                    maintain_order,
                    input_schema,
                    options.slice,
                )))
            }
        },
        Join {
            input_left,
            input_right,
            left_on,
            right_on,
            options,
            ..
        } => {
            let parallel = if options.force_parallel {
                true
            } else if options.allow_parallel {
                // check if two DataFrames come from a separate source.
                // If they don't we can parallelize,
                // we may deadlock if we don't check this
                let mut sources_left = PlHashSet::new();
                agg_source_paths(input_left, &mut sources_left, lp_arena);
                let mut sources_right = PlHashSet::new();
                agg_source_paths(input_right, &mut sources_right, lp_arena);
                sources_left.intersection(&sources_right).next().is_none()
            } else {
                false
            };

            let input_left = create_physical_plan_impl(input_left, lp_arena, expr_arena, state)?;
            let input_right = create_physical_plan_impl(input_right, lp_arena, expr_arena, state)?;
            let left_on = create_physical_expressions_from_irs(
                &left_on,
                Context::Default,
                expr_arena,
                None,
                &mut ExpressionConversionState::new(true, state.expr_depth),
            )?;
            let right_on = create_physical_expressions_from_irs(
                &right_on,
                Context::Default,
                expr_arena,
                None,
                &mut ExpressionConversionState::new(true, state.expr_depth),
            )?;
            let options = Arc::try_unwrap(options).unwrap_or_else(|options| (*options).clone());
            Ok(Box::new(executors::JoinExec::new(
                input_left,
                input_right,
                left_on,
                right_on,
                parallel,
                options.args,
            )))
        },
        HStack {
            input,
            exprs,
            schema: _schema,
            options,
        } => {
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
            let input = create_physical_plan_impl(input, lp_arena, expr_arena, state)?;

            let streamable =
                options.should_broadcast && all_streamable(&exprs, expr_arena, Context::Default);

            let mut state = ExpressionConversionState::new(
                POOL.current_num_threads() > exprs.len(),
                state.expr_depth,
            );

            let phys_exprs = create_physical_expressions_from_irs(
                &exprs,
                Context::Default,
                expr_arena,
                Some(&input_schema),
                &mut state,
            )?;
            Ok(Box::new(executors::StackExec {
                input,
                has_windows: state.has_windows,
                exprs: phys_exprs,
                input_schema,
                options,
                streamable,
            }))
        },
        MapFunction {
            input, function, ..
        } => {
            let input = create_physical_plan_impl(input, lp_arena, expr_arena, state)?;
            Ok(Box::new(executors::UdfExec { input, function }))
        },
        ExtContext {
            input, contexts, ..
        } => {
            let input = create_physical_plan_impl(input, lp_arena, expr_arena, state)?;
            let contexts = contexts
                .into_iter()
                .map(|node| create_physical_plan_impl(node, lp_arena, expr_arena, state))
                .collect::<PolarsResult<_>>()?;
            Ok(Box::new(executors::ExternalContext { input, contexts }))
        },
        SimpleProjection { input, columns } => {
            let input = create_physical_plan_impl(input, lp_arena, expr_arena, state)?;
            let exec = executors::ProjectionSimple { input, columns };
            Ok(Box::new(exec))
        },
        Invalid => unreachable!(),
    }
}
