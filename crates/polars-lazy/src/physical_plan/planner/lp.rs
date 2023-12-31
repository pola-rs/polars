use polars_core::prelude::*;
use polars_core::POOL;
use polars_plan::global::_set_n_rows_for_scan;

use super::super::executors::{self, Executor};
use super::*;
use crate::utils::*;

fn partitionable_gb(
    keys: &[Node],
    aggs: &[Node],
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
            if (expr_arena).iter(*key).count() > 1 {
                partitionable = false;
                break;
            }
        }

        if partitionable {
            for agg in aggs {
                let aexpr = expr_arena.get(*agg);
                let depth = (expr_arena).iter(*agg).count();

                // These single expressions are partitionable
                if matches!(aexpr, AExpr::Count) {
                    continue;
                }
                // col()
                // lit() etc.
                if depth == 1 {
                    partitionable = false;
                    break;
                }

                // it should end with an aggregation
                if let AExpr::Alias(input, _) = aexpr {
                    // col().agg().alias() is allowed: count of 3
                    // col().alias() is not allowed: count of 2
                    // count().alias() is allowed: count of 2
                    if depth <= 2 {
                        match expr_arena.get(*input) {
                            AExpr::Count => {},
                            _ => {
                                partitionable = false;
                                break;
                            },
                        }
                    }
                }

                let has_aggregation =
                    |node: Node| has_aexpr(node, expr_arena, |ae| matches!(ae, AExpr::Agg(_)));

                // check if the aggregation type is partitionable
                // only simple aggregation like col().sum
                // that can be divided in to the aggregation of their partitions are allowed
                if !((expr_arena).iter(*agg).all(|(_, ae)| {
                    use AExpr::*;
                    match ae {
                        // struct is needed to keep both states
                        #[cfg(feature = "dtype-struct")]
                        Agg(AAggExpr::Mean(_)) => {
                            // only numeric means for now.
                            // logical types seem to break because of casts to float.
                            matches!(expr_arena.get(*agg).get_type(_input_schema, Context::Default, expr_arena).map(|dt| {
                                        dt.is_numeric()}), Ok(true))
                        },
                        // only allowed expressions
                        Agg(agg_e) => {
                            matches!(
                                            agg_e,
                                            AAggExpr::Min{..}
                                                | AAggExpr::Max{..}
                                                | AAggExpr::Sum(_)
                                                | AAggExpr::Last(_)
                                                | AAggExpr::First(_)
                                                | AAggExpr::Count(_, true)
                                        )
                        },
                        Function {input, options, ..} => {
                            matches!(options.collect_groups, ApplyOptions::ElementWise) && input.len() == 1 &&
                                !has_aggregation(input[0])
                        }
                        BinaryExpr {left, right, ..} => {
                            !has_aggregation(*left) && !has_aggregation(*right)
                        }
                        Ternary {truthy, falsy, predicate,..} => {
                            !has_aggregation(*truthy) && !has_aggregation(*falsy) && !has_aggregation(*predicate)
                        }
                        Column(_) | Alias(_, _) | Count | Literal(_) | Cast {..} => {
                            true
                        }
                        _ => {
                            false
                        },
                    }
                }) &&
                    // we only allow expressions that end with an aggregation
                    matches!(aexpr, AExpr::Alias(_, _) | AExpr::Agg(_)))
                {
                    partitionable = false;
                    break;
                }

                #[cfg(feature = "object")]
                {
                    for name in aexpr_to_leaf_names(*agg, expr_arena) {
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

pub fn create_physical_plan(
    root: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<Box<dyn Executor>> {
    use ALogicalPlan::*;

    let logical_plan = lp_arena.take(root);
    match logical_plan {
        #[cfg(feature = "python")]
        PythonScan { options, .. } => Ok(Box::new(executors::PythonScanExec { options })),
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
                .map(|node| create_physical_plan(node, lp_arena, expr_arena))
                .collect::<PolarsResult<Vec<_>>>()?;
            Ok(Box::new(executors::UnionExec { inputs, options }))
        },
        #[cfg(feature = "horizontal_concat")]
        HConcat {
            inputs, options, ..
        } => {
            let inputs = inputs
                .into_iter()
                .map(|node| create_physical_plan(node, lp_arena, expr_arena))
                .collect::<PolarsResult<Vec<_>>>()?;
            Ok(Box::new(executors::HConcatExec { inputs, options }))
        },
        Slice { input, offset, len } => {
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            Ok(Box::new(executors::SliceExec { input, offset, len }))
        },
        Selection { input, predicate } => {
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            let mut state = ExpressionConversionState::default();
            let predicate = create_physical_expr(
                predicate,
                Context::Default,
                expr_arena,
                Some(&input_schema),
                &mut state,
            )?;
            Ok(Box::new(executors::FilterExec::new(
                predicate,
                input,
                state.has_windows,
            )))
        },
        #[allow(unused_variables)]
        Scan {
            paths,
            file_info,
            output_schema,
            scan_type,
            predicate,
            mut file_options,
        } => {
            file_options.n_rows = _set_n_rows_for_scan(file_options.n_rows);
            let mut state = ExpressionConversionState::default();
            let predicate = predicate
                .map(|pred| {
                    create_physical_expr(
                        pred,
                        Context::Default,
                        expr_arena,
                        output_schema.as_ref(),
                        &mut state,
                    )
                })
                .map_or(Ok(None), |v| v.map(Some))?;

            match scan_type {
                #[cfg(feature = "csv")]
                FileScan::Csv {
                    options: csv_options,
                } => {
                    assert_eq!(paths.len(), 1);
                    let path = paths[0].clone();
                    Ok(Box::new(executors::CsvExec {
                        path,
                        schema: file_info.schema,
                        options: csv_options,
                        predicate,
                        file_options,
                    }))
                },
                #[cfg(feature = "ipc")]
                FileScan::Ipc { options } => {
                    assert_eq!(paths.len(), 1);
                    let path = paths[0].clone();
                    Ok(Box::new(executors::IpcExec {
                        path,
                        schema: file_info.schema,
                        predicate,
                        options,
                        file_options,
                    }))
                },
                #[cfg(feature = "parquet")]
                FileScan::Parquet {
                    options,
                    cloud_options,
                    metadata,
                } => Ok(Box::new(executors::ParquetExec::new(
                    paths,
                    file_info,
                    predicate,
                    options,
                    cloud_options,
                    file_options,
                    metadata,
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
        Projection {
            expr,
            input,
            schema: _schema,
            options,
            ..
        } => {
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            let mut state = ExpressionConversionState::new(POOL.current_num_threads() > expr.len());
            let phys_expr = create_physical_expressions(
                expr.default_exprs(),
                Context::Default,
                expr_arena,
                Some(&input_schema),
                &mut state,
            )?;
            let cse_expr = create_physical_expressions(
                expr.cse_exprs(),
                Context::Default,
                expr_arena,
                Some(&input_schema),
                &mut state,
            )?;
            Ok(Box::new(executors::ProjectionExec {
                input,
                cse_exprs: cse_expr,
                expr: phys_expr,
                has_windows: state.has_windows,
                input_schema,
                #[cfg(test)]
                schema: _schema,
                options,
            }))
        },
        DataFrameScan {
            df,
            projection,
            selection: predicate,
            schema,
            ..
        } => {
            let mut state = ExpressionConversionState::default();
            let selection = predicate
                .map(|pred| {
                    create_physical_expr(
                        pred,
                        Context::Default,
                        expr_arena,
                        Some(&schema),
                        &mut state,
                    )
                })
                .transpose()?;
            Ok(Box::new(executors::DataFrameExec {
                df,
                projection,
                selection,
                predicate_has_windows: state.has_windows,
            }))
        },
        Sort {
            input,
            by_column,
            args,
        } => {
            let input_schema = lp_arena.get(input).schema(lp_arena);
            let by_column = create_physical_expressions(
                &by_column,
                Context::Default,
                expr_arena,
                Some(input_schema.as_ref()),
                &mut Default::default(),
            )?;
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            Ok(Box::new(executors::SortExec {
                input,
                by_column,
                args,
            }))
        },
        Cache { input, id, count } => {
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            Ok(Box::new(executors::CacheExec { id, input, count }))
        },
        Distinct { input, options } => {
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            Ok(Box::new(executors::UniqueExec { input, options }))
        },
        Aggregate {
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
            let phys_keys = create_physical_expressions(
                &keys,
                Context::Default,
                expr_arena,
                Some(&input_schema),
                &mut Default::default(),
            )?;
            let phys_aggs = create_physical_expressions(
                &aggs,
                Context::Aggregation,
                expr_arena,
                Some(&input_schema),
                &mut Default::default(),
            )?;

            let _slice = options.slice;
            #[cfg(feature = "dynamic_group_by")]
            if let Some(options) = options.dynamic {
                let input = create_physical_plan(input, lp_arena, expr_arena)?;
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
                let input = create_physical_plan(input, lp_arena, expr_arena)?;
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
                let input = create_physical_plan(input, lp_arena, expr_arena)?;
                let keys = keys
                    .iter()
                    .map(|node| node_to_expr(*node, expr_arena))
                    .collect::<Vec<_>>();
                let aggs = aggs
                    .iter()
                    .map(|node| node_to_expr(*node, expr_arena))
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
                let input = create_physical_plan(input, lp_arena, expr_arena)?;
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

            let input_left = create_physical_plan(input_left, lp_arena, expr_arena)?;
            let input_right = create_physical_plan(input_right, lp_arena, expr_arena)?;
            let left_on = create_physical_expressions(
                &left_on,
                Context::Default,
                expr_arena,
                None,
                &mut Default::default(),
            )?;
            let right_on = create_physical_expressions(
                &right_on,
                Context::Default,
                expr_arena,
                None,
                &mut Default::default(),
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
            let input = create_physical_plan(input, lp_arena, expr_arena)?;

            let mut state =
                ExpressionConversionState::new(POOL.current_num_threads() > exprs.len());

            let cse_exprs = create_physical_expressions(
                exprs.cse_exprs(),
                Context::Default,
                expr_arena,
                Some(&input_schema),
                &mut state,
            )?;

            let phys_exprs = create_physical_expressions(
                exprs.default_exprs(),
                Context::Default,
                expr_arena,
                Some(&input_schema),
                &mut state,
            )?;
            Ok(Box::new(executors::StackExec {
                input,
                has_windows: state.has_windows,
                cse_exprs,
                exprs: phys_exprs,
                input_schema,
                options,
            }))
        },
        MapFunction {
            input, function, ..
        } => {
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            Ok(Box::new(executors::UdfExec { input, function }))
        },
        ExtContext {
            input, contexts, ..
        } => {
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            let contexts = contexts
                .into_iter()
                .map(|node| create_physical_plan(node, lp_arena, expr_arena))
                .collect::<PolarsResult<_>>()?;
            Ok(Box::new(executors::ExternalContext { input, contexts }))
        },
    }
}
