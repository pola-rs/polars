use polars_core::prelude::*;

use super::super::executors::{self, Executor};
use super::*;
#[cfg(feature = "streaming")]
use crate::physical_plan::streaming::insert_streaming_nodes;
use crate::utils::*;

fn partitionable_gb(
    keys: &[Node],
    aggs: &[Node],
    _input_schema: &Schema,
    expr_arena: &Arena<AExpr>,
    apply: &Option<Arc<dyn DataFrameUdf>>,
) -> bool {
    // We first check if we can partition the groupby on the latest moment.
    let mut partitionable = true;

    // checks:
    //      1. complex expressions in the groupby itself are also not partitionable
    //          in this case anything more than col("foo")
    //      2. a custom function cannot be partitioned
    //      3. we don't bother with more than 2 keys, as the cardinality likely explodes
    //         by the combinations
    if !keys.is_empty() && keys.len() < 3 && apply.is_none() {
        // complex expressions in the groupby itself are also not partitionable
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
                            AExpr::Count => {}
                            _ => {
                                partitionable = false;
                                break;
                            }
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
                                                | AAggExpr::Count(_)
                                        )
                        },
                        Function {input, options, ..} => {
                            matches!(options.collect_groups, ApplyOptions::ApplyFlat) && input.len() == 1 &&
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

                        if let DataType::Object(_) = dtype {
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
        FileSink { .. } => panic!(
            "sink_parquet not yet supported in standard engine. Use 'collect().write_parquet()'"
        ),
        Union { inputs, options } => {
            let inputs = inputs
                .into_iter()
                .map(|node| create_physical_plan(node, lp_arena, expr_arena))
                .collect::<PolarsResult<Vec<_>>>()?;
            Ok(Box::new(executors::UnionExec { inputs, options }))
        }
        Melt { input, args, .. } => {
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            Ok(Box::new(executors::MeltExec { input, args }))
        }
        Slice { input, offset, len } => {
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            Ok(Box::new(executors::SliceExec { input, offset, len }))
        }
        Selection { input, predicate } => {
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            let predicate = create_physical_expr(predicate, Context::Default, expr_arena, None)?;
            Ok(Box::new(executors::FilterExec::new(predicate, input)))
        }
        #[cfg(feature = "csv-file")]
        CsvScan {
            path,
            file_info,
            output_schema,
            options,
            predicate,
        } => {
            let predicate = predicate
                .map(|pred| {
                    create_physical_expr(pred, Context::Default, expr_arena, output_schema.as_ref())
                })
                .map_or(Ok(None), |v| v.map(Some))?;
            Ok(Box::new(executors::CsvExec {
                path,
                schema: file_info.schema,
                options,
                predicate,
            }))
        }
        #[cfg(feature = "ipc")]
        IpcScan {
            path,
            file_info,
            output_schema,
            predicate,
            options,
        } => {
            let predicate = predicate
                .map(|pred| {
                    create_physical_expr(pred, Context::Default, expr_arena, output_schema.as_ref())
                })
                .map_or(Ok(None), |v| v.map(Some))?;

            Ok(Box::new(executors::IpcExec {
                path,
                schema: file_info.schema,
                predicate,
                options,
            }))
        }
        #[cfg(feature = "parquet")]
        ParquetScan {
            path,
            file_info,
            output_schema,
            predicate,
            options,
            cloud_options,
        } => {
            let predicate = predicate
                .map(|pred| {
                    create_physical_expr(pred, Context::Default, expr_arena, output_schema.as_ref())
                })
                .map_or(Ok(None), |v| v.map(Some))?;

            Ok(Box::new(executors::ParquetExec::new(
                path,
                file_info.schema,
                predicate,
                options,
                cloud_options,
            )))
        }
        Projection {
            expr,
            input,
            schema: _schema,
            ..
        } => {
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
            let has_windows = expr.iter().any(|node| has_aexpr_window(*node, expr_arena));
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            let phys_expr = create_physical_expressions(
                &expr,
                Context::Default,
                expr_arena,
                Some(&input_schema),
            )?;
            Ok(Box::new(executors::ProjectionExec {
                input,
                expr: phys_expr,
                has_windows,
                input_schema,
                #[cfg(test)]
                schema: _schema,
            }))
        }
        LocalProjection {
            expr,
            input,
            schema: _schema,
            ..
        } => {
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();

            let has_windows = expr.iter().any(|node| has_aexpr_window(*node, expr_arena));
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            let phys_expr = create_physical_expressions(
                &expr,
                Context::Default,
                expr_arena,
                Some(&input_schema),
            )?;
            Ok(Box::new(executors::ProjectionExec {
                input,
                expr: phys_expr,
                has_windows,
                input_schema,
                #[cfg(test)]
                schema: _schema,
            }))
        }
        DataFrameScan {
            df,
            projection,
            selection,
            schema,
            ..
        } => {
            let selection = selection
                .map(|pred| create_physical_expr(pred, Context::Default, expr_arena, Some(&schema)))
                .map_or(Ok(None), |v| v.map(Some))?;
            Ok(Box::new(executors::DataFrameExec {
                df,
                projection,
                selection,
            }))
        }
        AnonymousScan {
            function,
            predicate,
            options,
            output_schema,
            ..
        } => {
            let predicate = predicate
                .map(|pred| {
                    create_physical_expr(pred, Context::Default, expr_arena, output_schema.as_ref())
                })
                .map_or(Ok(None), |v| v.map(Some))?;
            Ok(Box::new(executors::AnonymousScanExec {
                function,
                predicate,
                options,
            }))
        }
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
            )?;
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            Ok(Box::new(executors::SortExec {
                input,
                by_column,
                args,
            }))
        }
        Explode { input, columns, .. } => {
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            Ok(Box::new(executors::ExplodeExec { input, columns }))
        }
        Cache { input, id, count } => {
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            Ok(Box::new(executors::CacheExec { id, input, count }))
        }
        Distinct { input, options } => {
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            Ok(Box::new(executors::DropDuplicatesExec { input, options }))
        }
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
            let phys_keys = create_physical_expressions(
                &keys,
                Context::Default,
                expr_arena,
                Some(&input_schema),
            )?;
            let phys_aggs = create_physical_expressions(
                &aggs,
                Context::Aggregation,
                expr_arena,
                Some(&input_schema),
            )?;

            let _slice = options.slice;
            #[cfg(feature = "dynamic_groupby")]
            if let Some(options) = options.dynamic {
                let input = create_physical_plan(input, lp_arena, expr_arena)?;
                return Ok(Box::new(executors::GroupByDynamicExec {
                    input,
                    keys: phys_keys,
                    aggs: phys_aggs,
                    options,
                    input_schema,
                    slice: _slice,
                }));
            }

            #[cfg(feature = "dynamic_groupby")]
            if let Some(options) = options.rolling {
                let input = create_physical_plan(input, lp_arena, expr_arena)?;
                return Ok(Box::new(executors::GroupByRollingExec {
                    input,
                    keys: phys_keys,
                    aggs: phys_aggs,
                    options,
                    input_schema,
                    slice: _slice,
                }));
            }

            // We first check if we can partition the groupby on the latest moment.
            let partitionable = partitionable_gb(&keys, &aggs, &input_schema, expr_arena, &apply);
            if partitionable {
                #[cfg(feature = "streaming")]
                if !maintain_order
                    // many aggregations are more expensive
                    // at a certain point the cost of collecting
                    // the indices is amortized
                    && aggs.len() < 10
                    && std::env::var("POLARS_NO_STREAMING_GROUPBY").is_err()
                {
                    let key_dtype = schema.get_index(0).unwrap().1.to_physical();
                    // only on numeric and string keys for now
                    let allowed_key = keys.len() == 1 && key_dtype.is_numeric()
                        || matches!(key_dtype, DataType::Utf8);
                    let allowed_aggs = schema.iter_dtypes().skip(1).all(|dtype| {
                        let dt = dtype.to_physical();
                        dt.is_numeric() || matches!(dt, DataType::Utf8 | DataType::Boolean)
                    });

                    let lp = Aggregate {
                        input,
                        keys,
                        aggs,
                        apply,
                        schema: schema.clone(),
                        maintain_order,
                        options: options.clone(),
                    };
                    let root = lp_arena.add(lp);

                    // do not jit insert join streaming nodes
                    // first we have to test them more and ensure solid perf
                    let has_joins = (&*lp_arena)
                        .iter(root)
                        .any(|(_, lp)| matches!(lp, Join { .. }));
                    if allowed_key
                        && allowed_aggs
                        && !has_joins
                        && insert_streaming_nodes(root, lp_arena, expr_arena, &mut vec![], false)?
                    {
                        return create_physical_plan(root, lp_arena, expr_arena);
                    }
                }

                let from_partitioned_ds = (&*lp_arena).iter(input).any(|(_, lp)| {
                    if let ALogicalPlan::Union { options, .. } = lp {
                        options.from_partitioned_ds
                    } else {
                        false
                    }
                });
                let input = create_physical_plan(input, lp_arena, expr_arena)?;
                Ok(Box::new(executors::PartitionGroupByExec::new(
                    input,
                    phys_keys,
                    phys_aggs,
                    maintain_order,
                    options.slice,
                    input_schema,
                    schema,
                    from_partitioned_ds,
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
        }
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
            let left_on =
                create_physical_expressions(&left_on, Context::Default, expr_arena, None)?;
            let right_on =
                create_physical_expressions(&right_on, Context::Default, expr_arena, None)?;
            Ok(Box::new(executors::JoinExec::new(
                input_left,
                input_right,
                options.how,
                left_on,
                right_on,
                parallel,
                options.suffix,
                options.slice,
            )))
        }
        HStack { input, exprs, .. } => {
            let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
            let has_windows = exprs.iter().any(|node| has_aexpr_window(*node, expr_arena));
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            let phys_expr = create_physical_expressions(
                &exprs,
                Context::Default,
                expr_arena,
                Some(&input_schema),
            )?;
            Ok(Box::new(executors::StackExec {
                input,
                has_windows,
                expr: phys_expr,
                input_schema,
            }))
        }
        MapFunction {
            input, function, ..
        } => {
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            Ok(Box::new(executors::UdfExec { input, function }))
        }
        ExtContext {
            input, contexts, ..
        } => {
            let input = create_physical_plan(input, lp_arena, expr_arena)?;
            let contexts = contexts
                .into_iter()
                .map(|node| create_physical_plan(node, lp_arena, expr_arena))
                .collect::<PolarsResult<_>>()?;
            Ok(Box::new(executors::ExternalContext { input, contexts }))
        }
    }
}
