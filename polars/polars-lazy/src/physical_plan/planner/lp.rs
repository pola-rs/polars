use polars_core::prelude::*;
use polars_io::aggregations::ScanAggregation;

use super::super::executors;
use crate::prelude::*;

#[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
fn aggregate_expr_to_scan_agg(
    aggregate: Vec<Node>,
    expr_arena: &mut Arena<AExpr>,
) -> Vec<ScanAggregation> {
    aggregate
        .into_iter()
        .map(|mut expr| {
            let mut alias = None;
            if let AExpr::Alias(e, name) = expr_arena.get(expr) {
                expr = *e;
                alias = Some((*name).to_string())
            };
            if let AExpr::Agg(agg) = expr_arena.get(expr) {
                match agg {
                    AAggExpr::Min {
                        input: e,
                        propagate_nans: false,
                    } => ScanAggregation::Min {
                        column: (*aexpr_to_leaf_names(*e, expr_arena).pop().unwrap()).to_string(),
                        alias,
                    },
                    AAggExpr::Max {
                        input: e,
                        propagate_nans: false,
                    } => ScanAggregation::Max {
                        column: (*aexpr_to_leaf_names(*e, expr_arena).pop().unwrap()).to_string(),
                        alias,
                    },
                    AAggExpr::Sum(e) => ScanAggregation::Sum {
                        column: (*aexpr_to_leaf_names(*e, expr_arena).pop().unwrap()).to_string(),
                        alias,
                    },
                    AAggExpr::First(e) => ScanAggregation::First {
                        column: (*aexpr_to_leaf_names(*e, expr_arena).pop().unwrap()).to_string(),
                        alias,
                    },
                    AAggExpr::Last(e) => ScanAggregation::Last {
                        column: (*aexpr_to_leaf_names(*e, expr_arena).pop().unwrap()).to_string(),
                        alias,
                    },
                    _ => todo!(),
                }
            } else {
                unreachable!()
            }
        })
        .collect()
}

#[derive(Default)]
pub struct PhysicalPlanner {}

impl PhysicalPlanner {
    pub fn create_physical_expressions(
        &self,
        exprs: &[Node],
        context: Context,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<Vec<Arc<dyn PhysicalExpr>>> {
        exprs
            .iter()
            .map(|e| self.create_physical_expr(*e, context, expr_arena))
            .collect()
    }
    pub fn create_physical_plan(
        &self,
        root: Node,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<Box<dyn Executor>> {
        use ALogicalPlan::*;
        let logical_plan = lp_arena.take(root);
        match logical_plan {
            #[cfg(feature = "python")]
            PythonScan { options } => Ok(Box::new(executors::PythonScanExec { options })),
            Union { inputs, options } => {
                let inputs = inputs
                    .into_iter()
                    .map(|node| self.create_physical_plan(node, lp_arena, expr_arena))
                    .collect::<PolarsResult<Vec<_>>>()?;
                Ok(Box::new(executors::UnionExec { inputs, options }))
            }
            Melt { input, args, .. } => {
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                Ok(Box::new(executors::MeltExec { input, args }))
            }
            Slice { input, offset, len } => {
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                Ok(Box::new(executors::SliceExec { input, offset, len }))
            }
            Selection { input, predicate } => {
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                let predicate =
                    self.create_physical_expr(predicate, Context::Default, expr_arena)?;
                Ok(Box::new(executors::FilterExec::new(predicate, input)))
            }
            #[cfg(feature = "csv-file")]
            CsvScan {
                path,
                schema,
                output_schema: _,
                options,
                predicate,
                aggregate,
            } => {
                let predicate = predicate
                    .map(|pred| self.create_physical_expr(pred, Context::Default, expr_arena))
                    .map_or(Ok(None), |v| v.map(Some))?;
                let aggregate = aggregate_expr_to_scan_agg(aggregate, expr_arena);
                Ok(Box::new(executors::CsvExec {
                    path,
                    schema,
                    options,
                    predicate,
                    aggregate,
                }))
            }
            #[cfg(feature = "ipc")]
            IpcScan {
                path,
                schema,
                output_schema: _,
                predicate,
                aggregate,
                options,
            } => {
                let predicate = predicate
                    .map(|pred| self.create_physical_expr(pred, Context::Default, expr_arena))
                    .map_or(Ok(None), |v| v.map(Some))?;

                let aggregate = aggregate_expr_to_scan_agg(aggregate, expr_arena);
                Ok(Box::new(executors::IpcExec {
                    path,
                    schema,
                    predicate,
                    aggregate,
                    options,
                }))
            }
            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                schema,
                output_schema: _,
                predicate,
                aggregate,
                options,
            } => {
                let predicate = predicate
                    .map(|pred| self.create_physical_expr(pred, Context::Default, expr_arena))
                    .map_or(Ok(None), |v| v.map(Some))?;

                let aggregate = aggregate_expr_to_scan_agg(aggregate, expr_arena);
                Ok(Box::new(executors::ParquetExec::new(
                    path, schema, predicate, aggregate, options,
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
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                let phys_expr =
                    self.create_physical_expressions(&expr, Context::Default, expr_arena)?;
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
                #[cfg(test)]
                    schema: _schema,
                ..
            } => {
                let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();

                let has_windows = expr.iter().any(|node| has_aexpr_window(*node, expr_arena));
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                let phys_expr =
                    self.create_physical_expressions(&expr, Context::Default, expr_arena)?;
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
                ..
            } => {
                let selection = selection
                    .map(|pred| self.create_physical_expr(pred, Context::Default, expr_arena))
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
                ..
            } => {
                let predicate = predicate
                    .map(|pred| self.create_physical_expr(pred, Context::Default, expr_arena))
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
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                let by_column =
                    self.create_physical_expressions(&by_column, Context::Default, expr_arena)?;
                Ok(Box::new(executors::SortExec {
                    input,
                    by_column,
                    args,
                }))
            }
            Explode { input, columns, .. } => {
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                Ok(Box::new(executors::ExplodeExec { input, columns }))
            }
            Cache { input, id, count } => {
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                Ok(Box::new(executors::CacheExec { id, input, count }))
            }
            Distinct { input, options } => {
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                Ok(Box::new(executors::DropDuplicatesExec { input, options }))
            }
            Aggregate {
                input,
                keys,
                aggs,
                apply,
                schema: _,
                maintain_order,
                options,
            } => {
                let input_schema = lp_arena.get(input).schema(lp_arena).into_owned();
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;

                let phys_keys =
                    self.create_physical_expressions(&keys, Context::Default, expr_arena)?;

                let phys_aggs =
                    self.create_physical_expressions(&aggs, Context::Aggregation, expr_arena)?;

                let slice = options.slice;
                if let Some(options) = options.dynamic {
                    return Ok(Box::new(executors::GroupByDynamicExec {
                        input,
                        keys: phys_keys,
                        aggs: phys_aggs,
                        options,
                        input_schema,
                        slice,
                    }));
                }

                if let Some(options) = options.rolling {
                    return Ok(Box::new(executors::GroupByRollingExec {
                        input,
                        keys: phys_keys,
                        aggs: phys_aggs,
                        options,
                        input_schema,
                        slice,
                    }));
                }

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
                        if (&*expr_arena).iter(key).count() > 1 {
                            partitionable = false;
                            break;
                        }
                    }

                    if partitionable {
                        for agg in &aggs {
                            let aexpr = expr_arena.get(*agg);
                            let depth = (&*expr_arena).iter(*agg).count();

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

                            let has_aggregation = |node: Node| {
                                has_aexpr(node, expr_arena, |ae| matches!(ae, AExpr::Agg(_)))
                            };

                            // check if the aggregation type is partitionable
                            // only simple aggregation like col().sum
                            // that can be divided in to the aggregation of their partitions are allowed
                            if !((&*expr_arena).iter(*agg).all(|(_, ae)| {
                                use AExpr::*;
                                match ae {
                                    // struct is needed to keep both states
                                    #[cfg(feature = "dtype-struct")]
                                    Agg(AAggExpr::Mean(_)) => {
                                        // only numeric means for now.
                                        // logical types seem to break because of casts to float.
                                        matches!(expr_arena.get(*agg).get_type(&input_schema, Context::Default, expr_arena).map(|dt| {
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
                                    let dtype = input_schema.get(&name).unwrap();

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
                if partitionable {
                    Ok(Box::new(executors::PartitionGroupByExec::new(
                        input,
                        phys_keys,
                        phys_aggs,
                        maintain_order,
                        options.slice,
                        input_schema,
                    )))
                } else {
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

                let input_left = self.create_physical_plan(input_left, lp_arena, expr_arena)?;
                let input_right = self.create_physical_plan(input_right, lp_arena, expr_arena)?;
                let left_on =
                    self.create_physical_expressions(&left_on, Context::Default, expr_arena)?;
                let right_on =
                    self.create_physical_expressions(&right_on, Context::Default, expr_arena)?;
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
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                let phys_expr =
                    self.create_physical_expressions(&exprs, Context::Default, expr_arena)?;
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
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                Ok(Box::new(executors::UdfExec { input, function }))
            }
            ExtContext {
                input, contexts, ..
            } => {
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                let contexts = contexts
                    .into_iter()
                    .map(|node| self.create_physical_plan(node, lp_arena, expr_arena))
                    .collect::<PolarsResult<_>>()?;
                Ok(Box::new(executors::ExternalContext { input, contexts }))
            }
        }
    }
}
