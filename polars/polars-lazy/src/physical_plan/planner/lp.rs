use super::super::executors;
use crate::prelude::*;
use polars_core::prelude::*;
use polars_io::aggregations::ScanAggregation;

#[cfg(any(feature = "parquet", feature = "csv-file"))]
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
                    AAggExpr::Min(e) => ScanAggregation::Min {
                        column: (*aexpr_to_root_names(*e, expr_arena).pop().unwrap()).to_string(),
                        alias,
                    },
                    AAggExpr::Max(e) => ScanAggregation::Max {
                        column: (*aexpr_to_root_names(*e, expr_arena).pop().unwrap()).to_string(),
                        alias,
                    },
                    AAggExpr::Sum(e) => ScanAggregation::Sum {
                        column: (*aexpr_to_root_names(*e, expr_arena).pop().unwrap()).to_string(),
                        alias,
                    },
                    AAggExpr::First(e) => ScanAggregation::First {
                        column: (*aexpr_to_root_names(*e, expr_arena).pop().unwrap()).to_string(),
                        alias,
                    },
                    AAggExpr::Last(e) => ScanAggregation::Last {
                        column: (*aexpr_to_root_names(*e, expr_arena).pop().unwrap()).to_string(),
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
pub struct DefaultPlanner {}

impl DefaultPlanner {
    pub fn create_physical_expressions(
        &self,
        exprs: &[Node],
        context: Context,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<Vec<Arc<dyn PhysicalExpr>>> {
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
    ) -> Result<Box<dyn Executor>> {
        use ALogicalPlan::*;
        let logical_plan = lp_arena.take(root);
        match logical_plan {
            Union { inputs, options } => {
                let inputs = inputs
                    .into_iter()
                    .map(|node| self.create_physical_plan(node, lp_arena, expr_arena))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Box::new(executors::UnionExec { inputs, options }))
            }
            Melt {
                input,
                id_vars,
                value_vars,
                ..
            } => {
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                Ok(Box::new(executors::MeltExec {
                    input,
                    id_vars,
                    value_vars,
                }))
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
                let input_schema = lp_arena.get(input).schema(lp_arena).clone();
                let has_windows = expr.iter().any(|node| has_window_aexpr(*node, expr_arena));
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
                let input_schema = lp_arena.get(input).schema(lp_arena).clone();

                let has_windows = expr.iter().any(|node| has_window_aexpr(*node, expr_arena));
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
                let has_windows = if let Some(projection) = &projection {
                    projection
                        .iter()
                        .any(|node| has_window_aexpr(*node, expr_arena))
                } else {
                    false
                };

                let selection = selection
                    .map(|pred| self.create_physical_expr(pred, Context::Default, expr_arena))
                    .map_or(Ok(None), |v| v.map(Some))?;
                let projection = projection
                    .map(|proj| {
                        self.create_physical_expressions(&proj, Context::Default, expr_arena)
                    })
                    .map_or(Ok(None), |v| v.map(Some))?;
                Ok(Box::new(executors::DataFrameExec {
                    df,
                    projection,
                    selection,
                    has_windows,
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
            Cache { input } => {
                let schema = lp_arena.get(input).schema(lp_arena);
                // todo! fix the unique constraint in the schema. Probably in projection pushdown at joins
                let mut unique = PlHashSet::with_capacity(schema.len());
                // assumption of 80 characters per column name
                let mut key = String::with_capacity(schema.len() * 80);
                for name in schema.iter_names() {
                    if unique.insert(name) {
                        key.push_str(name)
                    }
                }
                // mutable borrow otherwise
                drop(unique);
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                Ok(Box::new(executors::CacheExec { key, input }))
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
                let input_schema = lp_arena.get(input).schema(lp_arena).clone();
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;

                let mut phys_keys =
                    self.create_physical_expressions(&keys, Context::Default, expr_arena)?;

                let phys_aggs =
                    self.create_physical_expressions(&aggs, Context::Aggregation, expr_arena)?;

                if let Some(options) = options.dynamic {
                    return Ok(Box::new(executors::GroupByDynamicExec {
                        input,
                        keys: phys_keys,
                        aggs: phys_aggs,
                        options,
                        input_schema,
                    }));
                }

                if let Some(options) = options.rolling {
                    return Ok(Box::new(executors::GroupByRollingExec {
                        input,
                        keys: phys_keys,
                        aggs: phys_aggs,
                        options,
                        input_schema,
                    }));
                }

                // We first check if we can partition the groupby on the latest moment.
                // TODO: fix this brittle/ buggy state and implement partitioned groupby's in eager
                let mut partitionable = true;

                // checks:
                //      1. complex expressions in the groupby itself are also not partitionable
                //          in this case anything more than col("foo")
                //      2. a custom function cannot be partitioned
                if keys.len() == 1 && apply.is_none() {
                    // complex expressions in the groupby itself are also not partitionable
                    // in this case anything more than col("foo")
                    if (&*expr_arena).iter(keys[0]).count() > 1 {
                        partitionable = false;
                    }

                    for agg in &aggs {
                        // make sure that we don't have a binary expr in the expr tree
                        let matches = |e: &AExpr| {
                            matches!(
                                e,
                                AExpr::SortBy { .. }
                                    | AExpr::Filter { .. }
                                    | AExpr::BinaryExpr { .. }
                                    | AExpr::Function { .. }
                            )
                        };
                        if aexpr_to_root_nodes(*agg, expr_arena).len() != 1
                            || has_aexpr(*agg, expr_arena, matches)
                        {
                            partitionable = false;
                            break;
                        }

                        let agg = node_to_expr(*agg, expr_arena);

                        #[cfg(feature = "object")]
                        {
                            let name = expr_to_root_column_name(&agg).unwrap();
                            let dtype = input_schema.get(&name).unwrap();

                            if let DataType::Object(_) = dtype {
                                partitionable = false;
                                break;
                            }
                        }

                        // check if the aggregation type is partitionable
                        match agg {
                            Expr::Agg(AggExpr::Min(_))
                            | Expr::Agg(AggExpr::Max(_))
                            | Expr::Agg(AggExpr::Sum(_))
                            | Expr::Agg(AggExpr::Mean(_))
                            // first need to implement this correctly
                            // | Expr::Agg(AggExpr::Count(_))
                            | Expr::Agg(AggExpr::Last(_))
                            | Expr::Agg(AggExpr::List(_))
                            | Expr::Agg(AggExpr::First(_)) => {}
                            _ => {
                                partitionable = false;
                                break
                            }
                        }
                    }
                } else {
                    partitionable = false;
                }
                if partitionable {
                    Ok(Box::new(executors::PartitionGroupByExec::new(
                        input,
                        phys_keys.pop().unwrap(),
                        phys_aggs,
                        aggs.into_iter()
                            .map(|n| node_to_expr(n, expr_arena))
                            .collect(),
                        maintain_order,
                    )))
                } else {
                    Ok(Box::new(executors::GroupByExec::new(
                        input,
                        phys_keys,
                        phys_aggs,
                        apply,
                        maintain_order,
                        input_schema,
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
                    // Otherwise it is in cache.
                    let mut sources_left = PlHashSet::with_capacity(16);
                    agg_source_paths(input_left, &mut sources_left, lp_arena);
                    let mut sources_right = PlHashSet::with_capacity(16);
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
                )))
            }
            HStack { input, exprs, .. } => {
                let input_schema = lp_arena.get(input).schema(lp_arena).clone();
                let has_windows = exprs.iter().any(|node| has_window_aexpr(*node, expr_arena));
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
            Udf {
                input, function, ..
            } => {
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                Ok(Box::new(executors::UdfExec { input, function }))
            }
        }
    }
}
