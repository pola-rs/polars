use super::expressions as phys_expr;
use crate::logical_plan::Context;
use crate::physical_plan::executors::groupby_dynamic::GroupByDynamicExec;
use crate::physical_plan::executors::groupby_rolling::GroupByRollingExec;
#[cfg(feature = "ipc")]
use crate::physical_plan::executors::scan::IpcExec;
use crate::physical_plan::executors::union::UnionExec;
use crate::prelude::count::CountExpr;
use crate::prelude::shift::ShiftExpr;
use crate::prelude::*;
#[cfg(feature = "object")]
use crate::utils::expr_to_root_column_name;
use crate::utils::has_window_aexpr;
use crate::{
    logical_plan::iterator::ArenaExprIter,
    utils::{aexpr_to_root_names, aexpr_to_root_nodes, agg_source_paths, has_aexpr},
};
use polars_core::prelude::*;
use polars_core::{frame::groupby::GroupByMethod, utils::parallel_op_series};
#[cfg(any(feature = "parquet", feature = "csv-file", feature = "ipc"))]
use polars_io::aggregations::ScanAggregation;
use std::sync::Arc;

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

impl PhysicalPlanner for DefaultPlanner {
    fn create_physical_plan(
        &self,
        _root: Node,
        _lp_arena: &mut Arena<ALogicalPlan>,
        _expr_arena: &mut Arena<AExpr>,
    ) -> Result<Box<dyn Executor>> {
        self.create_physical_plan(_root, _lp_arena, _expr_arena)
    }
}

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
                Ok(Box::new(UnionExec { inputs, options }))
            }
            Melt {
                input,
                id_vars,
                value_vars,
                ..
            } => {
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                Ok(Box::new(MeltExec {
                    input,
                    id_vars,
                    value_vars,
                }))
            }
            Slice { input, offset, len } => {
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                Ok(Box::new(SliceExec { input, offset, len }))
            }
            Selection { input, predicate } => {
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                let predicate =
                    self.create_physical_expr(predicate, Context::Default, expr_arena)?;
                Ok(Box::new(FilterExec::new(predicate, input)))
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
                Ok(Box::new(CsvExec {
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
                Ok(Box::new(IpcExec {
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
                Ok(Box::new(ParquetExec::new(
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
                Ok(Box::new(ProjectionExec {
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
                Ok(Box::new(ProjectionExec {
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
                Ok(Box::new(DataFrameExec {
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
                Ok(Box::new(SortExec {
                    input,
                    by_column,
                    args,
                }))
            }
            Explode { input, columns } => {
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                Ok(Box::new(ExplodeExec { input, columns }))
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
                Ok(Box::new(CacheExec { key, input }))
            }
            Distinct { input, options } => {
                let input = self.create_physical_plan(input, lp_arena, expr_arena)?;
                Ok(Box::new(DropDuplicatesExec { input, options }))
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
                    return Ok(Box::new(GroupByDynamicExec {
                        input,
                        keys: phys_keys,
                        aggs: phys_aggs,
                        options,
                        input_schema,
                    }));
                }

                if let Some(options) = options.rolling {
                    return Ok(Box::new(GroupByRollingExec {
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
                    Ok(Box::new(PartitionGroupByExec::new(
                        input,
                        phys_keys.pop().unwrap(),
                        phys_aggs,
                        aggs.into_iter()
                            .map(|n| node_to_expr(n, expr_arena))
                            .collect(),
                        maintain_order,
                    )))
                } else {
                    Ok(Box::new(GroupByExec::new(
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
                Ok(Box::new(JoinExec::new(
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
                Ok(Box::new(StackExec {
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
                Ok(Box::new(UdfExec { input, function }))
            }
        }
    }

    pub fn create_physical_expr(
        &self,
        expression: Node,
        ctxt: Context,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        use AExpr::*;

        match expr_arena.get(expression).clone() {
            Count => Ok(Arc::new(CountExpr::new())),
            Window {
                mut function,
                partition_by,
                order_by: _,
                options,
            } => {
                // TODO! Order by
                let group_by =
                    self.create_physical_expressions(&partition_by, Context::Default, expr_arena)?;
                let phys_function =
                    self.create_physical_expr(function, Context::Aggregation, expr_arena)?;
                let mut out_name = None;
                let mut apply_columns = aexpr_to_root_names(function, expr_arena);
                // sort and then dedup removes consecutive duplicates == all duplicates
                apply_columns.sort();
                apply_columns.dedup();

                if apply_columns.is_empty() {
                    if has_aexpr(function, expr_arena, |e| matches!(e, AExpr::Literal(_))) {
                        apply_columns.push(Arc::from("literal"))
                    } else {
                        let e = node_to_expr(function, expr_arena);
                        return Err(PolarsError::ComputeError(
                            format!(
                                "Cannot apply a window function, did not find a root column. \
                            This is likely due to a syntax error in this expression: {:?}",
                                e
                            )
                            .into(),
                        ));
                    }
                }

                if let Alias(expr, name) = expr_arena.get(function) {
                    function = *expr;
                    out_name = Some(name.clone());
                };
                let function = node_to_expr(function, expr_arena);

                Ok(Arc::new(WindowExpr {
                    group_by,
                    apply_columns,
                    out_name,
                    function,
                    phys_function,
                    options,
                    expr: node_to_expr(expression, expr_arena),
                }))
            }
            Literal(value) => Ok(Arc::new(LiteralExpr::new(
                value,
                node_to_expr(expression, expr_arena),
            ))),
            BinaryExpr { left, op, right } => {
                let lhs = self.create_physical_expr(left, ctxt, expr_arena)?;
                let rhs = self.create_physical_expr(right, ctxt, expr_arena)?;
                Ok(Arc::new(phys_expr::binary::BinaryExpr::new(
                    lhs,
                    op,
                    rhs,
                    node_to_expr(expression, expr_arena),
                )))
            }
            Column(column) => Ok(Arc::new(ColumnExpr::new(
                column,
                node_to_expr(expression, expr_arena),
            ))),
            Sort { expr, options } => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                Ok(Arc::new(SortExpr::new(
                    phys_expr,
                    options,
                    node_to_expr(expression, expr_arena),
                )))
            }
            Take { expr, idx } => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                let phys_idx = self.create_physical_expr(idx, ctxt, expr_arena)?;
                Ok(Arc::new(TakeExpr {
                    phys_expr,
                    idx: phys_idx,
                    expr: node_to_expr(expression, expr_arena),
                }))
            }
            SortBy { expr, by, reverse } => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                let phys_by = self.create_physical_expressions(&by, ctxt, expr_arena)?;
                Ok(Arc::new(SortByExpr::new(
                    phys_expr,
                    phys_by,
                    reverse,
                    node_to_expr(expression, expr_arena),
                )))
            }
            Filter { input, by } => {
                let phys_input = self.create_physical_expr(input, ctxt, expr_arena)?;
                let phys_by = self.create_physical_expr(by, ctxt, expr_arena)?;
                Ok(Arc::new(FilterExpr::new(
                    phys_input,
                    phys_by,
                    node_to_expr(expression, expr_arena),
                )))
            }
            Not(expr) => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                Ok(Arc::new(NotExpr::new(
                    phys_expr,
                    node_to_expr(expression, expr_arena),
                )))
            }
            Alias(expr, name) => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                Ok(Arc::new(AliasExpr::new(
                    phys_expr,
                    name,
                    node_to_expr(expression, expr_arena),
                )))
            }
            IsNull(expr) => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                Ok(Arc::new(IsNullExpr::new(
                    phys_expr,
                    node_to_expr(expression, expr_arena),
                )))
            }
            IsNotNull(expr) => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                Ok(Arc::new(IsNotNullExpr::new(
                    phys_expr,
                    node_to_expr(expression, expr_arena),
                )))
            }
            Agg(agg) => {
                match agg {
                    AAggExpr::Min(expr) => {
                        // todo! Output type is dependent on schema.
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Min)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    parallel_op_series(|s| Ok(s.min_as_series()), s, None)
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Max(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Max)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    parallel_op_series(|s| Ok(s.max_as_series()), s, None)
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Sum(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Sum)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    parallel_op_series(|s| Ok(s.sum_as_series()), s, None)
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Std(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Std)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    Ok(s.std_as_series())
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Var(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Var)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    Ok(s.var_as_series())
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Mean(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Mean)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    Ok(s.mean_as_series())
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Median(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Median)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    Ok(s.median_as_series())
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::First(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::First)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    Ok(s.head(Some(1)))
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Last(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Last)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    Ok(s.tail(Some(1)))
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::List(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::List)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = &s[0];
                                    s.to_list().map(|ca| ca.into_series())
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::NUnique(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => Ok(Arc::new(AggregationExpr::new(
                                input,
                                GroupByMethod::NUnique,
                            ))),
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    s.n_unique().map(|count| {
                                        UInt32Chunked::from_slice(s.name(), &[count as u32])
                                            .into_series()
                                    })
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Quantile {
                        expr,
                        quantile,
                        interpol,
                    } => {
                        // todo! add schema to get correct output type
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggQuantileExpr::new(input, quantile, interpol)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    s.quantile_as_series(quantile, interpol)
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::AggGroups(expr) => {
                        if let Context::Default = ctxt {
                            panic!("agg groups expression only supported in aggregation context")
                        }
                        let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        Ok(Arc::new(AggregationExpr::new(
                            phys_expr,
                            GroupByMethod::Groups,
                        )))
                    }
                    AAggExpr::Count(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Count)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    let count = s.len();
                                    Ok(UInt32Chunked::from_slice(s.name(), &[count as u32])
                                        .into_series())
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                }
            }
            Cast {
                expr,
                data_type,
                strict,
            } => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                Ok(Arc::new(CastExpr {
                    input: phys_expr,
                    data_type,
                    expr: node_to_expr(expression, expr_arena),
                    strict,
                }))
            }
            Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                let predicate = self.create_physical_expr(predicate, ctxt, expr_arena)?;
                let truthy = self.create_physical_expr(truthy, ctxt, expr_arena)?;
                let falsy = self.create_physical_expr(falsy, ctxt, expr_arena)?;
                Ok(Arc::new(TernaryExpr::new(
                    predicate,
                    truthy,
                    falsy,
                    node_to_expr(expression, expr_arena),
                )))
            }
            Function {
                input,
                function,
                output_type: _,
                options,
            } => {
                let input = self.create_physical_expressions(&input, ctxt, expr_arena)?;

                Ok(Arc::new(ApplyExpr {
                    inputs: input,
                    function,
                    expr: node_to_expr(expression, expr_arena),
                    collect_groups: options.collect_groups,
                    auto_explode: options.auto_explode,
                }))
            }
            Shift { input, periods } => {
                let input = self.create_physical_expr(input, ctxt, expr_arena)?;
                Ok(Arc::new(ShiftExpr {
                    input,
                    periods,
                    expr: node_to_expr(expression, expr_arena),
                }))
            }
            Slice {
                input,
                offset,
                length,
            } => {
                let input = self.create_physical_expr(input, ctxt, expr_arena)?;
                let offset = self.create_physical_expr(offset, ctxt, expr_arena)?;
                let length = self.create_physical_expr(length, ctxt, expr_arena)?;
                Ok(Arc::new(SliceExpr {
                    input,
                    offset,
                    length,
                    expr: node_to_expr(expression, expr_arena),
                }))
            }
            Reverse(expr) => {
                let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                    let s = std::mem::take(&mut s[0]);
                    Ok(s.reverse())
                }) as Arc<dyn SeriesUdf>);
                Ok(Arc::new(ApplyExpr {
                    inputs: vec![input],
                    function,
                    expr: node_to_expr(expression, expr_arena),
                    collect_groups: ApplyOptions::ApplyGroups,
                    auto_explode: false,
                }))
            }
            Duplicated(expr) => {
                let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                    let s = std::mem::take(&mut s[0]);
                    s.is_duplicated().map(|ca| ca.into_series())
                }) as Arc<dyn SeriesUdf>);
                Ok(Arc::new(ApplyExpr {
                    inputs: vec![input],
                    function,
                    expr: node_to_expr(expression, expr_arena),
                    collect_groups: ApplyOptions::ApplyGroups,
                    auto_explode: false,
                }))
            }
            IsUnique(expr) => {
                let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                    let s = std::mem::take(&mut s[0]);
                    s.is_unique().map(|ca| ca.into_series())
                }) as Arc<dyn SeriesUdf>);
                Ok(Arc::new(ApplyExpr {
                    inputs: vec![input],
                    function,
                    expr: node_to_expr(expression, expr_arena),
                    collect_groups: ApplyOptions::ApplyGroups,
                    auto_explode: false,
                }))
            }
            Explode(expr) => {
                let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                    let s = std::mem::take(&mut s[0]);
                    s.explode()
                }) as Arc<dyn SeriesUdf>);
                Ok(Arc::new(ApplyExpr {
                    inputs: vec![input],
                    function,
                    expr: node_to_expr(expression, expr_arena),
                    collect_groups: ApplyOptions::ApplyFlat,
                    auto_explode: false,
                }))
            }
            Wildcard => panic!("should be no wildcard at this point"),
            Nth(_) => panic!("should be no nth at this point"),
        }
    }
}
