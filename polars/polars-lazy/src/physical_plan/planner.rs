use crate::logical_plan::Context;
use crate::physical_plan::executors::*;
use crate::prelude::*;
use crate::utils::{agg_source_paths, expr_to_root_column_name};
use ahash::RandomState;
use itertools::Itertools;
use polars_core::frame::group_by::GroupByMethod;
use polars_core::prelude::*;
use polars_io::ScanAggregation;
use std::collections::HashSet;
use std::sync::Arc;

fn aggregate_expr_to_scan_agg(aggregate: Vec<Expr>) -> Vec<ScanAggregation> {
    aggregate
        .into_iter()
        .map(|e| {
            let mut alias = None;
            let mut expr = e;
            if let Expr::Alias(e, name) = expr {
                expr = *e;
                alias = Some((*name).clone())
            };
            if let Expr::Agg(agg) = expr {
                match agg {
                    AggExpr::Min(e) => ScanAggregation::Min {
                        column: (*expr_to_root_column_name(&e).unwrap()).clone(),
                        alias,
                    },
                    AggExpr::Max(e) => ScanAggregation::Max {
                        column: (*expr_to_root_column_name(&e).unwrap()).clone(),
                        alias,
                    },
                    AggExpr::Sum(e) => ScanAggregation::Sum {
                        column: (*expr_to_root_column_name(&e).unwrap()).clone(),
                        alias,
                    },
                    AggExpr::First(e) => ScanAggregation::First {
                        column: (*expr_to_root_column_name(&e).unwrap()).clone(),
                        alias,
                    },
                    AggExpr::Last(e) => ScanAggregation::Last {
                        column: (*expr_to_root_column_name(&e).unwrap()).clone(),
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

pub struct DefaultPlanner {}
impl Default for DefaultPlanner {
    fn default() -> Self {
        Self {}
    }
}

impl PhysicalPlanner for DefaultPlanner {
    fn create_physical_plan(&self, logical_plan: LogicalPlan) -> Result<Box<dyn Executor>> {
        self.create_initial_physical_plan(logical_plan)
    }
}

impl DefaultPlanner {
    fn create_physical_expressions(
        &self,
        exprs: Vec<Expr>,
        context: Context,
    ) -> Result<Vec<Arc<dyn PhysicalExpr>>> {
        exprs
            .into_iter()
            .map(|e| self.create_physical_expr(e, context))
            .collect()
    }
    pub fn create_initial_physical_plan(
        &self,
        logical_plan: LogicalPlan,
    ) -> Result<Box<dyn Executor>> {
        match logical_plan {
            LogicalPlan::Melt {
                input,
                id_vars,
                value_vars,
                ..
            } => {
                let input = self.create_initial_physical_plan(*input)?;
                Ok(Box::new(MeltExec {
                    input,
                    id_vars,
                    value_vars,
                }))
            }
            LogicalPlan::Slice { input, offset, len } => {
                let input = self.create_initial_physical_plan(*input)?;
                Ok(Box::new(SliceExec { input, offset, len }))
            }
            LogicalPlan::Selection { input, predicate } => {
                let input = self.create_initial_physical_plan(*input)?;
                let predicate = self.create_physical_expr(predicate, Context::Other)?;
                Ok(Box::new(FilterExec::new(predicate, input)))
            }
            LogicalPlan::CsvScan {
                path,
                schema,
                has_header,
                delimiter,
                ignore_errors,
                skip_rows,
                stop_after_n_rows,
                with_columns,
                predicate,
                aggregate,
                cache,
            } => {
                let predicate = predicate
                    .map(|pred| self.create_physical_expr(pred, Context::Other))
                    .map_or(Ok(None), |v| v.map(Some))?;
                let aggregate = aggregate_expr_to_scan_agg(aggregate);
                Ok(Box::new(CsvExec::new(
                    path,
                    schema,
                    has_header,
                    delimiter,
                    ignore_errors,
                    skip_rows,
                    stop_after_n_rows,
                    with_columns,
                    predicate,
                    aggregate,
                    cache,
                )))
            }
            #[cfg(feature = "parquet")]
            LogicalPlan::ParquetScan {
                path,
                schema,
                with_columns,
                predicate,
                aggregate,
                stop_after_n_rows,
                cache,
            } => {
                let predicate = predicate
                    .map(|pred| self.create_physical_expr(pred, Context::Other))
                    .map_or(Ok(None), |v| v.map(Some))?;

                let aggregate = aggregate_expr_to_scan_agg(aggregate);
                Ok(Box::new(ParquetExec::new(
                    path,
                    schema,
                    with_columns,
                    predicate,
                    aggregate,
                    stop_after_n_rows,
                    cache,
                )))
            }
            LogicalPlan::Projection { expr, input, .. } => {
                let input = self.create_initial_physical_plan(*input)?;
                let phys_expr = self.create_physical_expressions(expr, Context::Other)?;
                Ok(Box::new(StandardExec::new("projection", input, phys_expr)))
            }
            LogicalPlan::LocalProjection { expr, input, .. } => {
                let input = self.create_initial_physical_plan(*input)?;
                let phys_expr = self.create_physical_expressions(expr, Context::Other)?;
                Ok(Box::new(StandardExec::new("projection", input, phys_expr)))
            }
            LogicalPlan::DataFrameScan {
                df,
                projection,
                selection,
                ..
            } => {
                let selection = selection
                    .map(|pred| self.create_physical_expr(pred, Context::Other))
                    .map_or(Ok(None), |v| v.map(Some))?;
                let projection = projection
                    .map(|proj| self.create_physical_expressions(proj, Context::Other))
                    .map_or(Ok(None), |v| v.map(Some))?;
                Ok(Box::new(DataFrameExec::new(df, projection, selection)))
            }
            LogicalPlan::Sort {
                input,
                by_column,
                reverse,
            } => {
                let input = self.create_initial_physical_plan(*input)?;
                Ok(Box::new(SortExec {
                    input,
                    by_column,
                    reverse,
                }))
            }
            LogicalPlan::Explode { input, columns } => {
                let input = self.create_initial_physical_plan(*input)?;
                Ok(Box::new(ExplodeExec { input, columns }))
            }
            LogicalPlan::Cache { input } => {
                let fields = input.schema().fields();
                // todo! fix the unique constraint in the schema. Probably in projection pushdown at joins
                let mut unique =
                    HashSet::with_capacity_and_hasher(fields.len(), RandomState::default());
                // assumption of 80 characters per column name
                let mut key = String::with_capacity(fields.len() * 80);
                for field in fields {
                    if unique.insert(field.name()) {
                        key.push_str(field.name())
                    }
                }
                let input = self.create_initial_physical_plan(*input)?;
                Ok(Box::new(CacheExec { input, key }))
            }
            LogicalPlan::Distinct {
                input,
                maintain_order,
                subset,
            } => {
                let input = self.create_initial_physical_plan(*input)?;
                let subset = Arc::try_unwrap(subset).unwrap_or_else(|subset| (*subset).clone());
                Ok(Box::new(DropDuplicatesExec {
                    input,
                    maintain_order,
                    subset,
                }))
            }
            LogicalPlan::Aggregate {
                input,
                keys,
                aggs,
                apply,
                ..
            } => {
                let input = self.create_initial_physical_plan(*input)?;
                let mut partitionable = true;

                for agg in &aggs {
                    match agg {
                        Expr::Agg(AggExpr::Min(_))
                        | Expr::Agg(AggExpr::Max(_))
                        | Expr::Agg(AggExpr::Sum(_))
                        | Expr::Agg(AggExpr::Count(_))
                        | Expr::Agg(AggExpr::Last(_))
                        | Expr::Agg(AggExpr::First(_)) => {}
                        _ => {
                            partitionable = false;
                        }
                    }
                }
                let phys_keys = self.create_physical_expressions(
                    Arc::try_unwrap(keys).unwrap_or_else(|keys| (&*keys).clone()),
                    Context::Other,
                )?;
                if partitionable {
                    let phys_aggs =
                        self.create_physical_expressions(aggs.clone(), Context::Aggregation)?;
                    Ok(Box::new(PartitionGroupByExec::new(
                        input, phys_keys, phys_aggs, aggs,
                    )))
                } else {
                    let phys_aggs = self.create_physical_expressions(aggs, Context::Aggregation)?;
                    Ok(Box::new(GroupByExec::new(
                        input, phys_keys, phys_aggs, apply,
                    )))
                }
            }
            LogicalPlan::Join {
                input_left,
                input_right,
                how,
                left_on,
                right_on,
                allow_par,
                force_par,
                ..
            } => {
                let parallel = if force_par {
                    force_par
                } else if allow_par {
                    // check if two DataFrames come from a separate source. If they don't we hope it is cached.
                    let mut sources_left =
                        HashSet::with_capacity_and_hasher(32, RandomState::default());
                    agg_source_paths(&input_left, &mut sources_left);
                    let mut sources_right =
                        HashSet::with_capacity_and_hasher(32, RandomState::default());
                    agg_source_paths(&input_right, &mut sources_right);
                    sources_left
                        .intersection(&sources_right)
                        .collect_vec()
                        .is_empty()
                } else {
                    false
                };

                let input_left = self.create_initial_physical_plan(*input_left)?;
                let input_right = self.create_initial_physical_plan(*input_right)?;
                let left_on = self.create_physical_expressions(left_on, Context::Other)?;
                let right_on = self.create_physical_expressions(right_on, Context::Other)?;
                Ok(Box::new(JoinExec::new(
                    input_left,
                    input_right,
                    how,
                    left_on,
                    right_on,
                    parallel,
                )))
            }
            LogicalPlan::HStack { input, exprs, .. } => {
                let input = self.create_initial_physical_plan(*input)?;
                let phys_expr = self.create_physical_expressions(exprs, Context::Other)?;
                Ok(Box::new(StackExec::new(input, phys_expr)))
            }
            LogicalPlan::Udf {
                input, function, ..
            } => {
                let input = self.create_initial_physical_plan(*input)?;
                Ok(Box::new(UdfExec { input, function }))
            }
        }
    }

    pub fn create_physical_expr(
        &self,
        expression: Expr,
        ctxt: Context,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        match expression.clone() {
            Expr::Window {
                function,
                partition_by,
                order_by: _,
            } => {
                // TODO! Order by
                let group_column = expr_to_root_column_name(&*partition_by)
                    .expect("need a partition_by column for a window function");
                let out_name;
                let apply_column = expr_to_root_column_name(&*function)
                    .expect("need a root column for a window function");

                let mut function = *function;
                if let Expr::Alias(expr, name) = function {
                    function = *expr;
                    out_name = name;
                } else {
                    out_name = group_column.clone();
                }

                Ok(Arc::new(WindowExpr {
                    group_column,
                    apply_column,
                    out_name,
                    function,
                }))
            }
            Expr::Literal(value) => Ok(Arc::new(LiteralExpr::new(value, expression))),
            Expr::BinaryExpr { left, op, right } => {
                let lhs = self.create_physical_expr(*left, ctxt)?;
                let rhs = self.create_physical_expr(*right, ctxt)?;
                Ok(Arc::new(BinaryExpr::new(lhs, op, rhs, expression)))
            }
            Expr::Column(column) => Ok(Arc::new(ColumnExpr::new(column, expression))),
            Expr::Sort { expr, reverse } => {
                let phys_expr = self.create_physical_expr(*expr, ctxt)?;
                Ok(Arc::new(SortExpr::new(phys_expr, reverse, expression)))
            }
            Expr::Not(expr) => {
                let phys_expr = self.create_physical_expr(*expr, ctxt)?;
                Ok(Arc::new(NotExpr::new(phys_expr, expression)))
            }
            Expr::Alias(expr, name) => {
                let phys_expr = self.create_physical_expr(*expr, ctxt)?;
                Ok(Arc::new(AliasExpr::new(phys_expr, name, expression)))
            }
            Expr::IsNull(expr) => {
                let phys_expr = self.create_physical_expr(*expr, ctxt)?;
                Ok(Arc::new(IsNullExpr::new(phys_expr, expression)))
            }
            Expr::IsNotNull(expr) => {
                let phys_expr = self.create_physical_expr(*expr, ctxt)?;
                Ok(Arc::new(IsNotNullExpr::new(phys_expr, expression)))
            }
            Expr::Agg(agg) => {
                match agg {
                    AggExpr::Min(expr) => {
                        // todo! Output type is dependent on schema.
                        let input = self.create_physical_expr(*expr, ctxt)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(PhysicalAggExpr::new(input, GroupByMethod::Min)))
                            }
                            Context::Other => {
                                let function = Arc::new(move |s: Series| Ok(s.min_as_series()));
                                Ok(Arc::new(ApplyExpr {
                                    input,
                                    function,
                                    output_type: None,
                                    expr: expression,
                                }))
                            }
                        }
                    }
                    AggExpr::Max(expr) => {
                        let input = self.create_physical_expr(*expr, ctxt)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(PhysicalAggExpr::new(input, GroupByMethod::Max)))
                            }
                            Context::Other => {
                                let function = Arc::new(move |s: Series| Ok(s.max_as_series()));
                                Ok(Arc::new(ApplyExpr {
                                    input,
                                    function,
                                    output_type: None,
                                    expr: expression,
                                }))
                            }
                        }
                    }
                    AggExpr::Sum(expr) => {
                        let input = self.create_physical_expr(*expr, ctxt)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(PhysicalAggExpr::new(input, GroupByMethod::Sum)))
                            }
                            Context::Other => {
                                let function = Arc::new(move |s: Series| Ok(s.sum_as_series()));
                                Ok(Arc::new(ApplyExpr {
                                    input,
                                    function,
                                    output_type: None,
                                    expr: expression,
                                }))
                            }
                        }
                    }
                    AggExpr::Std(expr) => {
                        let input = self.create_physical_expr(*expr, ctxt)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(PhysicalAggExpr::new(input, GroupByMethod::Std)))
                            }
                            Context::Other => {
                                let function = Arc::new(move |s: Series| Ok(s.std_as_series()));
                                Ok(Arc::new(ApplyExpr {
                                    input,
                                    function,
                                    output_type: None,
                                    expr: expression,
                                }))
                            }
                        }
                    }
                    AggExpr::Var(expr) => {
                        let input = self.create_physical_expr(*expr, ctxt)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(PhysicalAggExpr::new(input, GroupByMethod::Var)))
                            }
                            Context::Other => {
                                let function = Arc::new(move |s: Series| Ok(s.var_as_series()));
                                Ok(Arc::new(ApplyExpr {
                                    input,
                                    function,
                                    output_type: None,
                                    expr: expression,
                                }))
                            }
                        }
                    }
                    AggExpr::Mean(expr) => {
                        let input = self.create_physical_expr(*expr, ctxt)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(PhysicalAggExpr::new(input, GroupByMethod::Mean)))
                            }
                            Context::Other => {
                                let function = Arc::new(move |s: Series| Ok(s.mean_as_series()));
                                Ok(Arc::new(ApplyExpr {
                                    input,
                                    function,
                                    output_type: None,
                                    expr: expression,
                                }))
                            }
                        }
                    }
                    AggExpr::Median(expr) => {
                        let input = self.create_physical_expr(*expr, ctxt)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(PhysicalAggExpr::new(input, GroupByMethod::Median)))
                            }
                            Context::Other => {
                                let function = Arc::new(move |s: Series| Ok(s.median_as_series()));
                                Ok(Arc::new(ApplyExpr {
                                    input,
                                    function,
                                    output_type: None,
                                    expr: expression,
                                }))
                            }
                        }
                    }
                    AggExpr::First(expr) => {
                        let input = self.create_physical_expr(*expr, ctxt)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(PhysicalAggExpr::new(input, GroupByMethod::First)))
                            }
                            Context::Other => todo!(),
                        }
                    }
                    AggExpr::Last(expr) => {
                        let input = self.create_physical_expr(*expr, ctxt)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(PhysicalAggExpr::new(input, GroupByMethod::Last)))
                            }
                            Context::Other => todo!(),
                        }
                    }
                    AggExpr::List(expr) => {
                        let input = self.create_physical_expr(*expr, ctxt)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(PhysicalAggExpr::new(input, GroupByMethod::List)))
                            }
                            Context::Other => {
                                panic!(
                                    "list expression is only supported in the aggregation context"
                                )
                            }
                        }
                    }
                    AggExpr::NUnique(expr) => {
                        let input = self.create_physical_expr(*expr, ctxt)?;
                        match ctxt {
                            Context::Aggregation => Ok(Arc::new(PhysicalAggExpr::new(
                                input,
                                GroupByMethod::NUnique,
                            ))),
                            Context::Other => {
                                let function = Arc::new(move |s: Series| {
                                    s.n_unique().map(|count| {
                                        UInt32Chunked::new_from_slice(s.name(), &[count as u32])
                                            .into_series()
                                    })
                                });
                                Ok(Arc::new(ApplyExpr {
                                    input,
                                    function,
                                    output_type: Some(DataType::UInt32),
                                    expr: expression,
                                }))
                            }
                        }
                    }
                    AggExpr::Quantile { expr, quantile } => {
                        // todo! add schema to get correct output type
                        let input = self.create_physical_expr(*expr, ctxt)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggQuantileExpr::new(input, quantile)))
                            }
                            Context::Other => {
                                let function =
                                    Arc::new(move |s: Series| s.quantile_as_series(quantile));
                                Ok(Arc::new(ApplyExpr {
                                    input,
                                    function,
                                    output_type: None,
                                    expr: expression,
                                }))
                            }
                        }
                    }
                    AggExpr::AggGroups(expr) => {
                        if let Context::Other = ctxt {
                            panic!("agg groups expression only supported in aggregation context")
                        }
                        let phys_expr = self.create_physical_expr(*expr, ctxt)?;
                        Ok(Arc::new(PhysicalAggExpr::new(
                            phys_expr,
                            GroupByMethod::Groups,
                        )))
                    }
                    AggExpr::Count(expr) => {
                        let input = self.create_physical_expr(*expr, ctxt)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(PhysicalAggExpr::new(input, GroupByMethod::Count)))
                            }
                            Context::Other => {
                                let function = Arc::new(move |s: Series| {
                                    let count = s.len();
                                    Ok(UInt32Chunked::new_from_slice(s.name(), &[count as u32])
                                        .into_series())
                                });
                                Ok(Arc::new(ApplyExpr {
                                    input,
                                    function,
                                    output_type: Some(DataType::UInt32),
                                    expr: expression,
                                }))
                            }
                        }
                    }
                }
            }
            Expr::Cast { expr, data_type } => {
                let phys_expr = self.create_physical_expr(*expr, ctxt)?;
                Ok(Arc::new(CastExpr::new(phys_expr, data_type)))
            }
            Expr::Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                let predicate = self.create_physical_expr(*predicate, ctxt)?;
                let truthy = self.create_physical_expr(*truthy, ctxt)?;
                let falsy = self.create_physical_expr(*falsy, ctxt)?;
                Ok(Arc::new(TernaryExpr {
                    predicate,
                    truthy,
                    falsy,
                    expr: expression,
                }))
            }
            Expr::Udf {
                input,
                function,
                output_type,
            } => {
                let input = self.create_physical_expr(*input, ctxt)?;
                Ok(Arc::new(ApplyExpr {
                    input,
                    function,
                    output_type,
                    expr: expression,
                }))
            }
            Expr::BinaryFunction {
                input_a,
                input_b,
                function,
                output_field,
            } => {
                let input_a = self.create_physical_expr(*input_a, ctxt)?;
                let input_b = self.create_physical_expr(*input_b, ctxt)?;
                Ok(Arc::new(BinaryFunction {
                    input_a,
                    input_b,
                    function,
                    output_field,
                }))
            }
            Expr::Shift { input, periods } => {
                let input = self.create_physical_expr(*input, ctxt)?;
                let function = Arc::new(move |s: Series| s.shift(periods));
                Ok(Arc::new(ApplyExpr::new(input, function, None, expression)))
            }
            Expr::Slice {
                input,
                offset,
                length,
            } => {
                let input = self.create_physical_expr(*input, ctxt)?;
                Ok(Arc::new(SliceExpr {
                    input,
                    offset,
                    len: length,
                }))
            }
            Expr::Reverse(expr) => {
                let input = self.create_physical_expr(*expr, ctxt)?;
                let function = Arc::new(move |s: Series| Ok(s.reverse()));
                Ok(Arc::new(ApplyExpr::new(input, function, None, expression)))
            }
            Expr::Duplicated(expr) => {
                let input = self.create_physical_expr(*expr, ctxt)?;
                let function =
                    Arc::new(move |s: Series| s.is_duplicated().map(|ca| ca.into_series()));
                Ok(Arc::new(ApplyExpr::new(input, function, None, expression)))
            }
            Expr::Unique(expr) => {
                let input = self.create_physical_expr(*expr, ctxt)?;
                let function = Arc::new(move |s: Series| s.is_unique().map(|ca| ca.into_series()));
                Ok(Arc::new(ApplyExpr::new(input, function, None, expression)))
            }
            Expr::Explode(expr) => {
                let input = self.create_physical_expr(*expr, ctxt)?;
                let function = Arc::new(move |s: Series| s.explode());
                Ok(Arc::new(ApplyExpr::new(input, function, None, expression)))
            }
            Expr::Wildcard => panic!("should be no wildcard at this point"),
        }
    }
}
