use crate::lazy::prelude::*;
use crate::prelude::*;
use crate::utils::get_supertype;

pub struct TypeCoercion {}

impl TypeCoercion {
    /// Traverse the expressions from a level in the logical plan and maybe cast them.
    fn rewrite_expressions(&self, exprs: Vec<Expr>, input_schema: &Schema) -> Result<Vec<Expr>> {
        exprs
            .into_iter()
            .map(|expr| self.rewrite_expr(expr, input_schema))
            .collect()
    }

    fn rewrite_expr(&self, expr: Expr, input_schema: &Schema) -> Result<Expr> {
        // the important expression is BinaryExpr. The rest just traverses the tree.
        use Expr::*;
        match expr {
            Alias(expr, name) => Ok(Expr::Alias(
                Box::new(self.rewrite_expr(*expr, input_schema)?),
                name,
            )),
            Column(_) => Ok(expr.clone()),
            Literal(_) => Ok(expr.clone()),
            BinaryExpr { left, op, right } => {
                let left = self.rewrite_expr(*left, input_schema)?;
                let right = self.rewrite_expr(*right, input_schema)?;

                let type_left = left.get_type(input_schema)?;
                let type_right = right.get_type(input_schema)?;
                if type_left == type_right {
                    Ok(binary_expr(left, op, right))
                } else {
                    let st = get_supertype(&type_left, &type_right)?;
                    Ok(binary_expr(left.cast(st.clone()), op, right.cast(st)))
                }
            }
            Not(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.not())
            }
            IsNotNull(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.is_not_null())
            }
            IsNull(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.is_null())
            }
            Cast { expr, data_type } => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.cast(data_type))
            }
            Sort { expr, reverse } => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.sort(reverse))
            }
            AggMin(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.agg_min())
            }
            AggMax(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.agg_max())
            }
            AggMedian(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.agg_median())
            }
            AggNUnique(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.agg_n_unique())
            }
            AggFirst(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.agg_first())
            }
            AggLast(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.agg_last())
            }
            AggList(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.agg_list())
            }
            AggMean(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.agg_mean())
            }
            AggQuantile { expr, quantile } => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.agg_quantile(quantile))
            }
            AggSum(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.agg_sum())
            }
            AggGroups(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.agg_groups())
            }
            AggCount(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.agg_count())
            }
            Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                let predicate = self.rewrite_expr(*predicate, input_schema)?;
                let truthy = self.rewrite_expr(*truthy, input_schema)?;
                let falsy = self.rewrite_expr(*falsy, input_schema)?;
                let type_true = truthy.get_type(input_schema)?;
                let type_false = falsy.get_type(input_schema)?;

                if type_true == type_false {
                    Ok(ternary_expr(predicate, truthy, falsy))
                } else {
                    let st = get_supertype(&type_true, &type_false)?;
                    Ok(ternary_expr(
                        predicate,
                        truthy.cast(st.clone()),
                        falsy.cast(st),
                    ))
                }
            }
            Apply {
                input,
                function,
                output_type,
            } => {
                // todo Maybe don't coerce these types, as it may interfere with the function types.
                let input = self.rewrite_expr(*input, input_schema)?;
                Ok(Apply {
                    input: Box::new(input),
                    function,
                    output_type,
                })
            }
            Shift { input, periods } => {
                let input = self.rewrite_expr(*input, input_schema)?;
                Ok(Shift {
                    input: Box::new(input),
                    periods,
                })
            }
            Wildcard => panic!("should be no wildcard at this point"),
        }
    }

    fn coerce(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan> {
        use LogicalPlan::*;
        match logical_plan {
            Selection { input, predicate } => {
                let predicate = self.rewrite_expr(predicate, input.schema())?;
                let input = Box::new(self.coerce(*input)?);
                Ok(Selection { input, predicate })
            }
            CsvScan { .. } => Ok(logical_plan),
            DataFrameScan { .. } => Ok(logical_plan),
            Projection {
                expr,
                input,
                schema,
            } => {
                let expr = self.rewrite_expressions(expr, input.schema())?;
                Ok(Projection {
                    expr,
                    input,
                    schema,
                })
            }
            DataFrameOp { input, operation } => {
                let input = self.coerce(*input)?;
                Ok(DataFrameOp {
                    input: Box::new(input),
                    operation,
                })
            }
            Aggregate {
                input,
                keys,
                aggs,
                schema,
            } => {
                let input = Box::new(self.coerce(*input)?);
                let aggs = self.rewrite_expressions(aggs, input.schema())?;
                Ok(Aggregate {
                    input,
                    keys,
                    aggs,
                    schema,
                })
            }
            Join {
                input_left,
                input_right,
                schema,
                how,
                left_on,
                right_on,
            } => {
                let input_left = Box::new(self.coerce(*input_left)?);
                let input_right = Box::new(self.coerce(*input_right)?);
                Ok(Join {
                    input_left,
                    input_right,
                    schema,
                    how,
                    left_on,
                    right_on,
                })
            }
            HStack { input, exprs, .. } => {
                let input = self.coerce(*input)?;
                let exprs = self.rewrite_expressions(exprs, input.schema())?;
                Ok(LogicalPlanBuilder::from(input).with_columns(exprs).build())
            }
        }
    }
}

impl Optimize for TypeCoercion {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan> {
        self.coerce(logical_plan)
    }
}
