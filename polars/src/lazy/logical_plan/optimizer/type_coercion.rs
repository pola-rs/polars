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
            Reverse(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.reverse())
            }
            Unique(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.is_unique())
            }
            Duplicated(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.is_duplicated())
            }
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
            Min(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.min())
            }
            Max(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.max())
            }
            Median(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.median())
            }
            NUnique(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.n_unique())
            }
            First(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.first())
            }
            Last(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.last())
            }
            List(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.list())
            }
            Mean(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.mean())
            }
            Quantile { expr, quantile } => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.quantile(quantile))
            }
            Sum(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.sum())
            }
            AggGroups(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.agg_groups())
            }
            Count(expr) => {
                let expr = self.rewrite_expr(*expr, input_schema)?;
                Ok(expr.count())
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
            ParquetScan { .. } => Ok(logical_plan),
            CsvScan { .. } => Ok(logical_plan),
            DataFrameScan { .. } => Ok(logical_plan),
            Projection {
                expr,
                input,
                schema,
            } => {
                let input = Box::new(self.coerce(*input)?);
                let expr = self.rewrite_expressions(expr, input.schema())?;
                Ok(Projection {
                    expr,
                    input,
                    schema,
                })
            }
            LocalProjection {
                expr,
                input,
                schema,
            } => {
                let input = Box::new(self.coerce(*input)?);
                let expr = self.rewrite_expressions(expr, input.schema())?;
                Ok(LocalProjection {
                    expr,
                    input,
                    schema,
                })
            }
            Sort {
                input,
                by_column,
                reverse,
            } => {
                let input = Box::new(self.coerce(*input)?);
                Ok(Sort {
                    input,
                    by_column,
                    reverse,
                })
            }
            Explode { input, column } => {
                let input = Box::new(self.coerce(*input)?);
                Ok(Explode { input, column })
            }
            Cache { input } => {
                let input = Box::new(self.coerce(*input)?);
                Ok(Cache { input })
            }
            Distinct {
                input,
                maintain_order,
                subset,
            } => {
                let input = self.coerce(*input)?;
                Ok(Distinct {
                    input: Box::new(input),
                    maintain_order,
                    subset,
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
