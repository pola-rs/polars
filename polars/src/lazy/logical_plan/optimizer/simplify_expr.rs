use crate::lazy::prelude::*;
use crate::prelude::*;

pub struct SimplifyExpr {}

// Evaluates x + y if possible
fn eval_plus(left: &Expr, right: &Expr) -> Option<Expr> {
    match (left, right) {
        (Expr::Literal(ScalarValue::Float32(x)), Expr::Literal(ScalarValue::Float32(y))) => {
            Some(Expr::Literal(ScalarValue::Float32(x + y)))
        }
        (Expr::Literal(ScalarValue::Float64(x)), Expr::Literal(ScalarValue::Float64(y))) => {
            Some(Expr::Literal(ScalarValue::Float64(x + y)))
        }
        (Expr::Literal(ScalarValue::Int8(x)), Expr::Literal(ScalarValue::Int8(y))) => {
            Some(Expr::Literal(ScalarValue::Int8(x + y)))
        }
        (Expr::Literal(ScalarValue::Int16(x)), Expr::Literal(ScalarValue::Int16(y))) => {
            Some(Expr::Literal(ScalarValue::Int16(x + y)))
        }
        (Expr::Literal(ScalarValue::Int32(x)), Expr::Literal(ScalarValue::Int32(y))) => {
            Some(Expr::Literal(ScalarValue::Int32(x + y)))
        }
        (Expr::Literal(ScalarValue::Int64(x)), Expr::Literal(ScalarValue::Int64(y))) => {
            Some(Expr::Literal(ScalarValue::Int64(x + y)))
        }
        (Expr::Literal(ScalarValue::UInt8(x)), Expr::Literal(ScalarValue::UInt8(y))) => {
            Some(Expr::Literal(ScalarValue::UInt8(x + y)))
        }
        (Expr::Literal(ScalarValue::UInt16(x)), Expr::Literal(ScalarValue::UInt16(y))) => {
            Some(Expr::Literal(ScalarValue::UInt16(x + y)))
        }
        (Expr::Literal(ScalarValue::UInt32(x)), Expr::Literal(ScalarValue::UInt32(y))) => {
            Some(Expr::Literal(ScalarValue::UInt32(x + y)))
        }
        (Expr::Literal(ScalarValue::UInt64(x)), Expr::Literal(ScalarValue::UInt64(y))) => {
            Some(Expr::Literal(ScalarValue::UInt64(x + y)))
        }

        _ => None,
    }
}

impl SimplifyExpr {
    /// Traverse the expressions from a level in the logical plan and maybe cast them.
    fn rewrite_expressions(&self, exprs: Vec<Expr>) -> Result<Vec<Expr>> {
        exprs
            .into_iter()
            .map(|expr| self.rewrite_expr(expr))
            .collect()
    }

    fn rewrite_expr(&self, expr: Expr) -> Result<Expr> {
        // the important expression is BinaryExpr. The rest just traverses the tree.
        use Expr::*;
        match expr {
            Expr::BinaryExpr {
                left,
                op: Operator::Plus,
                right,
            } => {
                let l = self.rewrite_expr(*left)?;
                let r = self.rewrite_expr(*right)?;

                if let Some(x) = eval_plus(&l, &r) {
                    Ok(x)
                } else {
                    Ok(BinaryExpr {
                        left: Box::new(l),
                        op: Operator::Plus,
                        right: Box::new(r),
                    })
                }
            }
            Expr::BinaryExpr { left, op, right } => {
                let l = self.rewrite_expr(*left)?;
                let r = self.rewrite_expr(*right)?;
                Ok(Expr::BinaryExpr {
                    left: Box::new(l),
                    op,
                    right: Box::new(r),
                })
            }
            Alias(expr, name) => Ok(Expr::Alias(Box::new(self.rewrite_expr(*expr)?), name)),
            Not(expr) => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.not())
            }
            IsNotNull(expr) => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.is_not_null())
            }
            IsNull(expr) => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.is_null())
            }
            Cast { expr, data_type } => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.cast(data_type))
            }
            Sort { expr, reverse } => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.sort(reverse))
            }
            AggMin(expr) => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.agg_min())
            }
            AggMax(expr) => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.agg_max())
            }
            AggMedian(expr) => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.agg_median())
            }
            AggNUnique(expr) => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.agg_n_unique())
            }
            AggFirst(expr) => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.agg_first())
            }
            AggLast(expr) => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.agg_last())
            }
            AggList(expr) => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.agg_list())
            }
            AggMean(expr) => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.agg_mean())
            }
            AggQuantile { expr, quantile } => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.agg_quantile(quantile))
            }
            AggSum(expr) => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.agg_sum())
            }
            AggGroups(expr) => {
                let expr = self.rewrite_expr(*expr)?;
                Ok(expr.agg_groups())
            }
            Shift { input, periods } => {
                let input = self.rewrite_expr(*input)?;
                Ok(Shift {
                    input: Box::new(input),
                    periods,
                })
            }
            x => Ok(x),
        }
    }

    fn simplify_expr(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan> {
        use LogicalPlan::*;
        match logical_plan {
            Selection { input, predicate } => {
                let predicate = self.rewrite_expr(predicate)?;
                let input = Box::new(self.simplify_expr(*input)?);
                Ok(Selection { input, predicate })
            }

            Projection {
                expr,
                input,
                schema,
            } => {
                let expr = self.rewrite_expressions(expr)?;
                let input = self.simplify_expr(*input)?;

                Ok(Projection {
                    expr,
                    input: Box::new(input),
                    schema,
                })
            }
            DataFrameOp { input, operation } => {
                let input = self.simplify_expr(*input)?;
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
                let input = Box::new(self.simplify_expr(*input)?);
                let aggs = self.rewrite_expressions(aggs)?;
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
                let input_left = Box::new(self.simplify_expr(*input_left)?);
                let input_right = Box::new(self.simplify_expr(*input_right)?);
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
                let input = self.simplify_expr(*input)?;
                let exprs = self.rewrite_expressions(exprs)?;
                Ok(LogicalPlanBuilder::from(input).with_columns(exprs).build())
            }
            logical_plan => Ok(logical_plan),
        }
    }
}

impl Optimize for SimplifyExpr {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan> {
        self.simplify_expr(logical_plan)
    }
}
