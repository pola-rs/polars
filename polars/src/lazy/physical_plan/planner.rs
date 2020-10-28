use crate::lazy::physical_plan::executors::{JoinExec, StackExec};
use crate::{lazy::prelude::*, prelude::*};
use std::sync::Arc;

pub struct DefaultPlanner {}
impl Default for DefaultPlanner {
    fn default() -> Self {
        Self {}
    }
}

impl PhysicalPlanner for DefaultPlanner {
    fn create_physical_plan(&self, logical_plan: LogicalPlan) -> Result<Arc<dyn Executor>> {
        self.create_initial_physical_plan(logical_plan)
    }
}

impl DefaultPlanner {
    fn create_physical_expressions(&self, exprs: Vec<Expr>) -> Result<Vec<Arc<dyn PhysicalExpr>>> {
        exprs
            .into_iter()
            .map(|e| self.create_physical_expr(e))
            .collect()
    }
    pub fn create_initial_physical_plan(
        &self,
        logical_plan: LogicalPlan,
    ) -> Result<Arc<dyn Executor>> {
        match logical_plan {
            LogicalPlan::Selection { input, predicate } => {
                let input = self.create_initial_physical_plan(*input)?;
                let predicate = self.create_physical_expr(predicate)?;
                Ok(Arc::new(FilterExec::new(predicate, input)))
            }
            LogicalPlan::CsvScan {
                path,
                schema,
                has_header,
                delimiter,
            } => Ok(Arc::new(CsvExec::new(path, schema, has_header, delimiter))),
            LogicalPlan::Projection { expr, input, .. } => {
                let input = self.create_initial_physical_plan(*input)?;
                let phys_expr = self.create_physical_expressions(expr)?;
                Ok(Arc::new(PipeExec::new("projection", input, phys_expr)))
            }
            LogicalPlan::DataFrameScan { df, .. } => Ok(Arc::new(DataFrameExec::new(df))),
            LogicalPlan::DataFrameOp { input, operation } => {
                // this isn't a sort
                let input = self.create_initial_physical_plan(*input)?;

                Ok(Arc::new(DataFrameOpsExec::new(input, operation)))
            }
            LogicalPlan::Aggregate {
                input, keys, aggs, ..
            } => {
                let input = self.create_initial_physical_plan(*input)?;
                let phys_aggs = self.create_physical_expressions(aggs)?;
                Ok(Arc::new(GroupByExec::new(input, keys, phys_aggs)))
            }
            LogicalPlan::Join {
                input_left,
                input_right,
                how,
                left_on,
                right_on,
                ..
            } => {
                let input_left = self.create_initial_physical_plan(*input_left)?;
                let input_right = self.create_initial_physical_plan(*input_right)?;
                let left_on = self.create_physical_expr(left_on)?;
                let right_on = self.create_physical_expr(right_on)?;
                Ok(Arc::new(JoinExec::new(
                    input_left,
                    input_right,
                    how,
                    left_on,
                    right_on,
                )))
            }
            LogicalPlan::HStack { input, exprs, .. } => {
                let input = self.create_initial_physical_plan(*input)?;
                let phys_expr = self.create_physical_expressions(exprs)?;
                Ok(Arc::new(StackExec::new(input, phys_expr)))
            }
        }
    }

    // todo! add schema and ctxt
    pub fn create_physical_expr(&self, expr: Expr) -> Result<Arc<dyn PhysicalExpr>> {
        match expr {
            Expr::Literal(value) => Ok(Arc::new(LiteralExpr::new(value))),
            Expr::BinaryExpr { left, op, right } => {
                let lhs = self.create_physical_expr(*left)?;
                let rhs = self.create_physical_expr(*right)?;
                Ok(Arc::new(BinaryExpr::new(lhs, op, rhs)))
            }
            Expr::Column(column) => Ok(Arc::new(ColumnExpr::new(column))),
            Expr::Sort { expr, reverse } => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(SortExpr::new(phys_expr, reverse)))
            }
            Expr::Not(expr) => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(NotExpr::new(phys_expr)))
            }
            Expr::Alias(expr, name) => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(AliasExpr::new(phys_expr, name)))
            }
            Expr::IsNull(expr) => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(IsNullExpr::new(phys_expr)))
            }
            Expr::IsNotNull(expr) => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(IsNotNullExpr::new(phys_expr)))
            }
            Expr::AggMin(expr) => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(AggMinExpr::new(phys_expr)))
            }
            Expr::AggMax(expr) => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(AggMaxExpr::new(phys_expr)))
            }
            Expr::AggSum(expr) => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(AggSumExpr::new(phys_expr)))
            }
            Expr::AggMean(expr) => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(AggMeanExpr::new(phys_expr)))
            }
            Expr::AggMedian(expr) => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(AggMedianExpr::new(phys_expr)))
            }
            Expr::AggFirst(expr) => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(AggFirstExpr::new(phys_expr)))
            }
            Expr::AggLast(expr) => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(AggLastExpr::new(phys_expr)))
            }
            Expr::AggList(expr) => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(AggListExpr::new(phys_expr)))
            }
            Expr::AggNUnique(expr) => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(AggNUniqueExpr::new(phys_expr)))
            }
            Expr::AggQuantile { expr, quantile } => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(AggQuantileExpr::new(phys_expr, quantile)))
            }
            Expr::AggGroups(expr) => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(AggGroupsExpr::new(phys_expr)))
            }
            Expr::Cast { expr, data_type } => {
                let phys_expr = self.create_physical_expr(*expr)?;
                Ok(Arc::new(CastExpr::new(phys_expr, data_type)))
            }
            Expr::Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                let predicate = self.create_physical_expr(*predicate)?;
                let truthy = self.create_physical_expr(*truthy)?;
                let falsy = self.create_physical_expr(*falsy)?;
                Ok(Arc::new(TernaryExpr {
                    predicate,
                    truthy,
                    falsy,
                }))
            }
            Expr::Apply {
                input,
                function,
                output_type,
            } => {
                let input = self.create_physical_expr(*input)?;
                Ok(Arc::new(ApplyExpr {
                    input,
                    function,
                    output_type,
                }))
            }
            Expr::Shift { input, periods } => {
                let input = self.create_physical_expr(*input)?;
                let function = Arc::new(move |s: Series| s.shift(periods));
                Ok(Arc::new(ApplyExpr::new(input, function, None)))
            }
            Expr::Wildcard => panic!("should be no wildcard at this point"),
        }
    }
}
