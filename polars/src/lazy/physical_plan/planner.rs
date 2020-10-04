use crate::lazy::physical_plan::executors::JoinExec;
use crate::{lazy::prelude::*, prelude::*};
use std::rc::Rc;

pub struct DefaultPlanner {}
impl Default for DefaultPlanner {
    fn default() -> Self {
        Self {}
    }
}

impl PhysicalPlanner for DefaultPlanner {
    fn create_physical_plan(&self, logical_plan: &LogicalPlan) -> Result<Rc<dyn Executor>> {
        self.create_initial_physical_plan(logical_plan)
    }
}

impl DefaultPlanner {
    pub fn create_initial_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
    ) -> Result<Rc<dyn Executor>> {
        match logical_plan {
            LogicalPlan::Selection { input, predicate } => {
                let input = self.create_initial_physical_plan(input)?;
                let predicate = self.create_physical_expr(predicate)?;
                Ok(Rc::new(FilterExec::new(predicate, input)))
            }
            LogicalPlan::CsvScan {
                path,
                schema,
                has_header,
                delimiter,
            } => Ok(Rc::new(CsvExec::new(
                path.clone(),
                schema.clone(),
                *has_header,
                *delimiter,
            ))),
            LogicalPlan::Projection { expr, input, .. } => {
                let input = self.create_initial_physical_plan(input)?;
                let phys_expr = expr
                    .iter()
                    .map(|expr| self.create_physical_expr(expr))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Rc::new(PipeExec::new("projection", input, phys_expr)))
            }
            LogicalPlan::DataFrameScan { df, .. } => Ok(Rc::new(DataFrameExec::new(df.clone()))),
            LogicalPlan::Sort {
                input,
                column,
                reverse,
            } => {
                // this isn't a sort
                let input = self.create_initial_physical_plan(input)?;

                Ok(Rc::new(SortExec::new(input, column.clone(), *reverse)))
            }
            LogicalPlan::Aggregate {
                input, keys, aggs, ..
            } => {
                let input = self.create_initial_physical_plan(input)?;
                let phys_aggs = aggs
                    .iter()
                    .map(|e| self.create_physical_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Rc::new(GroupByExec::new(input, keys.clone(), phys_aggs)))
            }
            LogicalPlan::Join {
                input_left,
                input_right,
                how,
                left_on,
                right_on,
                ..
            } => {
                let input_left = self.create_initial_physical_plan(input_left)?;
                let input_right = self.create_initial_physical_plan(input_right)?;
                Ok(Rc::new(JoinExec::new(
                    input_left,
                    input_right,
                    how.clone(),
                    left_on.clone(),
                    right_on.clone(),
                )))
            }
        }
    }

    // todo! add schema and ctxt
    pub fn create_physical_expr(&self, expr: &Expr) -> Result<Rc<dyn PhysicalExpr>> {
        match expr {
            Expr::Literal(value) => Ok(Rc::new(LiteralExpr::new(value.clone()))),
            Expr::BinaryExpr { left, op, right } => {
                let lhs = self.create_physical_expr(left)?;
                let rhs = self.create_physical_expr(right)?;
                Ok(Rc::new(BinaryExpr::new(lhs.clone(), *op, rhs.clone())))
            }
            Expr::Column(column) => Ok(Rc::new(ColumnExpr::new(column.clone()))),
            Expr::Sort { expr, reverse } => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(SortExpr::new(phys_expr, *reverse)))
            }
            Expr::Not(expr) => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(NotExpr::new(phys_expr)))
            }
            Expr::Alias(expr, name) => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(AliasExpr::new(phys_expr, name.clone())))
            }
            Expr::IsNull(expr) => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(IsNullExpr::new(phys_expr)))
            }
            Expr::IsNotNull(expr) => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(IsNotNullExpr::new(phys_expr)))
            }
            Expr::AggMin(expr) => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(AggMinExpr::new(phys_expr)))
            }
            Expr::AggMax(expr) => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(AggMaxExpr::new(phys_expr)))
            }
            Expr::AggSum(expr) => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(AggSumExpr::new(phys_expr)))
            }
            Expr::AggMean(expr) => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(AggMeanExpr::new(phys_expr)))
            }
            Expr::AggMedian(expr) => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(AggMedianExpr::new(phys_expr)))
            }
            Expr::AggFirst(expr) => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(AggFirstExpr::new(phys_expr)))
            }
            Expr::AggLast(expr) => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(AggLastExpr::new(phys_expr)))
            }
            Expr::AggNUnique(expr) => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(AggNUniqueExpr::new(phys_expr)))
            }
            Expr::AggQuantile { expr, quantile } => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(AggQuantileExpr::new(phys_expr, *quantile)))
            }
            Expr::AggGroups(expr) => {
                let phys_expr = self.create_physical_expr(expr)?;
                Ok(Rc::new(AggGroupsExpr::new(phys_expr)))
            }
        }
    }
}
