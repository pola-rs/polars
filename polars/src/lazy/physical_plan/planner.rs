use super::{
    executors::{CsvExec, FilterExec, ProjectionExec},
    expressions::LiteralExpr,
    *,
};
use crate::lazy::physical_plan::executors::DataFrameExec;

pub(crate) struct SimplePlanner {}
impl Default for SimplePlanner {
    fn default() -> Self {
        SimplePlanner {}
    }
}

impl PhysicalPlanner for SimplePlanner {
    fn create_physical_plan(&self, logical_plan: &LogicalPlan) -> Result<Rc<dyn ExecutionPlan>> {
        self.create_initial_physical_plan(logical_plan)
    }
}

impl SimplePlanner {
    pub fn create_initial_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
    ) -> Result<Rc<dyn ExecutionPlan>> {
        match logical_plan {
            LogicalPlan::Filter { input, predicate } => {
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
            LogicalPlan::Projection { columns, input } => {
                let input = self.create_initial_physical_plan(input)?;
                let columns = columns
                    .iter()
                    .map(|expr| self.create_physical_expr(expr))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Rc::new(ProjectionExec::new(input, columns)))
            }
            LogicalPlan::DataFrameScan { df } => Ok(Rc::new(DataFrameExec::new(df.clone()))),
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
            e => panic!(format!("physical expr. for expr: {:?} not implemented", e)),
        }
    }
}
