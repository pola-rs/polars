use crate::physical_plan::executors::evaluate_physical_expressions;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

/// Take an input Executor (creates the input DataFrame)
/// and a multiple PhysicalExpressions (create the output Series)
pub struct StandardExec {
    /// i.e. sort, projection
    #[allow(dead_code)]
    operation: &'static str,
    input: Box<dyn Executor>,
    expr: Vec<Arc<dyn PhysicalExpr>>,
}

impl StandardExec {
    pub(crate) fn new(
        operation: &'static str,
        input: Box<dyn Executor>,
        expr: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        Self {
            operation,
            input,
            expr,
        }
    }
}

impl Executor for StandardExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let df = self.input.execute(state)?;

        let df = evaluate_physical_expressions(&df, &self.expr, state);
        state.clear_expr_cache();
        df
    }
}
