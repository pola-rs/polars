use crate::physical_plan::executors::evaluate_physical_expressions;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

/// Take an input Executor (creates the input DataFrame)
/// and a multiple PhysicalExpressions (create the output Series)
pub struct ProjectionExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) expr: Vec<Arc<dyn PhysicalExpr>>,
    #[cfg(test)]
    pub(crate) schema: SchemaRef,
}

impl Executor for ProjectionExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let df = self.input.execute(state)?;

        let df = evaluate_physical_expressions(&df, &self.expr, state);

        #[cfg(test)]
        {
            // TODO: check also the types.
            df.as_ref().map(|df| {
                for (l, r) in df.schema().fields().iter().zip(self.schema.fields()) {
                    assert_eq!(l.name(), r.name());
                }
            });
        }

        state.clear_expr_cache();
        df
    }
}
