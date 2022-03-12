use crate::physical_plan::executors::evaluate_physical_expressions;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

/// Take an input Executor (creates the input DataFrame)
/// and a multiple PhysicalExpressions (create the output Series)
pub struct ProjectionExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) expr: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) has_windows: bool,
    pub(crate) input_schema: SchemaRef,
    #[cfg(test)]
    pub(crate) schema: SchemaRef,
}

impl Executor for ProjectionExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let df = self.input.execute(state)?;
        state.set_schema(self.input_schema.clone());

        let df = evaluate_physical_expressions(&df, &self.expr, state, self.has_windows);

        // this only runs during testing and check if the runtime type matches the predicted schema
        #[cfg(test)]
        #[allow(unused_must_use)]
        {
            // TODO: check also the types.
            df.as_ref().map(|df| {
                for (l, r) in df.iter().zip(self.schema.iter_names()) {
                    assert_eq!(l.name(), r);
                }
            });
        }

        state.clear_expr_cache();
        df
    }
}
