use crate::physical_plan::executors::execute_projection_cached_window_fns;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::{prelude::*, POOL};
use rayon::prelude::*;

pub struct StackExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) has_windows: bool,
    pub(crate) expr: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) input_schema: SchemaRef,
}

impl Executor for StackExec {
    fn execute(&mut self, state: &mut ExecutionState) -> Result<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run StackExec")
            }
        }
        let mut df = self.input.execute(state)?;

        state.set_schema(self.input_schema.clone());
        let res = if self.has_windows {
            // we have a different run here
            // to ensure the window functions run sequential and share caches
            execute_projection_cached_window_fns(&df, &self.expr, state)?
        } else {
            POOL.install(|| {
                self.expr
                    .par_iter()
                    .map(|expr| expr.evaluate(&df, state))
                    .collect::<Result<Vec<_>>>()
            })?
        };
        state.clear_schema_cache();
        state.clear_expr_cache();

        let schema = &*self.input_schema;
        for (i, s) in res.into_iter().enumerate() {
            // we need to branch here
            // because users can add multiple columns with the same name
            if i == 0 || schema.get(s.name()).is_some() {
                df.with_column_and_schema(s, schema)?;
            } else {
                df.with_column(s.clone())?;
            }
        }

        Ok(df)
    }
}
