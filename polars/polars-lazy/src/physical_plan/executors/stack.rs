use std::borrow::Cow;

use polars_core::prelude::*;
use polars_core::POOL;
use rayon::prelude::*;

use crate::physical_plan::executors::execute_projection_cached_window_fns;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct StackExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) has_windows: bool,
    pub(crate) expr: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) input_schema: SchemaRef,
}

impl StackExec {
    fn execute_impl(&mut self, state: &mut ExecutionState, mut df: DataFrame) -> Result<DataFrame> {
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

impl Executor for StackExec {
    fn execute(&mut self, state: &mut ExecutionState) -> Result<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run StackExec")
            }
        }
        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            let by = self
                .expr
                .iter()
                .map(|s| Ok(s.to_field(&self.input_schema)?.name))
                .collect::<Result<Vec<_>>>()?;
            let name = column_delimited("with_column".to_string(), &by);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        if state.has_node_timer() {
            let new_state = state.clone();
            new_state.record(|| self.execute_impl(state, df), profile_name)
        } else {
            self.execute_impl(state, df)
        }
    }
}
