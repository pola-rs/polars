use crate::physical_plan::executors::execute_projection_cached_window_fns;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::{prelude::*, POOL};
use rayon::prelude::*;

pub struct StackExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) has_windows: bool,
    pub(crate) expr: Vec<Arc<dyn PhysicalExpr>>,
}

impl Executor for StackExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let mut df = self.input.execute(state)?;
        let height = df.height();

        let res = if self.has_windows {
            execute_projection_cached_window_fns(&df, &self.expr, state)?
        } else {
            POOL.install(|| {
                self.expr
                    .par_iter()
                    .map(|expr| {
                        expr.evaluate(&df, state).map(|series| {
                            // literal series. Should be whole column size
                            if series.len() == 1 && height > 1 {
                                series.expand_at_index(0, height)
                            } else {
                                series
                            }
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })?
        };

        for s in res {
            df.with_column(s)?;
        }

        state.clear_expr_cache();
        Ok(df)
    }
}
