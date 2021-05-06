use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;
pub struct StackExec {
    input: Box<dyn Executor>,
    expr: Vec<Arc<dyn PhysicalExpr>>,
}

impl StackExec {
    pub(crate) fn new(input: Box<dyn Executor>, expr: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        Self { input, expr }
    }
}

impl Executor for StackExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let mut df = self.input.execute(state)?;
        let height = df.height();

        let res: Result<_> = self.expr.iter().try_for_each(|expr| {
            let s = expr.evaluate(&df, state).map(|series| {
                // literal series. Should be whole column size
                if series.len() == 1 && height > 1 {
                    series.expand_at_index(0, height)
                } else {
                    series
                }
            })?;

            let name = s.name().to_string();
            df.replace_or_add(&name, s)?;
            Ok(())
        });
        let _ = res?;
        state.clear_expr_cache();
        Ok(df)
    }
}
