use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

pub(crate) struct ExplodeExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) columns: Vec<String>,
}

impl Executor for ExplodeExec {
    fn execute(&mut self, state: &mut ExecutionState) -> Result<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run ExplodeExec")
            }
        }
        let df = self.input.execute(state)?;
        df.explode(&self.columns)
    }
}
