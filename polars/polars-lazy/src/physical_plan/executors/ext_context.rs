use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

pub struct ExternalContext {
    pub input: Box<dyn Executor>,
    pub contexts: Vec<Box<dyn Executor>>,
}

impl Executor for ExternalContext {
    fn execute(&mut self, state: &mut ExecutionState) -> Result<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run ExternalContext")
            }
        }
        let df = self.input.execute(state)?;
        let contexts = self
            .contexts
            .iter_mut()
            .map(|e| e.execute(state))
            .collect::<Result<Vec<_>>>()?;

        state.ext_contexts = Arc::new(contexts);
        Ok(df)
    }
}
