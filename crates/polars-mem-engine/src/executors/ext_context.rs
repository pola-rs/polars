use super::*;

pub struct ExternalContext {
    pub input: Box<dyn Executor>,
    pub contexts: Vec<Box<dyn Executor>>,
}

impl Executor for ExternalContext {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run ExternalContext")
            }
        }
        // we evaluate contexts first as input may has pushed exprs.
        let contexts = self
            .contexts
            .iter_mut()
            .map(|e| e.execute(state))
            .collect::<PolarsResult<Vec<_>>>()?;
        state.ext_contexts = Arc::new(contexts);
        let df = self.input.execute(state)?;

        Ok(df)
    }
}
