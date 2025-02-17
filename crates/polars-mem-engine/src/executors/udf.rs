use super::*;

pub(crate) struct UdfExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) function: FunctionIR,
}

impl Executor for UdfExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run UdfExec")
            }
        }
        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            Cow::Owned(format!("{}", self.function))
        } else {
            Cow::Borrowed("")
        };
        state.record(|| self.function.evaluate(df), profile_name)
    }
}
