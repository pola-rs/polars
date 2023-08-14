use super::*;

pub(crate) struct UdfExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) function: FunctionNode,
}

impl Executor for UdfExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run UdfExec")
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
