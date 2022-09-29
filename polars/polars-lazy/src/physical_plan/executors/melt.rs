use super::*;

pub struct MeltExec {
    pub input: Box<dyn Executor>,
    pub args: Arc<MeltArgs>,
}

impl Executor for MeltExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run MeltExec")
            }
        }
        let df = self.input.execute(state)?;
        let args = std::mem::take(Arc::make_mut(&mut self.args));

        state.record(|| df.melt2(args), "melt()".into())
    }
}
