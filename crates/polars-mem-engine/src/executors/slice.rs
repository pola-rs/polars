use super::*;

pub struct SliceExec {
    pub input: Box<dyn Executor>,
    pub slice: Slice,
}

impl Executor for SliceExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run SliceExec")
            }
        }
        let df = self.input.execute(state)?;
        Ok(df.slice(self.slice))
    }
}
