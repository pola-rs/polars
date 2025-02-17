use super::*;

pub struct SliceExec {
    pub input: Box<dyn Executor>,
    pub offset: i64,
    pub len: IdxSize,
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

        state.record(
            || Ok(df.slice(self.offset, self.len as usize)),
            "slice".into(),
        )
    }
}
