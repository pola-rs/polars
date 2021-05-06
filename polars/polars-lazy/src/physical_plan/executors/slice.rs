use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

pub struct SliceExec {
    pub input: Box<dyn Executor>,
    pub offset: i64,
    pub len: usize,
}

impl Executor for SliceExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let df = self.input.execute(state)?;
        Ok(df.slice(self.offset, self.len))
    }
}
