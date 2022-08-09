use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

pub struct SliceExec {
    pub input: Box<dyn Executor>,
    pub offset: i64,
    pub len: IdxSize,
}

impl Executor for SliceExec {
    fn execute(&mut self, state: &mut ExecutionState) -> Result<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run SciceExec")
            }
        }
        let df = self.input.execute(state)?;
        Ok(df.slice(self.offset, self.len as usize))
    }
}
