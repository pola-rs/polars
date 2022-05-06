use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::explode::MeltArgs;
use polars_core::prelude::*;

pub struct MeltExec {
    pub input: Box<dyn Executor>,
    pub args: Arc<MeltArgs>,
}

impl Executor for MeltExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let df = self.input.execute(state)?;
        let args = std::mem::take(Arc::make_mut(&mut self.args));
        df.melt2(args)
    }
}
