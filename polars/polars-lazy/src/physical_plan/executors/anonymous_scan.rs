use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

pub(crate) struct AnonymousScanExec {
    pub(crate) options: AnonymousScanOptions,
    pub(crate) function: Arc<dyn AnonymousScan>,
}

impl Executor for AnonymousScanExec {
    fn execute(&mut self, _state: &ExecutionState) -> Result<DataFrame> {
        self.function.scan(self.options.clone())
    }
}
