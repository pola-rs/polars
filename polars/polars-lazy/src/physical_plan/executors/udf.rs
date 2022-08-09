use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

pub(crate) struct UdfExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) function: Arc<dyn DataFrameUdf>,
}

impl Executor for UdfExec {
    fn execute(&mut self, state: &mut ExecutionState) -> Result<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run UdfExec")
            }
        }
        let df = self.input.execute(state)?;
        self.function.call_udf(df)
    }
}
