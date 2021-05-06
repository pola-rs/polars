use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

pub struct MeltExec {
    pub input: Box<dyn Executor>,
    pub id_vars: Arc<Vec<String>>,
    pub value_vars: Arc<Vec<String>>,
}

impl Executor for MeltExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let df = self.input.execute(state)?;
        df.melt(&self.id_vars.as_slice(), &self.value_vars.as_slice())
    }
}
