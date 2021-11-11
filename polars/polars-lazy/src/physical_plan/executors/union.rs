use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;
use polars_core::utils::concat_df;

pub(crate) struct UnionExec {
    pub(crate) inputs: Vec<Box<dyn Executor>>,
}

impl Executor for UnionExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let dfs = self
            .inputs
            .iter_mut()
            .map(|input| input.execute(state))
            .collect::<Result<Vec<_>>>()?;
        concat_df(&dfs)
    }
}
