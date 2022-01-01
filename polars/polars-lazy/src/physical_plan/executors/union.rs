use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;
use polars_core::utils::concat_df;
use polars_core::POOL;
use rayon::prelude::*;

pub(crate) struct UnionExec {
    pub(crate) inputs: Vec<Box<dyn Executor>>,
}

impl Executor for UnionExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let inputs = std::mem::take(&mut self.inputs);

        let dfs = POOL.install(|| {
            inputs
                .into_par_iter()
                .map(|mut input| input.execute(state))
                .collect::<Result<Vec<_>>>()
        })?;

        concat_df(&dfs)
    }
}
