use polars_core::functions::concat_df_horizontal;

use super::*;

pub(crate) struct HConcatExec {
    pub(crate) inputs: Vec<Box<dyn Executor>>,
}

impl Executor for HConcatExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run HConcatExec")
            }
        }
        let inputs = std::mem::take(&mut self.inputs);

        // TODO: Parallel impl depending on options
        let mut dfs = Vec::with_capacity(inputs.len());

        for (idx, mut input) in inputs.into_iter().enumerate() {
            let mut state = state.split();
            state.branch_idx += idx;

            let df = input.execute(&mut state)?;

            dfs.push(df);
        }

        // TODO: does it make sense to allow rechunk?
        concat_df_horizontal(&dfs)
    }
}
