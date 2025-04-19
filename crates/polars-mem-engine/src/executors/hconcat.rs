use polars_core::functions::concat_df_horizontal;

use super::*;

pub(crate) struct HConcatExec {
    pub(crate) inputs: Vec<Box<dyn Executor>>,
    pub(crate) options: HConcatOptions,
}

impl Executor for HConcatExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run HConcatExec")
            }
        }
        let mut inputs = std::mem::take(&mut self.inputs);

        let dfs = if !self.options.parallel {
            if state.verbose() {
                eprintln!("HCONCAT: `parallel=false` hconcat is run sequentially")
            }
            let mut dfs = Vec::with_capacity(inputs.len());
            for (idx, mut input) in inputs.into_iter().enumerate() {
                let mut state = state.split();
                state.branch_idx += idx;

                let df = input.execute(&mut state)?;

                dfs.push(df);
            }
            dfs
        } else {
            if state.verbose() {
                eprintln!("HCONCAT: hconcat is run in parallel")
            }
            // We don't use par_iter directly because the LP may also start threads for every LP (for instance scan_csv)
            // this might then lead to a rayon SO. So we take a multitude of the threads to keep work stealing
            // within bounds
            let out = POOL.install(|| {
                inputs
                    .chunks_mut(POOL.current_num_threads() * 3)
                    .map(|chunk| {
                        chunk
                            .into_par_iter()
                            .enumerate()
                            .map(|(idx, input)| {
                                let mut input = std::mem::take(input);
                                let mut state = state.split();
                                state.branch_idx += idx;
                                input.execute(&mut state)
                            })
                            .collect::<PolarsResult<Vec<_>>>()
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            });
            out?.into_iter().flatten().collect()
        };

        // Invariant of IR. Schema is already checked to contain no duplicates.
        concat_df_horizontal(&dfs, false)
    }
}
