use polars_core::utils::concat_df;
use recursive::recursive;

use super::*;

pub(crate) struct UnionExec {
    pub(crate) inputs: Vec<Box<dyn Executor>>,
    pub(crate) options: UnionOptions,
}

impl Executor for UnionExec {
    #[recursive]
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run UnionExec")
            }
        }
        let mut inputs = std::mem::take(&mut self.inputs);
        let n_inputs = inputs.len();

        let positive_slice = self.options.slice.filter(|s| s.is_positive());

        if !self.options.parallel || positive_slice.is_some() {
            if state.verbose() {
                if !self.options.parallel {
                    eprintln!("UNION: `parallel=false` union is run sequentially")
                } else {
                    eprintln!("UNION: `slice is set` union is run sequentially")
                }
            }

            // The positive slice relative to the start of the current input.
            let mut remaining_slice = positive_slice;
            let mut dfs = Vec::with_capacity(inputs.len());

            for (idx, mut input) in inputs.into_iter().enumerate() {
                let mut state = state.split();
                state.branch_idx += idx;

                let df = input.execute(&mut state)?;

                let Some(slice) = remaining_slice else {
                    dfs.push(df);
                    continue;
                };

                // TODO!: don't read the file yet if it is before the slice!
                let height = df.height();
                let (range, next_slice) = slice.split_front(height);
                let is_exhausted = next_slice.is_some_and(|s| s.is_empty());
                let is_last = idx + 1 == n_inputs || is_exhausted;

                // Keep the union schema when the slice skips every input.
                if !range.is_empty() || (is_last && dfs.is_empty()) {
                    dfs.push(df.slice(range));
                }

                if is_exhausted {
                    break;
                }
                remaining_slice = next_slice;
            }

            let df = concat_df(&dfs)?;
            Ok(match self.options.slice.filter(|s| s.is_negative()) {
                Some(slice) => df.slice(slice),
                None => df,
            })
        } else {
            if state.verbose() {
                eprintln!("UNION: union is run in parallel")
            }

            // we don't use par_iter directly because the LP may also start threads for every LP (for instance scan_csv)
            // this might then lead to a rayon SO. So we take a multiple of the threads to keep work stealing
            // within bounds
            let out = RAYON.install(|| {
                inputs
                    .chunks_mut(RAYON.current_num_threads() * 3)
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

            concat_df(out?.iter().flat_map(|dfs| dfs.iter())).map(|df| {
                if let Some(slice) = self.options.slice {
                    df.slice(slice)
                } else {
                    df
                }
            })
        }
        .map(|mut df| {
            if self.options.rechunk {
                df.rechunk_mut_par();
            }
            df
        })
    }
}
