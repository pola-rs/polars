use polars_core::utils::concat_df;
use polars_plan::global::_is_fetch_query;

use super::*;

pub(crate) struct UnionExec {
    pub(crate) inputs: Vec<Box<dyn Executor>>,
    pub(crate) options: UnionOptions,
}

impl Executor for UnionExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run UnionExec")
            }
        }
        // keep scans thread local if 'fetch' is used.
        if _is_fetch_query() {
            self.options.parallel = false;
        }
        let inputs = std::mem::take(&mut self.inputs);

        let sliced_path = if let Some((offset, _)) = self.options.slice {
            offset >= 0
        } else {
            false
        };

        if !self.options.parallel || sliced_path {
            if state.verbose() {
                if !self.options.parallel {
                    println!("UNION: `parallel=false` union is run sequentially")
                } else {
                    println!("UNION: `slice is set` union is run sequentially")
                }
            }

            let (slice_offset, mut slice_len) = self.options.slice.unwrap_or((0, usize::MAX));
            let mut slice_offset = slice_offset as usize;
            let mut dfs = Vec::with_capacity(inputs.len());

            for (idx, mut input) in inputs.into_iter().enumerate() {
                let mut state = state.split();
                state.branch_idx += idx;

                let df = input.execute(&mut state)?;

                if !sliced_path {
                    dfs.push(df);
                    continue;
                }

                let height = df.height();
                // this part can be skipped as we haven't reached the offset yet
                // TODO!: don't read the file yet!
                if slice_offset > height {
                    slice_offset -= height;
                }
                // applying the slice
                // continue iteration
                else if slice_offset + slice_len > height {
                    slice_len -= height - slice_offset;
                    if slice_offset == 0 {
                        dfs.push(df);
                    } else {
                        dfs.push(df.slice(slice_offset as i64, usize::MAX));
                        slice_offset = 0;
                    }
                }
                // we finished the slice
                else {
                    dfs.push(df.slice(slice_offset as i64, slice_len));
                    break;
                }
            }

            concat_df(&dfs)
        } else {
            if state.verbose() {
                println!("UNION: union is run in parallel")
            }

            let dfs = POOL.install(|| {
                inputs
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, mut input)| {
                        let mut state = state.split();
                        state.branch_idx += idx;
                        input.execute(&mut state)
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            });

            concat_df(dfs.iter().flatten())
        }
    }
}
