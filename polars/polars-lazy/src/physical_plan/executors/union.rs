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
        let mut inputs = std::mem::take(&mut self.inputs);

        let sliced_path = self.options.slice && self.options.slice_offset >= 0;

        if !self.options.parallel || sliced_path {
            if state.verbose() {
                if !self.options.parallel {
                    println!("UNION: `parallel=false` union is run sequentially")
                } else {
                    println!("UNION: `slice is set` union is run sequentially")
                }
            }

            let mut offset = self.options.slice_offset as usize;
            let mut len = self.options.slice_len as usize;
            let dfs = inputs
                .into_iter()
                .enumerate()
                .map(|(idx, mut input)| {
                    let mut state = state.split();
                    state.branch_idx += idx;
                    let df = input.execute(&mut state)?;

                    if !sliced_path {
                        return Ok(Some(df));
                    }

                    Ok(if offset > df.height() {
                        offset -= df.height();
                        None
                    } else if offset + len > df.height() {
                        len -= df.height() - offset;
                        if offset == 0 {
                            Some(df)
                        } else {
                            let out = Some(df.slice(offset as i64, usize::MAX));
                            offset = 0;
                            out
                        }
                    } else {
                        let out = Some(df.slice(offset as i64, len));
                        len = 0;
                        offset = 0;
                        out
                    })
                })
                .collect::<PolarsResult<Vec<_>>>()?;

            concat_df(dfs.iter().flatten())
        } else {
            if state.verbose() {
                println!("UNION: union is run in parallel")
            }

            // we don't use par_iter directly because the LP may also start threads for every LP (for instance scan_csv)
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

            concat_df(out?.iter().flat_map(|dfs| dfs.iter()))
        }
    }
}
