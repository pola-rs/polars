use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;
use polars_core::utils::concat_df;
use polars_core::POOL;
use rayon::prelude::*;

pub(crate) struct UnionExec {
    pub(crate) inputs: Vec<Box<dyn Executor>>,
    pub(crate) options: UnionOptions,
}

impl Executor for UnionExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let inputs = std::mem::take(&mut self.inputs);

        let dfs = if self.options.slice && self.options.slice_offset >= 0 {
            let mut offset = self.options.slice_offset as usize;
            let mut len = self.options.slice_len as usize;
            let dfs = inputs
                .into_iter()
                .map(|mut input| {
                    let df = input.execute(state)?;

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
                .collect::<Result<Vec<_>>>()?;

            dfs.into_iter().flatten().collect()
        } else {
            POOL.install(|| {
                inputs
                    .into_par_iter()
                    .map(|mut input| input.execute(state))
                    .collect::<Result<Vec<_>>>()
            })?
        };

        concat_df(&dfs)
    }
}
