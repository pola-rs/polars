use polars_core::prelude::IdxSize;
use polars_core::POOL;
use polars_utils::index::check_bounds;
use recursive::recursive;

use super::*;

pub struct GatherExec {
    pub input: Box<dyn Executor>,
    pub indices: Box<dyn Executor>,
}

impl Executor for GatherExec {
    #[recursive]
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run GatherExec")
            }
        }

        // Execute both inputs in parallel
        let (df, indices_df) = {
            let mut state2 = state.split();
            state2.branch_idx += 1;
            let (df, indices_df) = POOL.join(
                || self.input.execute(state),
                || self.indices.execute(&mut state2),
            );
            (df?, indices_df?)
        };

        state.record(
            || {
                // Get the first (and only) column from indices_df as the indices
                polars_ensure!(
                    indices_df.width() == 1,
                    ComputeError: "gather indices DataFrame must have exactly one column, got {}",
                    indices_df.width()
                );
                let indices_col = indices_df.columns()[0].clone();
                let indices = indices_col.idx().map_err(|_| {
                    polars_err!(
                        ComputeError: "gather indices must be of type UInt32 (IdxSize), got {}",
                        indices_col.dtype()
                    )
                })?;

                // Rechunk to get contiguous memory, then get slice
                let indices = indices.rechunk();
                polars_ensure!(
                    !indices.has_nulls(),
                    ComputeError: "gather indices contain null values"
                );
                let indices_slice = indices.cont_slice().unwrap();

                check_bounds(indices_slice, df.height() as IdxSize)?;
                // SAFETY: bounds checked above
                Ok(unsafe { df.take_slice_unchecked(indices_slice) })
            },
            "gather".into(),
        )
    }
}
