use std::sync::Arc;

use polars_core::prelude::IdxSize;
use polars_utils::index::check_bounds;

use super::*;

pub struct GatherExec {
    pub input: Box<dyn Executor>,
    pub indices: Arc<[IdxSize]>,
}

impl Executor for GatherExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run GatherExec")
            }
        }
        let df = self.input.execute(state)?;

        state.record(
            || {
                check_bounds(&self.indices, df.height() as IdxSize)?;
                // SAFETY: bounds checked above
                Ok(unsafe { df.take_slice_unchecked(&self.indices) })
            },
            "gather".into(),
        )
    }
}
