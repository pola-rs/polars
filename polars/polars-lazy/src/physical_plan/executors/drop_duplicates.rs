use polars_core::prelude::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub(crate) struct DropDuplicatesExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) options: DistinctOptions,
}

impl Executor for DropDuplicatesExec {
    fn execute(&mut self, state: &mut ExecutionState) -> Result<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run DropDuplicatesExec")
            }
        }
        let df = self.input.execute(state)?;
        let subset = self.options.subset.as_ref().map(|v| &***v);
        let keep = self.options.keep_strategy;

        match self.options.maintain_order {
            true => df.unique_stable(subset, keep),
            false => df.unique(subset, keep),
        }
    }
}
