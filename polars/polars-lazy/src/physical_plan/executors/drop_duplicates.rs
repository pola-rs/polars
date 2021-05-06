use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

pub(crate) struct DropDuplicatesExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) maintain_order: bool,
    pub(crate) subset: Option<Vec<String>>,
}

impl Executor for DropDuplicatesExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let df = self.input.execute(state)?;
        df.drop_duplicates(
            self.maintain_order,
            self.subset.as_ref().map(|v| v.as_ref()),
        )
    }
}
