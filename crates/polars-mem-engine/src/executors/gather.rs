use polars_ops::frame::gather::GatherDf;
use recursive::recursive;

use super::*;

pub struct GatherExec {
    target: Box<dyn Executor>,
    idxs: Box<dyn Executor>,
    null_on_oob: bool,
}

impl GatherExec {
    pub fn new(target: Box<dyn Executor>, idxs: Box<dyn Executor>, null_on_oob: bool) -> Self {
        GatherExec {
            target,
            idxs,
            null_on_oob,
        }
    }
}

impl Executor for GatherExec {
    #[recursive]
    fn execute<'a>(&'a mut self, state: &'a mut ExecutionState) -> PolarsResult<DataFrame> {
        let target = self.target.execute(state)?;
        let idxs = self.idxs.execute(state)?;
        assert!(idxs.width() == 1);
        target.gather_with_column(&idxs.columns()[0], self.null_on_oob)
    }
}
