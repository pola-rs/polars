use std::any::Any;

use polars_core::prelude::*;

pub trait SExecutionContext: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn should_stop(&self) -> PolarsResult<()>;
}

pub struct PExecutionContext {
    // injected upstream in polars-lazy
    pub(crate) execution_state: Box<dyn SExecutionContext>,
    pub(crate) verbose: bool,
}

impl PExecutionContext {
    pub(crate) fn new(state: Box<dyn SExecutionContext>, verbose: bool) -> Self {
        PExecutionContext {
            execution_state: state,
            verbose,
        }
    }
}
