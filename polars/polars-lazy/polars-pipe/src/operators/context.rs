use std::any::Any;

pub struct PExecutionContext {
    // injected upstream in polars-lazy
    pub(crate) execution_state: Box<dyn Any + Send + Sync>,
}

impl PExecutionContext {
    pub(crate) fn new(state: Box<dyn Any + Send + Sync>) -> Self {
        PExecutionContext {
            execution_state: state,
        }
    }
}
