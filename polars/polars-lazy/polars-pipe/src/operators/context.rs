use std::any::Any;

pub trait SExecutionContext: Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

pub struct PExecutionContext {
    // injected upstream in polars-lazy
    pub(crate) execution_state: Box<dyn SExecutionContext>,
}

impl PExecutionContext {
    pub(crate) fn new(state: Box<dyn SExecutionContext>) -> Self {
        PExecutionContext {
            execution_state: state,
        }
    }
}
