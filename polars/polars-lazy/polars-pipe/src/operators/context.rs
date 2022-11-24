use std::any::Any;

use polars_core::schema::SchemaRef;

pub trait SExecutionContext: Send + Sync {
    fn input_schema_is_set(&self) -> bool;

    fn set_input_schema(&self, schema: SchemaRef);

    fn clear_input_schema(&self);

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
