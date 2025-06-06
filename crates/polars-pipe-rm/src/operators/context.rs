use polars_expr::state::ExecutionState;

pub struct PExecutionContext {
    // injected upstream in polars-lazy
    pub(crate) execution_state: ExecutionState,
    pub(crate) verbose: bool,
}

impl PExecutionContext {
    pub(crate) fn new(state: ExecutionState, verbose: bool) -> Self {
        PExecutionContext {
            execution_state: state,
            verbose,
        }
    }
}
