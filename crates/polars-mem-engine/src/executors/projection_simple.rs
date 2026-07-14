use super::*;

pub struct ProjectionSimple {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) columns: SchemaRef,
}

impl ProjectionSimple {
    fn execute_impl(&mut self, df: DataFrame, columns: &[PlSmallStr]) -> PolarsResult<DataFrame> {
        // No duplicate check as that an invariant of this node.
        unsafe { df.select_unchecked(columns) }
    }
}

impl Executor for ProjectionSimple {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        let columns = self.columns.iter_names_cloned().collect::<Vec<_>>();
        let df = self.input.execute(state)?;
        self.execute_impl(df, columns.as_slice())
    }
}
