use smartstring::alias::String as SmartString;

use super::*;

pub struct ProjectionSimple {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) columns: SchemaRef,
    pub(crate) duplicate_check: bool,
}

impl ProjectionSimple {
    fn execute_impl(&mut self, df: DataFrame, columns: &[SmartString]) -> PolarsResult<DataFrame> {
        if self.duplicate_check {
            df._select_impl(columns.as_ref())
        } else {
            df._select_impl_unchecked(columns.as_ref())
        }
    }
}

impl Executor for ProjectionSimple {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        let columns = self.columns.iter_names().cloned().collect::<Vec<_>>();

        let profile_name = if state.has_node_timer() {
            let name = comma_delimited("simple-projection".to_string(), &columns);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };
        let df = self.input.execute(state)?;

        if state.has_node_timer() {
            state.record(|| self.execute_impl(df, &columns), profile_name)
        } else {
            self.execute_impl(df, &columns)
        }
    }
}
