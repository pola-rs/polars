use super::*;

pub struct ProjectionSimple {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) columns: SchemaRef,
}

impl Executor for ProjectionSimple {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let df = self.input.execute(state)?;
        let height = df.height();
        let mut df = unsafe { df.select_unchecked(self.columns.iter_names_cloned())? };
        unsafe { df.set_height(height) };
        Ok(df)
    }
}
