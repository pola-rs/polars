use super::*;

pub(crate) struct UniqueExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) expr: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) options: DistinctOptions,
    pub(crate) input_schema: SchemaRef,
}

impl Executor for UniqueExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run UniqueExec")
            }
        }
        let df = self.input.execute(state)?;
        let subset = self
            .expr
            .iter()
            .map(|s| Ok(s.to_field(&self.input_schema)?.name.to_string()))
            .collect::<PolarsResult<Vec<_>>>()?;
        
        let subset = if subset.is_empty() {
            None
        } else {
            Some(subset.as_ref())
        };

        let keep = self.options.keep_strategy;

        state.record(
            || match self.options.maintain_order {
                true => df.unique_stable(subset, keep, self.options.slice),
                false => df.unique(subset, keep, self.options.slice),
            },
            Cow::Borrowed("unique()"),
        )
    }
}
