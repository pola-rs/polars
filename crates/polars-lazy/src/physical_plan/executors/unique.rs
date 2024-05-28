use super::*;

pub(crate) struct UniqueExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) options: DistinctOptions,
}

impl Executor for UniqueExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run UniqueExec")
            }
        }
        let df = self.input.execute(state)?;
        let subset = self.options.subset.as_ref().map(|v| &***v);
        let keep = self.options.keep_strategy;

        state.record(
            || {
                if df.is_empty() {
                    return Ok(df);
                }

                match self.options.maintain_order {
                    true => df.unique_stable(subset, keep, self.options.slice),
                    false => df.unique(subset, keep, self.options.slice),
                }
            },
            Cow::Borrowed("unique()"),
        )
    }
}
