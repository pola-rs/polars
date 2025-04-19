use super::*;

pub(crate) struct UniqueExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) options: DistinctOptionsIR,
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
        let subset = self
            .options
            .subset
            .as_ref()
            .map(|v| v.iter().cloned().collect::<Vec<_>>());
        let keep = self.options.keep_strategy;

        state.record(
            || {
                if df.is_empty() {
                    return Ok(df);
                }

                df.unique_impl(
                    self.options.maintain_order,
                    subset,
                    keep,
                    self.options.slice,
                )
            },
            Cow::Borrowed("unique()"),
        )
    }
}
