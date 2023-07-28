use super::*;

pub struct FilterExec {
    pub(crate) predicate: Arc<dyn PhysicalExpr>,
    pub(crate) input: Box<dyn Executor>,
    // if the predicate contains a window function
    has_window: bool,
}

impl FilterExec {
    pub fn new(
        predicate: Arc<dyn PhysicalExpr>,
        input: Box<dyn Executor>,
        has_window: bool,
    ) -> Self {
        Self {
            predicate,
            input,
            has_window,
        }
    }
}

impl Executor for FilterExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run FilterExec")
            }
        }
        let df = self.input.execute(state)?;

        if self.has_window {
            state.insert_has_window_function_flag()
        }
        let s = self.predicate.evaluate(&df, state)?;
        let mask = s.bool().map_err(|_| {
            polars_err!(
                ComputeError: "filter predicate must be of type `Boolean`, got `{}`", s.dtype()
            )
        })?;

        let profile_name = if state.has_node_timer() {
            Cow::Owned(format!(".filter({})", &self.predicate.as_ref()))
        } else {
            Cow::Borrowed("")
        };

        state.record(
            || {
                let df = df.filter(mask)?;
                if state.verbose() {
                    eprintln!("dataframe filtered");
                }
                Ok(df)
            },
            profile_name,
        )
    }
}
