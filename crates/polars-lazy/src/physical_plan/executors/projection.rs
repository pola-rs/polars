use super::*;

/// Take an input Executor (creates the input DataFrame)
/// and a multiple PhysicalExpressions (create the output Series)
pub struct ProjectionExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) cse_exprs: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) expr: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) has_windows: bool,
    pub(crate) input_schema: SchemaRef,
    #[cfg(test)]
    pub(crate) schema: SchemaRef,
    pub(crate) options: ProjectionOptions,
}

impl ProjectionExec {
    fn execute_impl(
        &mut self,
        state: &ExecutionState,
        mut df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        #[allow(clippy::let_and_return)]
        let selected_cols = evaluate_physical_expressions(
            &mut df,
            &self.cse_exprs,
            &self.expr,
            state,
            self.has_windows,
            self.options.run_parallel,
        )?;
        #[allow(unused_mut)]
        let mut df = check_expand_literals(selected_cols, df.height() == 0)?;

        // this only runs during testing and check if the runtime type matches the predicted schema
        #[cfg(test)]
        #[allow(unused_must_use)]
        {
            // TODO: also check the types.
            for (l, r) in df.iter().zip(self.schema.iter_names()) {
                assert_eq!(l.name(), r);
            }
        }

        Ok(df)
    }
}

impl Executor for ProjectionExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                if self.cse_exprs.is_empty() {
                    println!("run ProjectionExec");
                } else {
                    println!("run ProjectionExec with {} CSE", self.cse_exprs.len())
                };
            }
        }
        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            let by = self
                .expr
                .iter()
                .map(|s| {
                    profile_name(
                        s.as_ref(),
                        self.input_schema.as_ref(),
                        !self.cse_exprs.is_empty(),
                    )
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            let name = comma_delimited("projection".to_string(), &by);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        if state.has_node_timer() {
            let new_state = state.clone();
            new_state.record(|| self.execute_impl(state, df), profile_name)
        } else {
            self.execute_impl(state, df)
        }
    }
}
