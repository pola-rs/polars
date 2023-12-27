use super::*;

pub struct StackExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) has_windows: bool,
    pub(crate) cse_exprs: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) exprs: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) input_schema: SchemaRef,
    pub(crate) options: ProjectionOptions,
}

impl StackExec {
    fn execute_impl(
        &mut self,
        state: &ExecutionState,
        mut df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        let res = evaluate_physical_expressions(
            &mut df,
            &self.cse_exprs,
            &self.exprs,
            state,
            self.has_windows,
            self.options.run_parallel,
        )?;
        state.clear_window_expr_cache();

        let schema = &*self.input_schema;
        df._add_columns(res, schema)?;

        Ok(df)
    }
}

impl Executor for StackExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                if self.cse_exprs.is_empty() {
                    println!("run StackExec");
                } else {
                    println!("run StackExec with {} CSE", self.cse_exprs.len());
                };
            }
        }
        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            let by = self
                .exprs
                .iter()
                .map(|s| {
                    profile_name(
                        s.as_ref(),
                        self.input_schema.as_ref(),
                        !self.cse_exprs.is_empty(),
                    )
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            let name = comma_delimited("with_column".to_string(), &by);
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
