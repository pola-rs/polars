use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use super::*;

/// Take an input Executor (creates the input DataFrame)
/// and a multiple PhysicalExpressions (create the output Series)
pub struct ProjectionExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) expr: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) has_windows: bool,
    pub(crate) input_schema: SchemaRef,
    #[cfg(test)]
    pub(crate) schema: SchemaRef,
    pub(crate) options: ProjectionOptions,
    // Can run all operations elementwise
    pub(crate) streamable: bool,
}

impl ProjectionExec {
    fn execute_impl(
        &mut self,
        state: &ExecutionState,
        mut df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        // Vertical and horizontal parallelism.
        let df = if self.streamable
            && df.n_chunks() > 1
            && df.height() > POOL.current_num_threads() * 2
            && self.options.run_parallel
        {
            let chunks = df.split_chunks().collect::<Vec<_>>();
            let iter = chunks.into_par_iter().map(|mut df| {
                let selected_cols = evaluate_physical_expressions(
                    &mut df,
                    &self.expr,
                    state,
                    self.has_windows,
                    self.options.run_parallel,
                )?;
                check_expand_literals(selected_cols, df.is_empty(), self.options)
            });

            let df = POOL.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;
            accumulate_dataframes_vertical_unchecked(df)
        }
        // Only horizontal parallelism.
        else {
            #[allow(clippy::let_and_return)]
            let selected_cols = evaluate_physical_expressions(
                &mut df,
                &self.expr,
                state,
                self.has_windows,
                self.options.run_parallel,
            )?;
            check_expand_literals(selected_cols, df.is_empty(), self.options)?
        };

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
                eprintln!("run ProjectionExec");
            }
        }
        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            let by = self
                .expr
                .iter()
                .map(|s| profile_name(s.as_ref(), self.input_schema.as_ref()))
                .collect::<PolarsResult<Vec<_>>>()?;
            let name = comma_delimited("select".to_string(), &by);
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
