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
    pub(crate) allow_vertical_parallelism: bool,
    pub(crate) allow_single_chunk_vertical_parallelism: bool,
}

impl ProjectionExec {
    fn execute_impl(
        &mut self,
        state: &ExecutionState,
        mut df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        let n_partitions = RAYON.current_num_threads();
        let n_chunks = df.first_col_n_chunks();
        let split_single_chunk = self.allow_single_chunk_vertical_parallelism
            && n_chunks == 1
            && should_split_vertically(df.height(), self.expr.len(), n_partitions);

        // Vertical and horizontal parallelism.
        let df = if self.allow_vertical_parallelism
            && (n_chunks > 1 || split_single_chunk)
            && df.height() > n_partitions * 2
            && self.options.run_parallel
        {
            let chunks = if n_chunks > 1 {
                df.split_chunks().collect::<Vec<_>>()
            } else {
                df.split_chunks_by_n(n_partitions, true)
            };
            let iter = chunks.into_par_iter().map(|mut df| {
                let selected_cols = evaluate_physical_expressions(
                    &mut df,
                    &self.expr,
                    state,
                    self.has_windows,
                    self.options.run_parallel,
                )?;
                check_expand_literals(
                    &df,
                    &self.expr,
                    selected_cols,
                    df.shape_has_zero(),
                    self.options,
                )
            });

            let df = RAYON.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;
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
            check_expand_literals(
                &df,
                &self.expr,
                selected_cols,
                df.shape_has_zero(),
                self.options,
            )?
        };

        // this only runs during testing and check if the runtime type matches the predicted schema
        #[cfg(test)]
        #[allow(unused_must_use)]
        {
            // TODO: also check the types.
            for (l, r) in df.columns().iter().zip(self.schema.iter_names()) {
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
