use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use super::*;

pub struct StackExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) has_windows: bool,
    pub(crate) exprs: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) input_schema: SchemaRef,
    pub(crate) options: ProjectionOptions,
    // Can run all operations elementwise
    pub(crate) streamable: bool,
}

impl StackExec {
    fn execute_impl(
        &mut self,
        state: &ExecutionState,
        mut df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        let schema = &*self.input_schema;

        // Vertical and horizontal parallelism.
        let df =
            if self.streamable && df.n_chunks() > 1 && df.height() > 0 && self.options.run_parallel
            {
                let chunks = df.split_chunks().collect::<Vec<_>>();
                let iter = chunks.into_par_iter().map(|mut df| {
                    let res = evaluate_physical_expressions(
                        &mut df,
                        &self.exprs,
                        state,
                        self.has_windows,
                        self.options.run_parallel,
                    )?;
                    // We don't have to do a broadcast check as cse is not allowed to hit this.
                    df._add_columns(res, schema)?;
                    Ok(df)
                });

                let df = POOL.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;
                accumulate_dataframes_vertical_unchecked(df)
            }
            // Only horizontal parallelism
            else {
                let res = evaluate_physical_expressions(
                    &mut df,
                    &self.exprs,
                    state,
                    self.has_windows,
                    self.options.run_parallel,
                )?;
                if !self.options.should_broadcast {
                    debug_assert!(
                        res.iter()
                            .all(|column| column.name().starts_with("__POLARS_CSER_0x")),
                        "non-broadcasting hstack should only be used for CSE columns"
                    );
                    // Safety: this case only appears as a result of
                    // CSE optimization, and the usage there produces
                    // new, unique column names. It is immediately
                    // followed by a projection which pulls out the
                    // possibly mismatching column lengths.
                    unsafe { df.get_columns_mut().extend(res) };
                } else {
                    df._add_columns(res, schema)?;
                }
                df
            };

        state.clear_window_expr_cache();

        Ok(df)
    }
}

impl Executor for StackExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run StackExec");
            }
        }
        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            let by = self
                .exprs
                .iter()
                .map(|s| profile_name(s.as_ref(), self.input_schema.as_ref()))
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
