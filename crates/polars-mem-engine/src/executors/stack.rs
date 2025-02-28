use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_plan::constants::CSE_REPLACED;

use super::*;

pub struct StackExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) has_windows: bool,
    pub(crate) exprs: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) input_schema: SchemaRef,
    pub(crate) output_schema: SchemaRef,
    pub(crate) options: ProjectionOptions,
    // Can run all operations elementwise
    pub(crate) allow_vertical_parallelism: bool,
}

impl StackExec {
    fn execute_impl(
        &mut self,
        state: &ExecutionState,
        mut df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        let schema = &*self.output_schema;

        // Vertical and horizontal parallelism.
        let df = if self.allow_vertical_parallelism
            && df.first_col_n_chunks() > 1
            && df.height() > 0
            && self.options.run_parallel
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
                df._add_columns(res.into_iter().collect(), schema)?;
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
                unsafe { df.column_extend_unchecked(res) };
            } else {
                let (df_height, df_width) = df.shape();

                // When we have CSE we cannot verify scalars yet.
                let verify_scalar = if !df.get_columns().is_empty() {
                    !df.get_columns()[df.width() - 1]
                        .name()
                        .starts_with(CSE_REPLACED)
                } else {
                    true
                };
                for (i, c) in res.iter().enumerate() {
                    let len = c.len();
                    if verify_scalar && len != df_height && len == 1 && df_width > 0 {
                        #[allow(clippy::collapsible_if)]
                        if !self.exprs[i].is_scalar()
                            && std::env::var("POLARS_ALLOW_NON_SCALAR_EXP").as_deref() != Ok("1")
                        {
                            let identifier = match self.exprs[i].as_expression() {
                                Some(e) => format!("expression: {}", e),
                                None => "this Series".to_string(),
                            };
                            polars_bail!(InvalidOperation: "Series {}, length {} doesn't match the DataFrame height of {}\n\n\
                                If you want {} to be broadcasted, ensure it is a scalar (for instance by adding '.first()').",
                                c.name(), len, df_height, identifier
                            );
                        }
                    }
                }
                df._add_columns(res.into_iter().collect(), schema)?;
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
