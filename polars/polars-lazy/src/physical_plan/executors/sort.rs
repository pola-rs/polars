use super::*;

pub(crate) struct SortExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) by_column: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) args: SortArguments,
}

impl SortExec {
    fn execute_impl(
        &mut self,
        state: &mut ExecutionState,
        mut df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        df.as_single_chunk_par();

        let by_columns = self
            .by_column
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let mut s = e.evaluate(&df, state)?;
                // polars core will try to set the sorted columns as sorted
                // this should only be done with simple col("foo") expressions
                // therefore we rename more complex expressions so that
                // polars core does not match these
                if !matches!(e.as_expression(), Some(&Expr::Column(_))) {
                    s.rename(&format!("_POLARS_SORT_BY_{i}"));
                }
                Ok(s)
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        df.sort_impl(
            by_columns,
            std::mem::take(&mut self.args.reverse),
            self.args.nulls_last,
            self.args.slice,
            true,
        )
    }
}

impl Executor for SortExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                println!("run SortExec")
            }
        }
        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            let by = self
                .by_column
                .iter()
                .map(|s| Ok(s.to_field(&df.schema())?.name))
                .collect::<PolarsResult<Vec<_>>>()?;
            let name = column_delimited("sort".to_string(), &by);
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
