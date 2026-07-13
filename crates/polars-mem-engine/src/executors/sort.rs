use polars_utils::format_pl_smallstr;

use super::*;

pub(crate) struct SortExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) by_column: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) slice: Option<(i64, usize)>,
    pub(crate) sort_options: SortMultipleOptions,
}

impl SortExec {
    fn execute_impl(
        &mut self,
        state: &ExecutionState,
        mut df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        df.rechunk_mut_par();

        let height = df.height();

        let by_columns = self
            .by_column
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let mut s = e.evaluate(&df, state)?.into_column();
                // Polars core will try to set the sorted columns as sorted.
                // This should only be done with simple col("foo") expressions,
                // therefore we rename more complex expressions so that
                // polars core does not match these.
                if !matches!(e.as_expression(), Some(&Expr::Column(_))) {
                    s.rename(format_pl_smallstr!("_POLARS_SORT_BY_{i}"));
                }
                polars_ensure!(
                    s.len() == height,
                    ShapeMismatch: "sort expressions must have same \
                    length as DataFrame, got DataFrame height: {} and Series length: {}",
                    height, s.len()
                );
                Ok(s)
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        df.sort_impl(by_columns, self.sort_options.clone(), self.slice)
    }
}

impl Executor for SortExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run SortExec")
            }
        }
        let df = self.input.execute(state)?;
        self.execute_impl(state, df)
    }
}
