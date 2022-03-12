use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

pub(crate) struct SortExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) by_column: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) args: SortArguments,
}

impl Executor for SortExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let df = self.input.execute(state)?;

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
                if !matches!(e.as_expression(), Expr::Column(_)) {
                    s.rename(&format!("_POLARS_SORT_BY_{}", i));
                }
                Ok(s)
            })
            .collect::<Result<Vec<_>>>()?;
        df.sort_impl(
            by_columns,
            std::mem::take(&mut self.args.reverse),
            self.args.nulls_last,
        )
    }
}
