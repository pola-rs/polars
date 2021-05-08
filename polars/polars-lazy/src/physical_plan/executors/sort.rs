use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;

pub(crate) struct SortExec {
    pub(crate) input: Box<dyn Executor>,
    pub(crate) by_column: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) reverse: Vec<bool>,
}

impl Executor for SortExec {
    fn execute(&mut self, state: &ExecutionState) -> Result<DataFrame> {
        let mut df = self.input.execute(state)?;

        let by_columns = self
            .by_column
            .iter()
            .map(|e| e.evaluate(&df, state))
            .collect::<Result<Vec<_>>>()?;
        let mut column_names = Vec::with_capacity(by_columns.len());
        // replace the columns in the DataFrame with the expressions
        // for col("foo") this is redundant
        // for col("foo").reverse() this is not
        for column in by_columns {
            let name = column.name();
            column_names.push(name.to_string());
            // if error, expression create a new named column and we must add it to the DataFrame
            // if ok, we have replaced the column with the expression eval
            if df.apply(name, |_| column.clone()).is_err() {
                df.hstack(&[column])?;
            }
        }

        df.sort(&column_names, std::mem::take(&mut self.reverse))
    }
}
