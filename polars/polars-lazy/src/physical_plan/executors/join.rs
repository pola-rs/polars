use crate::logical_plan::FETCH_ROWS;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;
use polars_core::POOL;

pub struct JoinExec {
    input_left: Option<Box<dyn Executor>>,
    input_right: Option<Box<dyn Executor>>,
    how: JoinType,
    left_on: Vec<Arc<dyn PhysicalExpr>>,
    right_on: Vec<Arc<dyn PhysicalExpr>>,
    parallel: bool,
}

impl JoinExec {
    pub(crate) fn new(
        input_left: Box<dyn Executor>,
        input_right: Box<dyn Executor>,
        how: JoinType,
        left_on: Vec<Arc<dyn PhysicalExpr>>,
        right_on: Vec<Arc<dyn PhysicalExpr>>,
        parallel: bool,
    ) -> Self {
        JoinExec {
            input_left: Some(input_left),
            input_right: Some(input_right),
            how,
            left_on,
            right_on,
            parallel,
        }
    }
}

impl Executor for JoinExec {
    fn execute<'a>(&'a mut self, state: &'a ExecutionState) -> Result<DataFrame> {
        let mut input_left = self.input_left.take().unwrap();
        let mut input_right = self.input_right.take().unwrap();

        let (df_left, df_right) = if self.parallel {
            let state_left = state.clone();
            let state_right = state.clone();
            // propagate the fetch_rows static value to the spawning threads.
            let fetch_rows = FETCH_ROWS.with(|fetch_rows| fetch_rows.get());

            POOL.join(
                move || {
                    FETCH_ROWS.with(|fr| fr.set(fetch_rows));
                    input_left.execute(&state_left)
                },
                move || {
                    FETCH_ROWS.with(|fr| fr.set(fetch_rows));
                    input_right.execute(&state_right)
                },
            )
        } else {
            (input_left.execute(state), input_right.execute(state))
        };

        let df_left = df_left?;
        let df_right = df_right?;

        let left_names = self
            .left_on
            .iter()
            .map(|e| e.evaluate(&df_left, state).map(|s| s.name().to_string()))
            .collect::<Result<Vec<_>>>()?;

        let right_names = self
            .right_on
            .iter()
            .map(|e| e.evaluate(&df_right, state).map(|s| s.name().to_string()))
            .collect::<Result<Vec<_>>>()?;

        let df = df_left.join(&df_right, &left_names, &right_names, self.how);
        if state.verbose {
            eprintln!("{:?} join dataframes finished", self.how);
        };
        df
    }
}
