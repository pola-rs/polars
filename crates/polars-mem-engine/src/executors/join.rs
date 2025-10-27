use polars_ops::frame::DataFrameJoinOps;

use super::*;

pub struct JoinExec {
    input_left: Option<Box<dyn Executor>>,
    input_right: Option<Box<dyn Executor>>,
    left_on: Vec<Arc<dyn PhysicalExpr>>,
    right_on: Vec<Arc<dyn PhysicalExpr>>,
    parallel: bool,
    args: JoinArgs,
    options: Option<JoinTypeOptions>,
}

impl JoinExec {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input_left: Box<dyn Executor>,
        input_right: Box<dyn Executor>,
        left_on: Vec<Arc<dyn PhysicalExpr>>,
        right_on: Vec<Arc<dyn PhysicalExpr>>,
        parallel: bool,
        args: JoinArgs,
        options: Option<JoinTypeOptions>,
    ) -> Self {
        JoinExec {
            input_left: Some(input_left),
            input_right: Some(input_right),
            left_on,
            right_on,
            parallel,
            args,
            options,
        }
    }
}

impl Executor for JoinExec {
    fn execute<'a>(&'a mut self, state: &'a mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run JoinExec")
            }
        }
        if state.verbose() {
            eprintln!("join parallel: {}", self.parallel);
        };
        let mut input_left = self.input_left.take().unwrap();
        let mut input_right = self.input_right.take().unwrap();

        let (df_left, df_right) = if self.parallel {
            let mut state_right = state.split();
            let mut state_left = state.split();
            state_right.branch_idx += 1;

            POOL.join(
                move || input_left.execute(&mut state_left),
                move || input_right.execute(&mut state_right),
            )
        } else {
            (input_left.execute(state), input_right.execute(state))
        };

        let df_left = df_left?;
        let df_right = df_right?;

        let profile_name = if state.has_node_timer() {
            let by = self
                .left_on
                .iter()
                .map(|s| Ok(s.to_field(df_left.schema())?.name))
                .collect::<PolarsResult<Vec<_>>>()?;
            let name = comma_delimited("join".to_string(), &by);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(
            || {
                let left_on_series = self
                    .left_on
                    .iter()
                    .map(|e| e.evaluate(&df_left, state))
                    .collect::<PolarsResult<Vec<_>>>()?;

                let right_on_series = self
                    .right_on
                    .iter()
                    .map(|e| e.evaluate(&df_right, state))
                    .collect::<PolarsResult<Vec<_>>>()?;

                let df = df_left._join_impl(
                    &df_right,
                    left_on_series
                        .into_iter()
                        .map(|c| c.take_materialized_series())
                        .collect(),
                    right_on_series
                        .into_iter()
                        .map(|c| c.take_materialized_series())
                        .collect(),
                    self.args.clone(),
                    self.options.clone(),
                    true,
                    state.verbose(),
                );

                if state.verbose() {
                    eprintln!("{:?} join dataframes finished", self.args.how);
                };
                df
            },
            profile_name,
        )
    }
}
