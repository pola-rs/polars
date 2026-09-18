use polars_defs::join::{JoinArgs, JoinTypeOptions};
use polars_ops::frame::DataFrameJoinOps;
use recursive::recursive;

use super::*;

pub struct JoinExec {
    input_left: Option<Box<dyn Executor>>,
    input_right: Option<Box<dyn Executor>>,
    left_on: Vec<Arc<dyn PhysicalExpr>>,
    right_on: Vec<Arc<dyn PhysicalExpr>>,
    parallel: bool,
    args: JoinArgs,
    options: Option<JoinTypeOptions>,
    pass_through_above: Option<usize>,
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
        pass_through_above: Option<usize>,
    ) -> Self {
        JoinExec {
            input_left: Some(input_left),
            input_right: Some(input_right),
            left_on,
            right_on,
            parallel,
            args,
            options,
            pass_through_above,
        }
    }
}

impl Executor for JoinExec {
    #[recursive]
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

            RAYON.join(
                move || input_left.execute(&mut state_left),
                move || input_right.execute(&mut state_right),
            )
        } else {
            (input_left.execute(state), input_right.execute(state))
        };

        let df_left = df_left?;
        let df_right = df_right?;
        if self
            .pass_through_above
            .is_some_and(|limit| df_right.height() > limit)
        {
            if state.verbose() {
                eprintln!("semi join passed through: {} build rows", df_right.height());
            }
            return Ok(df_left);
        }

        let evaluate_keys = |df: &DataFrame, keys: &[Arc<dyn PhysicalExpr>]| {
            keys.iter()
                .map(|key| key.evaluate(df, state)?.broadcast_owned_to(df.height()))
                .collect::<PolarsResult<Vec<_>>>()
        };
        let left_on_series = evaluate_keys(&df_left, &self.left_on)?;
        let right_on_series = evaluate_keys(&df_right, &self.right_on)?;

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
    }
}
