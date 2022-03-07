use crate::logical_plan::FETCH_ROWS;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;
use polars_core::POOL;
use std::borrow::Cow;

pub struct JoinExec {
    input_left: Option<Box<dyn Executor>>,
    input_right: Option<Box<dyn Executor>>,
    how: JoinType,
    left_on: Vec<Arc<dyn PhysicalExpr>>,
    right_on: Vec<Arc<dyn PhysicalExpr>>,
    parallel: bool,
    suffix: Cow<'static, str>,
}

impl JoinExec {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input_left: Box<dyn Executor>,
        input_right: Box<dyn Executor>,
        how: JoinType,
        left_on: Vec<Arc<dyn PhysicalExpr>>,
        right_on: Vec<Arc<dyn PhysicalExpr>>,
        parallel: bool,
        suffix: Cow<'static, str>,
    ) -> Self {
        JoinExec {
            input_left: Some(input_left),
            input_right: Some(input_right),
            how,
            left_on,
            right_on,
            parallel,
            suffix,
        }
    }
}

impl Executor for JoinExec {
    fn execute<'a>(&'a mut self, state: &'a ExecutionState) -> Result<DataFrame> {
        if state.verbose {
            eprintln!("join parallel: {}", self.parallel);
        };
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

        // prepare the tolerance
        // we must ensure that we use the right units
        #[cfg(feature = "asof_join")]
        {
            if let JoinType::AsOf(options) = &mut self.how {
                use polars_core::utils::arrow::temporal_conversions::MILLISECONDS_IN_DAY;
                if let Some(tol) = &options.tolerance_str {
                    let duration = polars_time::Duration::parse(tol);
                    if duration.months() != 0 {
                        return Err(PolarsError::ComputeError("Cannot use month offset in timedelta of an asof join. Consider using 4 weeks".into()));
                    }
                    let left_asof = df_left.column(&left_names[0])?;
                    use DataType::*;
                    match left_asof.dtype() {
                        Datetime(tu, _) | Duration(tu) => {
                            let tolerance = match tu {
                                TimeUnit::Nanoseconds => duration.duration_ns(),
                                TimeUnit::Microseconds => duration.duration_us(),
                                TimeUnit::Milliseconds => duration.duration_ms(),
                            };
                            options.tolerance = Some(AnyValue::from(tolerance))
                        }
                        Date => {
                            let days = (duration.duration_ms() / MILLISECONDS_IN_DAY) as i32;
                            options.tolerance = Some(AnyValue::from(days))
                        }
                        Time => {
                            let tolerance = duration.duration_ns();
                            options.tolerance = Some(AnyValue::from(tolerance))
                        }
                        _ => {
                            panic!("can only use timedelta string language with Date/Datetime/Duration/Time dtypes")
                        }
                    }
                }
            }
        }

        let df = df_left.join(
            &df_right,
            &left_names,
            &right_names,
            self.how.clone(),
            Some(self.suffix.clone().into_owned()),
        );

        if state.verbose {
            eprintln!("{:?} join dataframes finished", self.how);
        };
        df
    }
}
