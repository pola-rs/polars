use std::ops::BitAnd;

use arrow::array::BooleanArray;
use arrow::pushable::Pushable;
use polars_core::frame::column::ScalarColumn;
use polars_core::functions::concat_df_horizontal;
use polars_ops::frame::DataFrameJoinOps;

use super::*;

pub struct JoinExec {
    input_left: Option<Box<dyn Executor>>,
    input_right: Option<Box<dyn Executor>>,
    left_on: Vec<Arc<dyn PhysicalExpr>>,
    right_on: Vec<Arc<dyn PhysicalExpr>>,
    non_equi_predicates: Vec<Arc<dyn PhysicalExpr>>,
    parallel: bool,
    args: JoinArgs,
}

impl JoinExec {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input_left: Box<dyn Executor>,
        input_right: Box<dyn Executor>,
        left_on: Vec<Arc<dyn PhysicalExpr>>,
        right_on: Vec<Arc<dyn PhysicalExpr>>,
        non_equi_predicates: Vec<Arc<dyn PhysicalExpr>>,
        parallel: bool,
        args: JoinArgs,
    ) -> Self {
        JoinExec {
            input_left: Some(input_left),
            input_right: Some(input_right),
            left_on,
            right_on,
            non_equi_predicates,
            parallel,
            args,
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
            // propagate the fetch_rows static value to the spawning threads.
            let fetch_rows = FETCH_ROWS.with(|fetch_rows| fetch_rows.get());

            POOL.join(
                move || {
                    FETCH_ROWS.with(|fr| fr.set(fetch_rows));
                    input_left.execute(&mut state_left)
                },
                move || {
                    FETCH_ROWS.with(|fr| fr.set(fetch_rows));
                    input_right.execute(&mut state_right)
                },
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
                .map(|s| Ok(s.to_field(&df_left.schema())?.name))
                .collect::<PolarsResult<Vec<_>>>()?;
            let name = comma_delimited("join".to_string(), &by);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(|| {

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

            // prepare the tolerance
            // we must ensure that we use the right units
            #[cfg(feature = "asof_join")]
            {
                if let JoinType::AsOf(options) = &mut self.args.how {
                    use polars_core::utils::arrow::temporal_conversions::MILLISECONDS_IN_DAY;
                    if let Some(tol) = &options.tolerance_str {
                        let duration = polars_time::Duration::parse(tol);
                        polars_ensure!(
                            duration.months() == 0,
                            ComputeError: "cannot use month offset in timedelta of an asof join; \
                            consider using 4 weeks"
                        );
                        let left_asof = df_left.column(left_on_series[0].name())?;
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

            let df = if !self.non_equi_predicates.is_empty() {
                polars_ensure!(
                    self.args.how == JoinType::Inner,
                    ComputeError: "Can currently only use non-equi join predicates with an inner join");
                polars_ensure!(
                    left_on_series.is_empty() && right_on_series.is_empty(),
                    ComputeError: "Can't mix non-equi join predicates with equi or inequality predicates");
                nested_loop_join(&df_left, &df_right, &self.non_equi_predicates, state)
            } else {
                df_left._join_impl(
                    &df_right,
                    left_on_series.into_iter().map(|c| c.take_materialized_series()).collect(),
                    right_on_series.into_iter().map(|c| c.take_materialized_series()).collect(),
                    self.args.clone(),
                    true,
                    state.verbose(),
                )
            };

            if state.verbose() {
                eprintln!("{:?} join dataframes finished", self.args.how);
            };
            df

        }, profile_name)
    }
}

fn nested_loop_join(
    left: &DataFrame,
    right: &DataFrame,
    predicates: &[Arc<dyn PhysicalExpr>],
    state: &ExecutionState,
) -> PolarsResult<DataFrame> {
    // Nested loop joins are implemented separately to other join types as polars-ops doesn't have a concept of expressions
    debug_assert!(!predicates.is_empty());
    let (outer, inner, reversed) = if left.height() <= right.height() {
        (left, right, false)
    } else {
        (right, left, true)
    };
    let mut outer_indices = vec![];
    let mut inner_indices = vec![];
    let inner_height = inner.height();
    // TODO: Parallelise over outer row index?
    for outer_row_idx in 0..outer.height() {
        let outer_scalars = expand_row_to_scalars(outer, outer_row_idx, inner_height)?;
        let to_concat = if reversed {
            &[inner.clone(), outer_scalars]
        } else {
            &[outer_scalars, inner.clone()]
        };
        let combined = concat_df_horizontal(to_concat, true)?;
        let mut matching = predicates[0].evaluate(&combined, state)?.bool()?.clone();
        for pred in predicates[1..].iter() {
            let other_matching = pred.evaluate(&combined, state)?.bool()?.clone();
            matching = matching.bitand(other_matching);
        }
        let mut chunk_offset = 0;
        for chunk in matching.chunks().iter() {
            let chunk: &BooleanArray = chunk.as_any().downcast_ref().unwrap();
            for (i, is_match) in chunk.iter().enumerate() {
                if is_match == Some(true) {
                    inner_indices.push((chunk_offset + i) as IdxSize);
                }
            }
            chunk_offset += chunk.len();
        }

        outer_indices.extend_constant(
            inner_indices.len() - outer_indices.len(),
            outer_row_idx as IdxSize,
        );
    }

    let outer_indices = IdxCa::from_vec(PlSmallStr::EMPTY, outer_indices);
    let inner_indices = IdxCa::from_vec(PlSmallStr::EMPTY, inner_indices);
    let (left_rows, right_rows) = if reversed {
        (left.take(&inner_indices)?, right.take(&outer_indices)?)
    } else {
        (left.take(&outer_indices)?, right.take(&inner_indices)?)
    };
    // TODO: Handle renaming columns etc, can we reuse existing join logic for this?
    concat_df_horizontal(&[left_rows, right_rows], true)
}

// Take a row from the outer DataFrame and expand it to a DataFrame of scalar columns
fn expand_row_to_scalars(
    data_frame: &DataFrame,
    row_index: usize,
    height: usize,
) -> PolarsResult<DataFrame> {
    let row = unsafe { data_frame.take_slice_unchecked(&[row_index as IdxSize]) };
    let scalar_columns = row
        .iter()
        .cloned()
        .map(|series| Column::Scalar(ScalarColumn::from_single_value_series(series, height)))
        .collect::<Vec<_>>();
    DataFrame::new(scalar_columns)
}
