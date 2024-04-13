use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use super::*;

pub struct FilterExec {
    pub(crate) predicate: Arc<dyn PhysicalExpr>,
    pub(crate) input: Box<dyn Executor>,
    // if the predicate contains a window function
    has_window: bool,
    streamable: bool,
}

fn series_to_mask(s: &Series) -> PolarsResult<&BooleanChunked> {
    s.bool().map_err(|_| {
        polars_err!(
            ComputeError: "filter predicate must be of type `Boolean`, got `{}`", s.dtype()
        )
    })
}

impl FilterExec {
    pub fn new(
        predicate: Arc<dyn PhysicalExpr>,
        input: Box<dyn Executor>,
        has_window: bool,
        streamable: bool,
    ) -> Self {
        Self {
            predicate,
            input,
            has_window,
            streamable,
        }
    }

    fn execute_impl(
        &mut self,
        df: DataFrame,
        state: &mut ExecutionState,
    ) -> PolarsResult<DataFrame> {
        // Vertical parallelism.
        let df = if self.streamable && df.n_chunks() > 1 && df.height() > 0 {
            let chunks = df.split_chunks().collect::<Vec<_>>();
            let iter = chunks.into_par_iter().map(|df| {
                let s = self.predicate.evaluate(&df, state)?;
                df.filter(series_to_mask(&s)?)
            });

            let df = POOL.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;
            accumulate_dataframes_vertical_unchecked(df)
        } else {
            if self.has_window {
                state.insert_has_window_function_flag()
            }
            let s = self.predicate.evaluate(&df, state)?;
            if self.has_window {
                state.clear_window_expr_cache()
            }
            df.filter(series_to_mask(&s)?)?
        };
        Ok(df)
    }
}

impl Executor for FilterExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run FilterExec")
            }
        }
        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            Cow::Owned(format!(".filter({})", &self.predicate.as_ref()))
        } else {
            Cow::Borrowed("")
        };

        state.clone().record(
            || {
                let df = self.execute_impl(df, state);
                if state.verbose() {
                    eprintln!("dataframe filtered");
                }
                df
            },
            profile_name,
        )
    }
}
