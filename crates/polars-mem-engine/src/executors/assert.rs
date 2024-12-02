use std::borrow::Cow;
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::AnyValue;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_core::POOL;
use polars_error::{polars_bail, polars_err, PolarsResult};
use polars_expr::prelude::PhysicalExpr;
use polars_expr::state::ExecutionState;
use polars_plan::plans::OnAssertionFail;
use polars_utils::pl_str::PlSmallStr;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::Executor;

pub struct AssertExec {
    pub(crate) input: Box<dyn Executor>,

    pub(crate) name: Option<PlSmallStr>,
    pub(crate) predicate: Arc<dyn PhysicalExpr>,
    pub(crate) on_fail: OnAssertionFail,

    has_window: bool,
    streamable: bool,
}

fn fail(
    name: Option<&PlSmallStr>,
    predicate: &Arc<dyn PhysicalExpr>,
    on_fail: OnAssertionFail,
) -> PolarsResult<()> {
    let predicate = &predicate.as_ref();

    match on_fail {
        OnAssertionFail::Warn => {
            if std::env::var("POLARS_SILENCE_ASSERT_WARN").as_deref() == Ok("1") {
                return Ok(());
            }

            match &name {
                None => eprintln!("WARN: Assertion with predicate '{predicate}' failed."),
                Some(name) => {
                    eprintln!("WARN: Assertion '{name}' with predicate '{predicate}' failed.")
                },
            }

            Ok(())
        },
        OnAssertionFail::Error => Err(polars_err!(AssertionFailed: match &name {
            None => format!("Assertion with predicate '{predicate}' failed."),
            Some(name) => format!("Assertion '{name}' with predicate '{predicate}' failed."),
        })),
    }
}

impl AssertExec {
    pub fn new(
        input: Box<dyn Executor>,
        name: Option<PlSmallStr>,
        predicate: Arc<dyn PhysicalExpr>,
        on_fail: OnAssertionFail,

        has_window: bool,
        streamable: bool,
    ) -> Self {
        Self {
            input,
            name,
            predicate,
            on_fail,

            has_window,
            streamable,
        }
    }

    fn execute_hor(
        &mut self,
        df: DataFrame,
        state: &mut ExecutionState,
    ) -> PolarsResult<DataFrame> {
        if self.has_window {
            state.insert_has_window_function_flag()
        }
        let c = self.predicate.evaluate(&df, state)?;
        if self.has_window {
            state.clear_window_expr_cache()
        }

        match c.and_reduce()?.value() {
            AnyValue::Null | AnyValue::Boolean(false) => {
                fail(self.name.as_ref(), &self.predicate, self.on_fail)?
            },
            AnyValue::Boolean(true) => {},
            _ => polars_bail!(InvalidOperation: "Assertion produced a non-boolean"),
        }

        Ok(df)
    }

    fn execute_chunks(
        &mut self,
        chunks: Vec<DataFrame>,
        state: &ExecutionState,
    ) -> PolarsResult<DataFrame> {
        let iter = chunks.into_par_iter().map(|df| {
            let c = self.predicate.evaluate(&df, state)?;

            match c.and_reduce()?.value() {
                AnyValue::Null | AnyValue::Boolean(false) => {
                    fail(self.name.as_ref(), &self.predicate, self.on_fail)?
                },
                AnyValue::Boolean(true) => {},
                _ => polars_bail!(InvalidOperation: "Assertion produced a non-boolean"),
            }

            Ok(df)
        });
        let df = POOL.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;
        Ok(accumulate_dataframes_vertical_unchecked(df))
    }

    fn execute_impl(
        &mut self,
        mut df: DataFrame,
        state: &mut ExecutionState,
    ) -> PolarsResult<DataFrame> {
        if std::env::var("POLARS_SKIP_ASSERTS").as_deref() == Ok("1") {
            return Ok(df);
        }

        let n_partitions = POOL.current_num_threads();
        // Vertical parallelism.
        if self.streamable && df.height() > 0 {
            if df.first_col_n_chunks() > 1 {
                let chunks = df.split_chunks().collect::<Vec<_>>();
                self.execute_chunks(chunks, state)
            } else if df.width() < n_partitions {
                self.execute_hor(df, state)
            } else {
                let chunks = df.split_chunks_by_n(n_partitions, true);
                self.execute_chunks(chunks, state)
            }
        } else {
            self.execute_hor(df, state)
        }
    }
}

impl Executor for AssertExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run AssertExec")
            }
        }
        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            let fn_name = match self.on_fail {
                OnAssertionFail::Warn => "assert_warn",
                OnAssertionFail::Error => "assert",
            };

            let predicate = &self.predicate.as_ref();

            match &self.name {
                None => Cow::Owned(format!(".{fn_name}({predicate})",)),
                Some(name) => Cow::Owned(format!(".{fn_name}({name} = {predicate})")),
            }
        } else {
            Cow::Borrowed("")
        };

        state.clone().record(
            || {
                let df = self.execute_impl(df, state);
                if state.verbose() {
                    eprintln!("dataframe asserted");
                }
                df
            },
            profile_name,
        )
    }
}
