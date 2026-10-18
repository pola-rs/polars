use polars_core::utils::{_split_offsets, accumulate_dataframes_vertical_unchecked};

use super::*;

pub struct FilterExec {
    pub(crate) predicate: Arc<dyn PhysicalExpr>,
    pub(crate) input: Box<dyn Executor>,
    // if the predicate contains a window function
    has_window: bool,
    streamable: bool,
}

/// Minimum number of rows per chunk when evaluating the predicate in parallel.
const MIN_ROWS_PER_PREDICATE_CHUNK: usize = 8192;

/// Minimum number of rows per chunk when filtering in row chunks.
const MIN_ROWS_PER_FILTER_CHUNK: usize = 32768;

pub fn column_to_mask<'a>(
    c: &'a Column,
    expected_len: usize,
) -> PolarsResult<Cow<'a, BooleanChunked>> {
    match c {
        // We don't want to materialize scalars
        Column::Scalar(s) => {
            let len = s.len();
            polars_ensure!(len == expected_len || len == 1, ShapeMismatch: "filter predicate length of {len} doesn't match that of the DataFrame");

            if matches!(s.dtype(), DataType::Boolean) {
                let ca = match s.scalar().value() {
                    AnyValue::Null => BooleanChunked::full_null(PlSmallStr::EMPTY, 1),
                    AnyValue::Boolean(v) => BooleanChunked::new(PlSmallStr::EMPTY, [*v]),
                    _ => unreachable!(),
                };
                Ok(Cow::Owned(ca))
            } else {
                polars_bail!(ComputeError: "filter predicate must be of type `Boolean`, got `{}`", s.dtype())
            }
        },
        _ => c
            .bool()
            .map_err(|_| {
                polars_err!(
                    ComputeError: "filter predicate must be of type `Boolean`, got `{}`", c.dtype()
                )
            })
            .map(Cow::Borrowed),
    }
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

    /// Evaluates the predicate, in parallel over row chunks if worthwhile.
    fn evaluate_predicate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let n_chunks =
            (df.height() / MIN_ROWS_PER_PREDICATE_CHUNK).min(RAYON.current_num_threads());
        let is_column = matches!(self.predicate.as_expression(), Some(Expr::Column(_)));
        if !self.streamable || n_chunks < 2 || is_column {
            return self.predicate.evaluate(df, state);
        }

        let chunks = df.clone().split_chunks_by_n(n_chunks, true);
        let columns = RAYON.install(|| {
            chunks
                .into_par_iter()
                .map(|chunk| self.predicate.evaluate(&chunk, state))
                .collect::<PolarsResult<Vec<_>>>()
        })?;
        let mut columns = columns.into_iter();
        let mut out = columns.next().unwrap();
        for c in columns {
            out.append_owned(c)?;
        }
        Ok(out)
    }

    /// Evaluates the predicate and filters each of the existing chunks in parallel.
    fn execute_chunks(
        &mut self,
        chunks: Vec<DataFrame>,
        state: &ExecutionState,
    ) -> PolarsResult<DataFrame> {
        let iter = chunks.into_par_iter().map(|df| {
            let c = self.predicate.evaluate(&df, state)?;

            // @scalar-opt
            // @partition-opt
            df.filter(column_to_mask(&c, df.height())?.as_ref())
        });
        let df = RAYON.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;
        Ok(accumulate_dataframes_vertical_unchecked(df))
    }

    fn execute_impl(
        &mut self,
        mut df: DataFrame,
        state: &mut ExecutionState,
    ) -> PolarsResult<DataFrame> {
        // Use pre-chunked data.
        if self.streamable && df.height() > 0 && df.first_col_n_chunks() > 1 {
            let chunks = df.split_chunks().collect::<Vec<_>>();
            return self.execute_chunks(chunks, state);
        }

        if self.has_window {
            state.insert_has_window_function_flag()
        }
        let c = self.evaluate_predicate(&df, state)?;
        if self.has_window {
            state.clear_window_expr_cache()
        }

        // @scalar-opt
        // @partition-opt
        let mask = column_to_mask(&c, df.height())?;
        filter_par(&df, &mask)
    }
}

/// Filters the columns of `df` in parallel. Frames too narrow to keep all
/// threads busy are also split into row chunks.
fn filter_par(df: &DataFrame, mask: &BooleanChunked) -> PolarsResult<DataFrame> {
    let max_chunks = RAYON.current_num_threads().div_ceil(df.width().max(1));
    let n_chunks = (df.height() / MIN_ROWS_PER_FILTER_CHUNK).min(max_chunks);
    if n_chunks < 2 || mask.len() != df.height() {
        return df.filter(mask);
    }

    let dfs = RAYON.install(|| {
        _split_offsets(df.height(), n_chunks)
            .into_par_iter()
            .map(|(offset, len)| {
                let offset = offset as i64;
                df.slice(offset, len).filter(&mask.slice(offset, len))
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;
    Ok(accumulate_dataframes_vertical_unchecked(dfs))
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
        self.execute_impl(df, state)
    }
}
