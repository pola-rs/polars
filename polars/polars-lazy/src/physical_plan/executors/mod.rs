pub mod cache;
pub mod drop_duplicates;
pub mod explode;
pub mod filter;
pub mod groupby;
pub mod join;
pub mod melt;
pub mod scan;
pub mod slice;
pub mod sort;
pub mod stack;
pub mod udf;
pub mod various;

use super::*;
use crate::logical_plan::FETCH_ROWS;
use itertools::Itertools;
use polars_core::POOL;
use rayon::prelude::*;
use std::path::PathBuf;

const POLARS_VERBOSE: &str = "POLARS_VERBOSE";

fn set_n_rows(stop_after_n_rows: Option<usize>) -> Option<usize> {
    let fetch_rows = FETCH_ROWS.with(|fetch_rows| fetch_rows.get());
    match fetch_rows {
        None => stop_after_n_rows,
        Some(n) => Some(n),
    }
}

pub(crate) fn evaluate_physical_expressions(
    df: &DataFrame,
    exprs: &[Arc<dyn PhysicalExpr>],
    state: &ExecutionState,
) -> Result<DataFrame> {
    let height = df.height();
    let mut selected_columns = POOL.install(|| {
        exprs
            .par_iter()
            .map(|expr| expr.evaluate(df, state))
            .collect::<Result<Vec<Series>>>()
    })?;

    // If all series are the same length it is ok. If not we can broadcast Series of length one.
    if selected_columns.len() > 1 {
        let all_equal_len = selected_columns.iter().map(|s| s.len()).all_equal();
        if !all_equal_len {
            selected_columns = selected_columns
                .into_iter()
                .map(|series| {
                    if series.len() == 1 && height > 1 {
                        series.expand_at_index(0, height)
                    } else {
                        series
                    }
                })
                .collect()
        }
    }

    Ok(DataFrame::new_no_checks(selected_columns))
}
