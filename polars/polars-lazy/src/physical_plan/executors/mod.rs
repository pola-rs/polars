pub(crate) mod cache;
pub(crate) mod drop_duplicates;
pub(crate) mod explode;
pub(crate) mod filter;
pub(crate) mod groupby;
pub(crate) mod groupby_dynamic;
pub(crate) mod join;
pub(crate) mod melt;
pub(crate) mod projection;
pub(crate) mod scan;
pub(crate) mod slice;
pub(crate) mod sort;
pub(crate) mod stack;
pub(crate) mod udf;
pub(crate) mod union;

use super::*;
use crate::logical_plan::FETCH_ROWS;
use polars_core::POOL;
use rayon::prelude::*;
use std::path::PathBuf;

const POLARS_VERBOSE: &str = "POLARS_VERBOSE";

fn set_n_rows(n_rows: Option<usize>) -> Option<usize> {
    let fetch_rows = FETCH_ROWS.with(|fetch_rows| fetch_rows.get());
    match fetch_rows {
        None => n_rows,
        Some(n) => Some(n),
    }
}

pub(crate) fn evaluate_physical_expressions(
    df: &DataFrame,
    exprs: &[Arc<dyn PhysicalExpr>],
    state: &ExecutionState,
) -> Result<DataFrame> {
    let zero_length = df.height() == 0;
    let mut selected_columns = POOL.install(|| {
        exprs
            .par_iter()
            .map(|expr| expr.evaluate(df, state))
            .collect::<Result<Vec<Series>>>()
    })?;
    let first_len = selected_columns[0].len();
    let mut df_height = 0;
    let mut all_equal_len = true;
    {
        let mut names = PlHashSet::with_capacity(exprs.len());
        for s in &selected_columns {
            let len = s.len();
            df_height = std::cmp::max(df_height, len);
            if len != first_len {
                all_equal_len = false;
            }
            let name = s.name();
            if !names.insert(name) {
                return Err(PolarsError::Duplicate(
                    format!("Column with name: '{}' has more than one occurrences", name).into(),
                ));
            }
        }
    }
    // If all series are the same length it is ok. If not we can broadcast Series of length one.
    if !all_equal_len {
        selected_columns = selected_columns
            .into_iter()
            .map(|series| {
                if series.len() == 1 && df_height > 1 {
                    series.expand_at_index(0, df_height)
                } else {
                    series
                }
            })
            .collect()
    }

    let df = DataFrame::new_no_checks(selected_columns);

    // a literal could be projected to a zero length dataframe.
    // This prevents a panic.
    let df = if zero_length {
        let min = df.get_columns().iter().map(|s| s.len()).min();
        if min.is_some() {
            df.head(min)
        } else {
            df
        }
    } else {
        df
    };
    Ok(df)
}
