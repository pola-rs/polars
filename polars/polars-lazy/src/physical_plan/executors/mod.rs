pub mod cache;
pub mod drop_duplicates;
pub mod explode;
pub mod filter;
pub mod groupby;
pub mod join;
pub mod melt;
pub mod projection;
pub mod scan;
pub mod slice;
pub mod sort;
pub mod stack;
pub mod udf;

use super::*;
use crate::logical_plan::FETCH_ROWS;
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
    let zero_length = df.height() == 0;
    let selected_columns = POOL.install(|| {
        exprs
            .par_iter()
            .map(|expr| expr.evaluate(df, state))
            .collect::<Result<Vec<Series>>>()
    })?;
    {
        let mut names = PlHashSet::with_capacity(exprs.len());
        for s in &selected_columns {
            let name = s.name();
            if !names.insert(name) {
                return Err(PolarsError::Duplicate(
                    format!("Column with name: '{}' has more than one occurrences", name).into(),
                ));
            }
        }
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
