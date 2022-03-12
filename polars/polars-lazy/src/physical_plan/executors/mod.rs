pub(crate) mod cache;
pub(crate) mod drop_duplicates;
pub(crate) mod explode;
pub(crate) mod filter;
pub(crate) mod groupby;
pub(crate) mod groupby_dynamic;
pub(crate) mod groupby_rolling;
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

fn execute_projection_cached_window_fns(
    df: &DataFrame,
    exprs: &[Arc<dyn PhysicalExpr>],
    state: &ExecutionState,
) -> Result<Vec<Series>> {
    // We partition by normal expression and window expression
    // - the normal expressions can run in parallel
    // - the window expression take more memory and often use the same groupby keys and join tuples
    //   so they are cached and run sequential

    // the partitioning messes with column order, so we also store the idx
    // and use those to restore the original projection order
    #[allow(clippy::type_complexity)]
    // String: partion_name,
    // u32: index,
    // bool: flatten (we must run those first because they need a sorted group tuples.
    //       if we cache the group tuples we must ensure we cast the sorted onces.
    let mut windows: Vec<(String, Vec<(u32, bool, Arc<dyn PhysicalExpr>)>)> = vec![];
    let mut other = Vec::with_capacity(exprs.len());

    // first we partition the window function by the values they group over.
    // the groupby values should be cached
    let mut index = 0u32;
    exprs.iter().for_each(|phys| {
        index += 1;
        let e = phys.as_expression();

        let mut is_window = false;
        for e in e.into_iter() {
            if let Expr::Window {
                partition_by,
                options,
                ..
            } = e
            {
                let groupby = format!("{:?}", partition_by.as_slice());
                if let Some(tpl) = windows.iter_mut().find(|tpl| tpl.0 == groupby) {
                    tpl.1.push((index, options.explode, phys.clone()))
                } else {
                    windows.push((groupby, vec![(index, options.explode, phys.clone())]))
                }
                is_window = true;
                break;
            }
        }
        if !is_window {
            other.push((index, phys))
        }
    });

    let mut selected_columns = POOL.install(|| {
        other
            .par_iter()
            .map(|(idx, expr)| expr.evaluate(df, state).map(|s| (*idx, s)))
            .collect::<Result<Vec<_>>>()
    })?;

    for mut partition in windows {
        // clear the cache for every partitioned group
        let mut state = state.clone();
        state.clear_expr_cache();

        // don't bother caching if we only have a single window function in this partition
        if partition.1.len() == 1 {
            state.cache_window = false;
        } else {
            state.cache_window = true;
        }

        partition.1.sort_unstable_by_key(|(_idx, explode, _)| {
            // negate as `false` will be first and we want the exploded
            // e.g. the sorted groups cd to be the first to fill the cache.
            !explode
        });

        for (index, _, e) in partition.1 {
            // caching more than one window expression is a complicated topic for another day
            // see issue #2523
            state.cache_window = e
                .as_expression()
                .into_iter()
                .filter(|e| matches!(e, Expr::Window { .. }))
                .count()
                == 1;

            let s = e.evaluate(df, &state)?;
            selected_columns.push((index, s));
        }
    }

    selected_columns.sort_unstable_by_key(|tpl| tpl.0);
    let selected_columns = selected_columns.into_iter().map(|tpl| tpl.1).collect();
    Ok(selected_columns)
}

pub(crate) fn evaluate_physical_expressions(
    df: &DataFrame,
    exprs: &[Arc<dyn PhysicalExpr>],
    state: &ExecutionState,
    has_windows: bool,
) -> Result<DataFrame> {
    let zero_length = df.height() == 0;
    let selected_columns = if has_windows {
        execute_projection_cached_window_fns(df, exprs, state)?
    } else {
        POOL.install(|| {
            exprs
                .par_iter()
                .map(|expr| expr.evaluate(df, state))
                .collect::<Result<_>>()
        })?
    };
    state.clear_schema_cache();

    check_expand_literals(selected_columns, zero_length)
}

fn check_expand_literals(
    mut selected_columns: Vec<Series>,
    zero_length: bool,
) -> Result<DataFrame> {
    let first_len = selected_columns[0].len();
    let mut df_height = 0;
    let mut all_equal_len = true;
    {
        let mut names = PlHashSet::with_capacity(selected_columns.len());
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
