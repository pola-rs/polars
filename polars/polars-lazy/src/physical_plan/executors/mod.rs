mod cache;
mod drop_duplicates;
mod executor;
mod explode;
mod ext_context;
mod filter;
mod groupby;
mod groupby_dynamic;
mod groupby_partitioned;
mod groupby_rolling;
mod join;
mod melt;
mod projection;
#[cfg(feature = "python")]
mod python_scan;
mod scan;
mod slice;
mod sort;
mod stack;
mod udf;
mod union;

use std::borrow::Cow;
use std::path::PathBuf;

pub use executor::*;
use polars_core::POOL;
use polars_plan::global::FETCH_ROWS;
use polars_plan::utils::*;
use rayon::prelude::*;

pub(super) use self::cache::*;
pub(super) use self::drop_duplicates::*;
pub(super) use self::explode::*;
pub(super) use self::ext_context::*;
pub(super) use self::filter::*;
pub(super) use self::groupby::*;
#[cfg(feature = "dynamic_groupby")]
pub(super) use self::groupby_dynamic::*;
pub(super) use self::groupby_partitioned::*;
#[cfg(feature = "dynamic_groupby")]
pub(super) use self::groupby_rolling::*;
pub(super) use self::join::*;
pub(super) use self::melt::*;
pub(super) use self::projection::*;
#[cfg(feature = "python")]
pub(super) use self::python_scan::*;
pub(super) use self::scan::*;
pub(super) use self::slice::*;
pub(super) use self::sort::*;
pub(super) use self::stack::*;
pub(super) use self::udf::*;
pub(super) use self::union::*;
use super::*;

fn execute_projection_cached_window_fns(
    df: &DataFrame,
    exprs: &[Arc<dyn PhysicalExpr>],
    state: &ExecutionState,
) -> PolarsResult<Vec<Series>> {
    // We partition by normal expression and window expression
    // - the normal expressions can run in parallel
    // - the window expression take more memory and often use the same groupby keys and join tuples
    //   so they are cached and run sequential

    // the partitioning messes with column order, so we also store the idx
    // and use those to restore the original projection order
    #[allow(clippy::type_complexity)]
    // String: partition_name,
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
        let e = phys.as_expression().unwrap();

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
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    for mut partition in windows {
        // clear the cache for every partitioned group
        let mut state = state.split();

        // don't bother caching if we only have a single window function in this partition
        if partition.1.len() == 1 {
            state.remove_cache_window_flag();
        } else {
            state.insert_cache_window_flag();
        }

        partition.1.sort_unstable_by_key(|(_idx, explode, _)| {
            // negate as `false` will be first and we want the exploded
            // e.g. the sorted groups cd to be the first to fill the cache.
            !explode
        });

        for (index, _, e) in partition.1 {
            if e.as_expression()
                .unwrap()
                .into_iter()
                .filter(|e| matches!(e, Expr::Window { .. }))
                .count()
                == 1
            {
                state.insert_cache_window_flag();
            }
            // caching more than one window expression is a complicated topic for another day
            // see issue #2523
            else {
                state.remove_cache_window_flag();
            }

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
    state: &mut ExecutionState,
    has_windows: bool,
) -> PolarsResult<DataFrame> {
    let zero_length = df.height() == 0;
    let selected_columns = if has_windows {
        execute_projection_cached_window_fns(df, exprs, state)?
    } else {
        POOL.install(|| {
            exprs
                .par_iter()
                .map(|expr| expr.evaluate(df, state))
                .collect::<PolarsResult<_>>()
        })?
    };

    check_expand_literals(selected_columns, zero_length)
}

fn check_expand_literals(
    mut selected_columns: Vec<Series>,
    zero_length: bool,
) -> PolarsResult<DataFrame> {
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
                    format!("Column with name: '{name}' has more than one occurrences").into(),
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
                    Ok(series.new_from_index(0, df_height))
                } else if series.len() == df_height || series.len() == 0 {
                    Ok(series)
                } else {
                    Err(PolarsError::ComputeError(
                        format!(
                            "Series {series:?} does not match the DataFrame height of {df_height}",
                        )
                        .into(),
                    ))
                }
            })
            .collect::<PolarsResult<_>>()?
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
