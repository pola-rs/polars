mod cache;
mod executor;
mod ext_context;
mod filter;
mod groupby;
mod groupby_dynamic;
mod groupby_partitioned;
mod groupby_rolling;
mod join;
mod projection;
#[cfg(feature = "python")]
mod python_scan;
mod scan;
mod slice;
mod sort;
mod stack;
mod udf;
mod union;
mod unique;

use std::borrow::Cow;

pub use executor::*;
use polars_core::POOL;
use polars_plan::global::FETCH_ROWS;
use polars_plan::utils::*;
use rayon::prelude::*;

pub(super) use self::cache::*;
pub(super) use self::ext_context::*;
pub(super) use self::filter::*;
pub(super) use self::groupby::*;
#[cfg(feature = "dynamic_groupby")]
pub(super) use self::groupby_dynamic::*;
pub(super) use self::groupby_partitioned::*;
#[cfg(feature = "dynamic_groupby")]
pub(super) use self::groupby_rolling::*;
pub(super) use self::join::*;
pub(super) use self::projection::*;
#[cfg(feature = "python")]
pub(super) use self::python_scan::*;
pub(super) use self::scan::*;
pub(super) use self::slice::*;
pub(super) use self::sort::*;
pub(super) use self::stack::*;
pub(super) use self::udf::*;
pub(super) use self::union::*;
pub(super) use self::unique::*;
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
    let mut windows: Vec<(String, Vec<(u32, Arc<dyn PhysicalExpr>)>)> = vec![];
    let mut other = Vec::with_capacity(exprs.len());

    // first we partition the window function by the values they group over.
    // the groupby values should be cached
    let mut index = 0u32;
    exprs.iter().for_each(|phys| {
        index += 1;
        let e = phys.as_expression().unwrap();

        let mut is_window = false;
        for e in e.into_iter() {
            if let Expr::Window { partition_by, .. } = e {
                let groupby = format!("{:?}", partition_by.as_slice());
                if let Some(tpl) = windows.iter_mut().find(|tpl| tpl.0 == groupby) {
                    tpl.1.push((index, phys.clone()))
                } else {
                    windows.push((groupby, vec![(index, phys.clone())]))
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

    for partition in windows {
        // clear the cache for every partitioned group
        let mut state = state.split();
        // inform the expression it has window functions.
        state.insert_has_window_function_flag();

        // don't bother caching if we only have a single window function in this partition
        if partition.1.len() == 1 {
            state.remove_cache_window_flag();
        } else {
            state.insert_cache_window_flag();
        }

        for (index, e) in partition.1 {
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
    state.expr_cache = Some(Default::default());
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
    state.clear_window_expr_cache();
    state.expr_cache = None;

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
            polars_ensure!(names.insert(name), duplicate = name);
        }
    }
    // If all series are the same length it is ok. If not we can broadcast Series of length one.
    if !all_equal_len {
        selected_columns = selected_columns
            .into_iter()
            .map(|series| {
                Ok(if series.len() == 1 && df_height > 1 {
                    series.new_from_index(0, df_height)
                } else if series.len() == df_height || series.len() == 0 {
                    series
                } else {
                    polars_bail!(
                        ComputeError: "series length {} doesn't match the dataframe height of {}",
                        series.len(), df_height
                    );
                })
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
