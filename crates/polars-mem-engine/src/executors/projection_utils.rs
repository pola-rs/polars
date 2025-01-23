use polars_plan::constants::CSE_REPLACED;
use polars_utils::itertools::Itertools;

use super::*;

pub(super) fn profile_name(
    s: &dyn PhysicalExpr,
    input_schema: &Schema,
) -> PolarsResult<PlSmallStr> {
    match s.to_field(input_schema) {
        Err(e) => Err(e),
        Ok(fld) => Ok(fld.name),
    }
}

type IdAndExpression = (u32, Arc<dyn PhysicalExpr>);

#[cfg(feature = "dynamic_group_by")]
fn rolling_evaluate(
    df: &DataFrame,
    state: &ExecutionState,
    rolling: PlHashMap<&RollingGroupOptions, Vec<IdAndExpression>>,
) -> PolarsResult<Vec<Vec<(u32, Column)>>> {
    POOL.install(|| {
        rolling
            .par_iter()
            .map(|(options, partition)| {
                // clear the cache for every partitioned group
                let state = state.split();

                let (_time_key, _keys, groups) = df.rolling(vec![], options)?;

                let groups_key = format!("{:?}", options);
                // Set the groups so all expressions in partition can use it.
                // Create a separate scope, so the lock is dropped, otherwise we deadlock when the
                // rolling expression try to get read access.
                state.window_cache.insert_groups(groups_key, groups);
                partition
                    .par_iter()
                    .map(|(idx, expr)| expr.evaluate(df, &state).map(|s| (*idx, s)))
                    .collect::<PolarsResult<Vec<_>>>()
            })
            .collect()
    })
}

fn window_evaluate(
    df: &DataFrame,
    state: &ExecutionState,
    window: PlHashMap<String, Vec<IdAndExpression>>,
) -> PolarsResult<Vec<Vec<(u32, Column)>>> {
    if window.is_empty() {
        return Ok(vec![]);
    }
    let n_threads = POOL.current_num_threads();

    let max_hor = window.values().map(|v| v.len()).max().unwrap_or(0);
    let vert = window.len();

    // We don't want to cache and parallel horizontally and vertically as that keeps many cache
    // states alive.
    let (cache, par_vertical, par_horizontal) = if max_hor >= n_threads || max_hor >= vert {
        (true, false, true)
    } else {
        (false, true, true)
    };

    let apply = |partition: &[(u32, Arc<dyn PhysicalExpr>)]| {
        // clear the cache for every partitioned group
        let mut state = state.split();
        // inform the expression it has window functions.
        state.insert_has_window_function_flag();

        // caching more than one window expression is a complicated topic for another day
        // see issue #2523
        let cache = cache
            && partition.len() > 1
            && partition.iter().all(|(_, e)| {
                e.as_expression()
                    .unwrap()
                    .into_iter()
                    .filter(|e| matches!(e, Expr::Window { .. }))
                    .count()
                    == 1
            });
        let mut first_result = None;
        // First run 1 to fill the cache. Condvars and such don't work as
        //  rayon threads should not be blocked.
        if cache {
            let first = &partition[0];
            let c = first.1.evaluate(df, &state)?;
            first_result = Some((first.0, c));
            state.insert_cache_window_flag();
        } else {
            state.remove_cache_window_flag();
        }

        let apply =
            |index: &u32, e: &Arc<dyn PhysicalExpr>| e.evaluate(df, &state).map(|c| (*index, c));

        let slice = &partition[first_result.is_some() as usize..];
        let mut results = if par_horizontal {
            slice
                .par_iter()
                .map(|(index, e)| apply(index, e))
                .collect::<PolarsResult<Vec<_>>>()?
        } else {
            slice
                .iter()
                .map(|(index, e)| apply(index, e))
                .collect::<PolarsResult<Vec<_>>>()?
        };

        if let Some(item) = first_result {
            results.push(item)
        }

        Ok(results)
    };

    if par_vertical {
        POOL.install(|| window.par_iter().map(|t| apply(t.1)).collect())
    } else {
        window.iter().map(|t| apply(t.1)).collect()
    }
}

fn execute_projection_cached_window_fns(
    df: &DataFrame,
    exprs: &[Arc<dyn PhysicalExpr>],
    state: &ExecutionState,
) -> PolarsResult<Vec<Column>> {
    // We partition by normal expression and window expression
    // - the normal expressions can run in parallel
    // - the window expression take more memory and often use the same group_by keys and join tuples
    //   so they are cached and run sequential

    // the partitioning messes with column order, so we also store the idx
    // and use those to restore the original projection order
    #[allow(clippy::type_complexity)]
    // String: partition_name,
    // u32: index,
    let mut windows: PlHashMap<String, Vec<IdAndExpression>> = PlHashMap::default();
    #[cfg(feature = "dynamic_group_by")]
    let mut rolling: PlHashMap<&RollingGroupOptions, Vec<IdAndExpression>> = PlHashMap::default();
    let mut other = Vec::with_capacity(exprs.len());

    // first we partition the window function by the values they group over.
    // the group_by values should be cached
    exprs.iter().enumerate_u32().for_each(|(index, phys)| {
        let mut is_window = false;
        if let Some(e) = phys.as_expression() {
            for e in e.into_iter() {
                if let Expr::Window {
                    partition_by,
                    options,
                    order_by,
                    ..
                } = e
                {
                    let entry = match options {
                        WindowType::Over(g) => {
                            let g: &str = g.into();
                            let mut key = format!("{:?}_{}", partition_by.as_slice(), g);
                            if let Some((e, k)) = order_by {
                                polars_expr::prelude::window_function_format_order_by(
                                    &mut key,
                                    e.as_ref(),
                                    k,
                                )
                            }
                            windows.entry(key).or_insert_with(Vec::new)
                        },
                        #[cfg(feature = "dynamic_group_by")]
                        WindowType::Rolling(options) => {
                            rolling.entry(options).or_insert_with(Vec::new)
                        },
                    };
                    entry.push((index, phys.clone()));
                    is_window = true;
                    break;
                }
            }
        } else {
            // Window physical expressions always have the `Expr`.
            is_window = false;
        }
        if !is_window {
            other.push((index, phys.as_ref()))
        }
    });

    let mut selected_columns = POOL.install(|| {
        other
            .par_iter()
            .map(|(idx, expr)| expr.evaluate(df, state).map(|s| (*idx, s)))
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    // Run partitioned rolling expressions.
    // Per partition we run in parallel. We compute the groups before and store them once per partition.
    // The rolling expression knows how to fetch the groups.
    #[cfg(feature = "dynamic_group_by")]
    {
        let (a, b) = POOL.join(
            || rolling_evaluate(df, state, rolling),
            || window_evaluate(df, state, windows),
        );

        let partitions = a?;
        for part in partitions {
            selected_columns.extend_from_slice(&part)
        }
        let partitions = b?;
        for part in partitions {
            selected_columns.extend_from_slice(&part)
        }
    }
    #[cfg(not(feature = "dynamic_group_by"))]
    {
        let partitions = window_evaluate(df, state, windows)?;
        for part in partitions {
            selected_columns.extend_from_slice(&part)
        }
    }

    selected_columns.sort_unstable_by_key(|tpl| tpl.0);
    let selected_columns = selected_columns.into_iter().map(|tpl| tpl.1).collect();
    Ok(selected_columns)
}

fn run_exprs_par(
    df: &DataFrame,
    exprs: &[Arc<dyn PhysicalExpr>],
    state: &ExecutionState,
) -> PolarsResult<Vec<Column>> {
    POOL.install(|| {
        exprs
            .par_iter()
            .map(|expr| expr.evaluate(df, state))
            .collect()
    })
}

fn run_exprs_seq(
    df: &DataFrame,
    exprs: &[Arc<dyn PhysicalExpr>],
    state: &ExecutionState,
) -> PolarsResult<Vec<Column>> {
    exprs.iter().map(|expr| expr.evaluate(df, state)).collect()
}

pub(super) fn evaluate_physical_expressions(
    df: &mut DataFrame,
    exprs: &[Arc<dyn PhysicalExpr>],
    state: &ExecutionState,
    has_windows: bool,
    run_parallel: bool,
) -> PolarsResult<Vec<Column>> {
    let expr_runner = if has_windows {
        execute_projection_cached_window_fns
    } else if run_parallel && exprs.len() > 1 {
        run_exprs_par
    } else {
        run_exprs_seq
    };

    let selected_columns = expr_runner(df, exprs, state)?;

    if has_windows {
        state.clear_window_expr_cache();
    }

    Ok(selected_columns)
}

pub(super) fn check_expand_literals(
    df: &DataFrame,
    phys_expr: &[Arc<dyn PhysicalExpr>],
    mut selected_columns: Vec<Column>,
    zero_length: bool,
    options: ProjectionOptions,
) -> PolarsResult<DataFrame> {
    let Some(first_len) = selected_columns.first().map(|s| s.len()) else {
        return Ok(DataFrame::empty());
    };
    let duplicate_check = options.duplicate_check;
    let should_broadcast = options.should_broadcast;

    // When we have CSE we cannot verify scalars yet.
    let verify_scalar = if !df.get_columns().is_empty() {
        !df.get_columns()[df.width() - 1]
            .name()
            .starts_with(CSE_REPLACED)
    } else {
        true
    };

    let mut df_height = 0;
    let mut has_empty = false;
    let mut all_equal_len = true;
    {
        let mut names = PlHashSet::with_capacity(selected_columns.len());
        for s in &selected_columns {
            let len = s.len();
            has_empty |= len == 0;
            df_height = std::cmp::max(df_height, len);
            if len != first_len {
                all_equal_len = false;
            }
            let name = s.name();

            if duplicate_check && !names.insert(name) {
                let msg = format!(
                    "the name '{}' is duplicate\n\n\
                    It's possible that multiple expressions are returning the same default column \
                    name. If this is the case, try renaming the columns with \
                    `.alias(\"new_name\")` to avoid duplicate column names.",
                    name
                );
                return Err(PolarsError::Duplicate(msg.into()));
            }
        }
    }

    // If all series are the same length it is ok. If not we can broadcast Series of length one.
    if !all_equal_len && should_broadcast {
        selected_columns = selected_columns
            .into_iter()
            .zip(phys_expr)
            .map(|(series, phys)| {
                Ok(match series.len() {
                    0 if df_height == 1 => series,
                    1 => {
                         if !has_empty && df_height == 1 {
                            series
                        } else {
                            if has_empty {
                                polars_ensure!(df_height == 1,
                                ShapeMismatch: "Series length {} doesn't match the DataFrame height of {}",
                                series.len(), df_height
                            );

                            }

                            if verify_scalar && !phys.is_scalar() && std::env::var("POLARS_ALLOW_NON_SCALAR_EXP").as_deref() != Ok("1") {
                                    let identifier = match phys.as_expression() {
                                        Some(e) => format!("expression: {}", e),
                                        None => "this Series".to_string(),
                                    };
                                    polars_bail!(ShapeMismatch: "Series {}, length {} doesn't match the DataFrame height of {}\n\n\
                                        If you want {} to be broadcasted, ensure it is a scalar (for instance by adding '.first()').",
                                        series.name(), series.len(), df_height *(!has_empty as usize), identifier
                                    );
                            }
                            series.new_from_index(0, df_height * (!has_empty as usize) )
                        }
                    },
                    len if len == df_height => {
                        series
                    },
                    _ => {
                        polars_bail!(
                        ShapeMismatch: "Series length {} doesn't match the DataFrame height of {}",
                        series.len(), df_height
                    )
                    }
                })
            })
            .collect::<PolarsResult<_>>()?
    }

    // @scalar-opt
    let selected_columns = selected_columns.into_iter().collect::<Vec<_>>();

    let df = unsafe { DataFrame::new_no_checks_height_from_first(selected_columns) };

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
