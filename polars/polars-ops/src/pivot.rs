use polars_core::export::rayon::prelude::*;
use polars_core::frame::groupby::expr::PhysicalAggExpr;
use polars_core::prelude::*;
use polars_core::POOL;

const HASHMAP_INIT_SIZE: usize = 512;

#[derive(Clone)]
pub enum PivotAgg {
    First,
    Sum,
    Min,
    Max,
    Mean,
    Median,
    Count,
    Last,
    Expr(Arc<dyn PhysicalAggExpr + Send + Sync>),
}

fn restore_logical_type(s: &Series, logical_type: &DataType) -> Series {
    // restore logical type
    match logical_type {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(Some(rev_map)) => {
            let cats = s.u32().unwrap().clone();
            // safety:
            // the rev-map comes from these categoricals
            unsafe {
                CategoricalChunked::from_cats_and_rev_map_unchecked(cats, rev_map.clone())
                    .into_series()
            }
        }
        _ => s.cast(logical_type).unwrap(),
    }
}

/// Do a pivot operation based on the group key, a pivot column and an aggregation function on the values column.
///
/// # Note
/// Polars'/arrow memory is not ideal for transposing operations like pivots.
/// If you have a relatively large table, consider using a groupby over a pivot.
pub fn pivot<I0, S0, I1, S1, I2, S2>(
    pivot_df: &DataFrame,
    values: I0,
    index: I1,
    columns: I2,
    agg_fn: PivotAgg,
    sort_columns: bool,
) -> Result<DataFrame>
where
    I0: IntoIterator<Item = S0>,
    S0: AsRef<str>,
    I1: IntoIterator<Item = S1>,
    S1: AsRef<str>,
    I2: IntoIterator<Item = S2>,
    S2: AsRef<str>,
{
    let values = values
        .into_iter()
        .map(|s| s.as_ref().to_string())
        .collect::<Vec<_>>();
    let index = index
        .into_iter()
        .map(|s| s.as_ref().to_string())
        .collect::<Vec<_>>();
    let columns = columns
        .into_iter()
        .map(|s| s.as_ref().to_string())
        .collect::<Vec<_>>();
    pivot_impl(
        pivot_df,
        &values,
        &index,
        &columns,
        agg_fn,
        sort_columns,
        false,
    )
}

/// Do a pivot operation based on the group key, a pivot column and an aggregation function on the values column.
///
/// # Note
/// Polars'/arrow memory is not ideal for transposing operations like pivots.
/// If you have a relatively large table, consider using a groupby over a pivot.
pub fn pivot_stable<I0, S0, I1, S1, I2, S2>(
    pivot_df: &DataFrame,
    values: I0,
    index: I1,
    columns: I2,
    agg_fn: PivotAgg,
    sort_columns: bool,
) -> Result<DataFrame>
where
    I0: IntoIterator<Item = S0>,
    S0: AsRef<str>,
    I1: IntoIterator<Item = S1>,
    S1: AsRef<str>,
    I2: IntoIterator<Item = S2>,
    S2: AsRef<str>,
{
    let values = values
        .into_iter()
        .map(|s| s.as_ref().to_string())
        .collect::<Vec<_>>();
    let index = index
        .into_iter()
        .map(|s| s.as_ref().to_string())
        .collect::<Vec<_>>();
    let columns = columns
        .into_iter()
        .map(|s| s.as_ref().to_string())
        .collect::<Vec<_>>();

    pivot_impl(
        pivot_df,
        &values,
        &index,
        &columns,
        agg_fn,
        sort_columns,
        true,
    )
}

fn compute_col_idx(
    pivot_df: &DataFrame,
    column: &str,
    groups: &GroupsProxy,
) -> Result<(Vec<IdxSize>, Series)> {
    let column_s = pivot_df.column(column)?;
    let column_agg = unsafe { column_s.agg_first(groups) };
    let column_agg_physical = column_agg.to_physical_repr();

    let mut col_to_idx = PlHashMap::with_capacity(HASHMAP_INIT_SIZE);
    let mut idx = 0 as IdxSize;
    let col_locations = column_agg_physical
        .iter()
        .map(|v| {
            let idx = *col_to_idx.entry(v).or_insert_with(|| {
                let old_idx = idx;
                idx += 1;
                old_idx
            });
            idx
        })
        .collect();

    drop(col_to_idx);
    Ok((col_locations, column_agg))
}

fn compute_row_idx(
    pivot_df: &DataFrame,
    index: &[String],
    groups: &GroupsProxy,
    count: usize,
) -> Result<(Vec<IdxSize>, usize, Option<Vec<Series>>)> {
    let (row_locations, n_rows, row_index) = if index.len() == 1 {
        let index_s = pivot_df.column(&index[0])?;
        let index_agg = unsafe { index_s.agg_first(groups) };
        let index_agg_physical = index_agg.to_physical_repr();

        let mut row_to_idx =
            PlIndexMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());
        let mut idx = 0 as IdxSize;
        let row_locations = index_agg_physical
            .iter()
            .map(|v| {
                let idx = *row_to_idx.entry(v).or_insert_with(|| {
                    let old_idx = idx;
                    idx += 1;
                    old_idx
                });
                idx
            })
            .collect::<Vec<_>>();

        let row_index = match count {
            0 => {
                let s = Series::new(
                    &index[0],
                    row_to_idx.into_iter().map(|(k, _)| k).collect::<Vec<_>>(),
                );
                let s = restore_logical_type(&s, index_s.dtype());
                Some(vec![s])
            }
            _ => None,
        };

        (row_locations, idx as usize, row_index)
    } else {
        let index_s = pivot_df.columns(index)?;
        let index_agg_physical = index_s
            .iter()
            .map(|s| unsafe { s.agg_first(groups).to_physical_repr().into_owned() })
            .collect::<Vec<_>>();
        let mut iters = index_agg_physical
            .iter()
            .map(|s| s.iter())
            .collect::<Vec<_>>();
        let mut row_to_idx =
            PlIndexMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());
        let mut idx = 0 as IdxSize;

        let mut row_locations = Vec::with_capacity(groups.len());
        loop {
            match iters
                .iter_mut()
                .map(|it| it.next())
                .collect::<Option<Vec<_>>>()
            {
                None => break,
                Some(items) => {
                    let idx = *row_to_idx.entry(items).or_insert_with(|| {
                        let old_idx = idx;
                        idx += 1;
                        old_idx
                    });
                    row_locations.push(idx)
                }
            }
        }
        let row_index = match count {
            0 => Some(
                index
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let s = Series::new(
                            name,
                            row_to_idx
                                .iter()
                                .map(|(k, _)| {
                                    debug_assert!(i < k.len());
                                    unsafe { k.get_unchecked(i).clone() }
                                })
                                .collect::<Vec<_>>(),
                        );
                        restore_logical_type(&s, index_s[i].dtype())
                    })
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        };

        (row_locations, idx as usize, row_index)
    };

    Ok((row_locations, n_rows, row_index))
}

fn pivot_impl(
    pivot_df: &DataFrame,
    // these columns will be aggregated in the nested groupby
    values: &[String],
    // keys of the first groupby operation
    index: &[String],
    // these columns will be used for a nested groupby
    // the rows of this nested groupby will be pivoted as header column values
    columns: &[String],
    // aggregation function
    agg_fn: PivotAgg,
    sort_columns: bool,
    stable: bool,
) -> Result<DataFrame> {
    if index.is_empty() {
        return Err(PolarsError::ComputeError(
            "index cannot be zero length".into(),
        ));
    }

    let mut final_cols = vec![];

    let mut count = 0;
    let out: Result<()> = POOL.install(|| {
        for column in columns {
            let mut groupby = index.to_vec();
            groupby.push(column.clone());

            let groups = pivot_df.groupby_stable(groupby)?.take_groups();

            // these are the row locations
            if !stable {
                println!("unstable pivot not yet supported, using stable pivot");
            };

            let (col, row) = POOL.join(
                || compute_col_idx(pivot_df, column, &groups),
                || compute_row_idx(pivot_df, index, &groups, count),
            );
            let (col_locations, column_agg) = col?;
            let (row_locations, n_rows, mut row_index) = row?;

            for value_col_name in values {
                let value_col = pivot_df.column(value_col_name)?;

                use PivotAgg::*;
                let value_agg = unsafe {
                    match agg_fn {
                        Sum => value_col.agg_sum(&groups),
                        Min => value_col.agg_min(&groups),
                        Max => value_col.agg_max(&groups),
                        Last => value_col.agg_last(&groups),
                        First => value_col.agg_first(&groups),
                        Mean => value_col.agg_mean(&groups),
                        Median => value_col.agg_median(&groups),
                        Count => groups.group_count().into_series(),
                        Expr(ref expr) => {
                            let name = expr.root_name()?;
                            let mut value_col = value_col.clone();
                            value_col.rename(name);
                            let tmp_df = DataFrame::new_no_checks(vec![value_col]);
                            let mut aggregated = expr.evaluate(&tmp_df, &groups)?;
                            aggregated.rename(value_col_name);
                            aggregated
                        }
                    }
                };

                let headers = column_agg.unique_stable()?.cast(&DataType::Utf8)?;
                let headers = headers.utf8().unwrap();
                let n_cols = headers.len();

                let mut buf = vec![AnyValue::Null; n_rows * n_cols];

                let value_agg_phys = value_agg.to_physical_repr();

                for ((row_idx, col_idx), val) in row_locations
                    .iter()
                    .zip(&col_locations)
                    .zip(value_agg_phys.iter())
                {
                    // Safety:
                    // in bounds
                    unsafe {
                        let idx = *row_idx as usize + *col_idx as usize * n_rows;
                        debug_assert!(idx < buf.len());
                        *buf.get_unchecked_mut(idx) = val;
                    }
                }

                let headers_iter = headers.par_iter_indexed();

                let mut cols = (0..n_cols)
                    .into_par_iter()
                    .zip(headers_iter)
                    .map(|(i, opt_name)| {
                        let offset = i * n_rows;
                        let avs = &buf[offset..offset + n_rows];
                        let name = opt_name.unwrap_or("null");
                        let mut out = Series::new(name, avs);
                        finish_logical_type(&mut out, value_agg.dtype());
                        out
                    })
                    .collect::<Vec<_>>();

                if sort_columns {
                    cols.sort_unstable_by(|a, b| a.name().partial_cmp(b.name()).unwrap());
                }

                let cols = if count == 0 {
                    let mut final_cols = row_index.take().unwrap();
                    final_cols.extend(cols);
                    final_cols
                } else {
                    cols
                };
                count += 1;
                final_cols.extend_from_slice(&cols);
            }
        }
        Ok(())
    });
    out?;
    Ok(DataFrame::new_no_checks(final_cols))
}

// Takes a `DataFrame` that only consists of the column aggregates that are pivoted by
// the values in `columns`
fn finish_logical_type(column: &mut Series, dtype: &DataType) {
    *column = match dtype {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(Some(rev_map)) => {
            let ca = column.u32().unwrap();
            unsafe {
                CategoricalChunked::from_cats_and_rev_map_unchecked(ca.clone(), rev_map.clone())
            }
            .into_series()
        }
        _ => column.cast(dtype).unwrap(),
    };
}
