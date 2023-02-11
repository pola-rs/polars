mod positioning;

use std::borrow::Cow;

use polars_core::export::rayon::prelude::*;
use polars_core::frame::groupby::expr::PhysicalAggExpr;
use polars_core::prelude::*;
use polars_core::utils::_split_offsets;
use polars_core::{downcast_as_macro_arg_physical, POOL};

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
        DataType::Float32 if matches!(s.dtype(), DataType::UInt32) => {
            let ca = s.u32().unwrap();
            ca._reinterpret_float().into_series()
        }
        DataType::Float64 if matches!(s.dtype(), DataType::UInt64) => {
            let ca = s.u64().unwrap();
            ca._reinterpret_float().into_series()
        }
        DataType::Int32 if matches!(s.dtype(), DataType::UInt32) => {
            let ca = s.u32().unwrap();
            ca.reinterpret_signed().into_series()
        }
        DataType::Int64 if matches!(s.dtype(), DataType::UInt64) => {
            let ca = s.u64().unwrap();
            ca.reinterpret_signed().into_series()
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
    separator: Option<&str>,
) -> PolarsResult<DataFrame>
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
        separator,
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
    separator: Option<&str>,
) -> PolarsResult<DataFrame>
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
        separator,
    )
}

#[allow(clippy::too_many_arguments)]
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
    // used as separator/delimiter in generated column names.
    separator: Option<&str>,
) -> PolarsResult<DataFrame> {
    let sep = separator.unwrap_or("_");
    if index.is_empty() {
        return Err(PolarsError::ComputeError(
            "index cannot be zero length".into(),
        ));
    }

    let mut final_cols = vec![];

    let mut count = 0;
    let out: PolarsResult<()> = POOL.install(|| {
        for column in columns {
            let mut groupby = index.to_vec();
            groupby.push(column.clone());

            let groups = pivot_df.groupby_stable(groupby)?.take_groups();

            // these are the row locations
            if !stable {
                println!("unstable pivot not yet supported, using stable pivot");
            };

            let (col, row) = POOL.join(
                || positioning::compute_col_idx(pivot_df, column, &groups),
                || positioning::compute_row_idx(pivot_df, index, &groups, count),
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
                let mut headers = headers.utf8().unwrap().clone();
                if values.len() > 1 {
                    headers = headers.apply(|v| Cow::from(format!("{value_col_name}{sep}{v}")))
                }

                let n_cols = headers.len();
                let value_agg_phys = value_agg.to_physical_repr();
                let logical_type = value_agg.dtype();

                debug_assert_eq!(row_locations.len(), col_locations.len());
                debug_assert_eq!(value_agg_phys.len(), row_locations.len());

                let mut cols = if value_agg_phys.dtype().is_numeric() {
                    macro_rules! dispatch {
                        ($ca:expr) => {{
                            positioning::position_aggregates_numeric(
                                n_rows,
                                n_cols,
                                &row_locations,
                                &col_locations,
                                $ca,
                                logical_type,
                                &headers,
                            )
                        }};
                    }
                    downcast_as_macro_arg_physical!(value_agg_phys, dispatch)
                } else {
                    positioning::position_aggregates(
                        n_rows,
                        n_cols,
                        &row_locations,
                        &col_locations,
                        &value_agg_phys,
                        logical_type,
                        &headers,
                    )
                };

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
