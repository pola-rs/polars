use std::borrow::Cow;

use polars_error::{polars_bail, PolarsResult};
use polars_utils::pl_str::PlSmallStr;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

#[cfg(feature = "zip_with")]
use super::min_max_binary::min_max_binary_columns;
use super::{
    ChunkCompareEq, ChunkSet, Column, DataType, FillNullStrategy, IntoColumn, Scalar, Series,
    UInt32Chunked,
};
use crate::chunked_array::cast::CastOptions;
use crate::frame::NullStrategy;
use crate::POOL;

/// Aggregate the column horizontally to their min values.
///
/// All columns need to be the same length or a scalar.
#[cfg(feature = "zip_with")]
pub fn min_horizontal(columns: &[Column]) -> PolarsResult<Option<Column>> {
    let min_fn = |acc: &Column, s: &Column| min_max_binary_columns(acc, s, true);

    match columns.len() {
        0 => Ok(None),
        1 => Ok(Some(columns[0].clone())),
        2 => min_fn(&columns[0], &columns[1]).map(Some),
        _ => {
            // the try_reduce_with is a bit slower in parallelism,
            // but I don't think it matters here as we parallelize over columns, not over elements
            POOL.install(|| {
                columns
                    .par_iter()
                    .map(|s| Ok(Cow::Borrowed(s)))
                    .try_reduce_with(|l, r| min_fn(&l, &r).map(Cow::Owned))
                    // we can unwrap the option, because we are certain there is a column
                    // we started this operation on 3 columns
                    .unwrap()
                    .map(|cow| Some(cow.into_owned()))
            })
        },
    }
}

/// Aggregate the column horizontally to their max values.
///
/// All columns need to be the same length or a scalar.
#[cfg(feature = "zip_with")]
pub fn max_horizontal(columns: &[Column]) -> PolarsResult<Option<Column>> {
    let max_fn = |acc: &Column, s: &Column| min_max_binary_columns(acc, s, false);

    match columns.len() {
        0 => Ok(None),
        1 => Ok(Some(columns[0].clone())),
        2 => max_fn(&columns[0], &columns[1]).map(Some),
        _ => {
            // the try_reduce_with is a bit slower in parallelism,
            // but I don't think it matters here as we parallelize over columns, not over elements
            POOL.install(|| {
                columns
                    .par_iter()
                    .map(|s| Ok(Cow::Borrowed(s)))
                    .try_reduce_with(|l, r| max_fn(&l, &r).map(Cow::Owned))
                    // we can unwrap the option, because we are certain there is a column
                    // we started this operation on 3 columns
                    .unwrap()
                    .map(|cow| Some(cow.into_owned()))
            })
        },
    }
}

/// Sum all values horizontally across columns.
///
/// All columns need to be the same length or a scalar.
pub fn sum_horizontal(
    columns: &[Column],
    null_strategy: NullStrategy,
) -> PolarsResult<Option<Column>> {
    let apply_null_strategy = |s: Series, null_strategy: NullStrategy| -> PolarsResult<Series> {
        if let NullStrategy::Ignore = null_strategy {
            // if has nulls
            if s.null_count() > 0 {
                return s.fill_null(FillNullStrategy::Zero);
            }
        }
        Ok(s)
    };

    let sum_fn = |acc: Series, s: Series, null_strategy: NullStrategy| -> PolarsResult<Series> {
        let acc: Series = apply_null_strategy(acc, null_strategy)?;
        let s = apply_null_strategy(s, null_strategy)?;
        // This will do owned arithmetic and can be mutable
        std::ops::Add::add(acc, s)
    };

    // @scalar-opt
    let non_null_cols = columns
        .iter()
        .filter(|x| x.dtype() != &DataType::Null)
        .map(|c| c.as_materialized_series())
        .collect::<Vec<_>>();

    match non_null_cols.len() {
        0 => {
            if columns.is_empty() {
                Ok(None)
            } else {
                // all columns are null dtype, so result is null dtype
                Ok(Some(columns[0].clone()))
            }
        },
        1 => Ok(Some(
            apply_null_strategy(
                if non_null_cols[0].dtype() == &DataType::Boolean {
                    non_null_cols[0].cast(&DataType::UInt32)?
                } else {
                    non_null_cols[0].clone()
                },
                null_strategy,
            )?
            .into(),
        )),
        2 => sum_fn(
            non_null_cols[0].clone(),
            non_null_cols[1].clone(),
            null_strategy,
        )
        .map(Column::from)
        .map(Some),
        _ => {
            // the try_reduce_with is a bit slower in parallelism,
            // but I don't think it matters here as we parallelize over columns, not over elements
            let out = POOL.install(|| {
                non_null_cols
                    .into_par_iter()
                    .cloned()
                    .map(Ok)
                    .try_reduce_with(|l, r| sum_fn(l, r, null_strategy))
                    // We can unwrap because we started with at least 3 columns, so we always get a Some
                    .unwrap()
            });
            out.map(Column::from).map(Some)
        },
    }
}

/// Compute the mean of all values horizontally across columns.
///
/// All columns need to be the same length or a scalar.
pub fn mean_horizontal(
    columns: &[Column],
    null_strategy: NullStrategy,
) -> PolarsResult<Option<Column>> {
    let (numeric_columns, non_numeric_columns): (Vec<_>, Vec<_>) = columns.iter().partition(|s| {
        let dtype = s.dtype();
        dtype.is_numeric() || dtype.is_decimal() || dtype.is_bool() || dtype.is_null()
    });

    if !non_numeric_columns.is_empty() {
        let col = non_numeric_columns.first().cloned();
        polars_bail!(
            InvalidOperation: "'horizontal_mean' expects numeric expressions, found {:?} (dtype={})",
            col.unwrap().name(),
            col.unwrap().dtype(),
        );
    }
    let columns = numeric_columns.into_iter().cloned().collect::<Vec<_>>();
    match columns.len() {
        0 => Ok(None),
        1 => Ok(Some(match columns[0].dtype() {
            dt if dt != &DataType::Float32 && !dt.is_decimal() => {
                columns[0].cast(&DataType::Float64)?
            },
            _ => columns[0].clone(),
        })),
        _ => {
            let sum = || sum_horizontal(&columns, null_strategy);
            let null_count = || {
                columns
                    .par_iter()
                    .map(|c| {
                        c.is_null()
                            .into_column()
                            .cast_with_options(&DataType::UInt32, CastOptions::NonStrict)
                    })
                    .reduce_with(|l, r| {
                        let l = l?;
                        let r = r?;
                        let result = std::ops::Add::add(&l, &r)?;
                        PolarsResult::Ok(result)
                    })
                    // we can unwrap the option, because we are certain there is a column
                    // we started this operation on 2 columns
                    .unwrap()
            };

            let (sum, null_count) = POOL.install(|| rayon::join(sum, null_count));
            let sum = sum?;
            let null_count = null_count?;

            // value lengths: len - null_count
            let value_length: UInt32Chunked = (Column::new_scalar(
                PlSmallStr::EMPTY,
                Scalar::from(columns.len() as u32),
                null_count.len(),
            ) - null_count)?
                .u32()
                .unwrap()
                .clone();

            // make sure that we do not divide by zero
            // by replacing with None
            let value_length = value_length
                .set(&value_length.equal(0), None)?
                .into_column()
                .cast(&DataType::Float64)?;

            sum.map(|sum| std::ops::Div::div(&sum, &value_length))
                .transpose()
        },
    }
}
