use std::borrow::Cow;

use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_core::series::arithmetic::coerce_lhs_rhs;
use polars_core::utils::dtypes_to_supertype;
use polars_core::{with_match_physical_numeric_polars_type, POOL};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

fn validate_column_lengths(cs: &[Column]) -> PolarsResult<()> {
    let mut length = 1;
    for c in cs {
        let len = c.len();
        if len != 1 && len != length {
            if length == 1 {
                length = len;
            } else {
                polars_bail!(ShapeMismatch: "cannot evaluate two Series of different lengths ({len} and {length})");
            }
        }
    }
    Ok(())
}

pub trait MinMaxHorizontal {
    /// Aggregate the column horizontally to their min values.
    fn min_horizontal(&self) -> PolarsResult<Option<Column>>;
    /// Aggregate the column horizontally to their max values.
    fn max_horizontal(&self) -> PolarsResult<Option<Column>>;
}

impl MinMaxHorizontal for DataFrame {
    fn min_horizontal(&self) -> PolarsResult<Option<Column>> {
        min_horizontal(self.get_columns())
    }
    fn max_horizontal(&self) -> PolarsResult<Option<Column>> {
        max_horizontal(self.get_columns())
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum NullStrategy {
    Ignore,
    Propagate,
}

pub trait SumMeanHorizontal {
    /// Sum all values horizontally across columns.
    fn sum_horizontal(&self, null_strategy: NullStrategy) -> PolarsResult<Option<Column>>;

    /// Compute the mean of all numeric values horizontally across columns.
    fn mean_horizontal(&self, null_strategy: NullStrategy) -> PolarsResult<Option<Column>>;
}

impl SumMeanHorizontal for DataFrame {
    fn sum_horizontal(&self, null_strategy: NullStrategy) -> PolarsResult<Option<Column>> {
        sum_horizontal(self.get_columns(), null_strategy)
    }
    fn mean_horizontal(&self, null_strategy: NullStrategy) -> PolarsResult<Option<Column>> {
        mean_horizontal(self.get_columns(), null_strategy)
    }
}

fn min_binary<T>(left: &ChunkedArray<T>, right: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: PartialOrd,
{
    let op = |l: T::Native, r: T::Native| {
        if l < r {
            l
        } else {
            r
        }
    };
    arity::binary_elementwise_values(left, right, op)
}

fn max_binary<T>(left: &ChunkedArray<T>, right: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: PartialOrd,
{
    let op = |l: T::Native, r: T::Native| {
        if l > r {
            l
        } else {
            r
        }
    };
    arity::binary_elementwise_values(left, right, op)
}

fn min_max_binary_columns(left: &Column, right: &Column, min: bool) -> PolarsResult<Column> {
    if left.dtype().to_physical().is_primitive_numeric()
        && left.null_count() == 0
        && right.null_count() == 0
        && left.len() == right.len()
    {
        match (left, right) {
            (Column::Series(left), Column::Series(right)) => {
                let (lhs, rhs) = coerce_lhs_rhs(left, right)?;
                let logical = lhs.dtype();
                let lhs = lhs.to_physical_repr();
                let rhs = rhs.to_physical_repr();

                with_match_physical_numeric_polars_type!(lhs.dtype(), |$T| {
                    let a: &ChunkedArray<$T> = lhs.as_ref().as_ref().as_ref();
                    let b: &ChunkedArray<$T> = rhs.as_ref().as_ref().as_ref();

                    unsafe {
                        if min {
                            min_binary(a, b).into_series().from_physical_unchecked(logical)
                        } else {
                            max_binary(a, b).into_series().from_physical_unchecked(logical)
                        }
                    }
                })
                .map(Column::from)
            },
            _ => {
                let mask = if min {
                    left.lt(right)?
                } else {
                    left.gt(right)?
                };

                left.zip_with(&mask, right)
            },
        }
    } else {
        let mask = if min {
            left.lt(right)? & left.is_not_null() | right.is_null()
        } else {
            left.gt(right)? & left.is_not_null() | right.is_null()
        };
        left.zip_with(&mask, right)
    }
}

pub fn max_horizontal(columns: &[Column]) -> PolarsResult<Option<Column>> {
    validate_column_lengths(columns)?;

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

pub fn min_horizontal(columns: &[Column]) -> PolarsResult<Option<Column>> {
    validate_column_lengths(columns)?;

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

pub fn sum_horizontal(
    columns: &[Column],
    null_strategy: NullStrategy,
) -> PolarsResult<Option<Column>> {
    validate_column_lengths(columns)?;
    let ignore_nulls = null_strategy == NullStrategy::Ignore;

    let apply_null_strategy = |s: Series| -> PolarsResult<Series> {
        if ignore_nulls && s.null_count() > 0 {
            s.fill_null(FillNullStrategy::Zero)
        } else {
            Ok(s)
        }
    };

    let sum_fn = |acc: Series, s: Series| -> PolarsResult<Series> {
        let acc: Series = apply_null_strategy(acc)?;
        let s = apply_null_strategy(s)?;
        // This will do owned arithmetic and can be mutable
        std::ops::Add::add(acc, s)
    };

    // @scalar-opt
    let non_null_cols = columns
        .iter()
        .filter(|x| x.dtype() != &DataType::Null)
        .map(|c| c.as_materialized_series())
        .collect::<Vec<_>>();

    // If we have any null columns and null strategy is not `Ignore`, we can return immediately.
    if !ignore_nulls && non_null_cols.len() < columns.len() {
        // We must determine the correct return dtype.
        let return_dtype = match dtypes_to_supertype(non_null_cols.iter().map(|c| c.dtype()))? {
            DataType::Boolean => IDX_DTYPE,
            dt => dt,
        };
        return Ok(Some(Column::full_null(
            columns[0].name().clone(),
            columns[0].len(),
            &return_dtype,
        )));
    }

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
            apply_null_strategy(if non_null_cols[0].dtype() == &DataType::Boolean {
                non_null_cols[0].cast(&IDX_DTYPE)?
            } else {
                non_null_cols[0].clone()
            })?
            .into(),
        )),
        2 => sum_fn(non_null_cols[0].clone(), non_null_cols[1].clone())
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
                    .try_reduce_with(sum_fn)
                    // We can unwrap because we started with at least 3 columns, so we always get a Some
                    .unwrap()
            });
            out.map(Column::from).map(Some)
        },
    }
}

pub fn mean_horizontal(
    columns: &[Column],
    null_strategy: NullStrategy,
) -> PolarsResult<Option<Column>> {
    validate_column_lengths(columns)?;

    let (numeric_columns, non_numeric_columns): (Vec<_>, Vec<_>) = columns.iter().partition(|s| {
        let dtype = s.dtype();
        dtype.is_primitive_numeric() || dtype.is_decimal() || dtype.is_bool() || dtype.is_null()
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
    let num_rows = columns.len();
    match num_rows {
        0 => Ok(None),
        1 => Ok(Some(match columns[0].dtype() {
            dt if dt != &DataType::Float32 && !dt.is_decimal() => {
                columns[0].cast(&DataType::Float64)?
            },
            _ => columns[0].clone(),
        })),
        _ => {
            let sum = || sum_horizontal(columns.as_slice(), null_strategy);
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
                Scalar::from(num_rows as u32),
                null_count.len(),
            ) - null_count)?
                .u32()
                .unwrap()
                .clone();

            // make sure that we do not divide by zero
            // by replacing with None
            let dt = if sum
                .as_ref()
                .is_some_and(|s| s.dtype() == &DataType::Float32)
            {
                &DataType::Float32
            } else {
                &DataType::Float64
            };
            let value_length = value_length
                .set(&value_length.equal(0), None)?
                .into_column()
                .cast(dt)?;

            sum.map(|sum| std::ops::Div::div(&sum, &value_length))
                .transpose()
        },
    }
}

pub fn coalesce_columns(s: &[Column]) -> PolarsResult<Column> {
    // TODO! this can be faster if we have more than two inputs.
    polars_ensure!(!s.is_empty(), NoData: "cannot coalesce empty list");
    let mut out = s[0].clone();
    for s in s {
        if !out.null_count() == 0 {
            return Ok(out);
        } else {
            let mask = out.is_not_null();
            out = out
                .as_materialized_series()
                .zip_with_same_type(&mask, s.as_materialized_series())?
                .into();
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_horizontal_agg() {
        let a = Column::new("a".into(), [1, 2, 6]);
        let b = Column::new("b".into(), [Some(1), None, None]);
        let c = Column::new("c".into(), [Some(4), None, Some(3)]);

        let df = DataFrame::new(vec![a, b, c]).unwrap();
        assert_eq!(
            Vec::from(
                df.mean_horizontal(NullStrategy::Ignore)
                    .unwrap()
                    .unwrap()
                    .f64()
                    .unwrap()
            ),
            &[Some(2.0), Some(2.0), Some(4.5)]
        );
        assert_eq!(
            Vec::from(
                df.sum_horizontal(NullStrategy::Ignore)
                    .unwrap()
                    .unwrap()
                    .i32()
                    .unwrap()
            ),
            &[Some(6), Some(2), Some(9)]
        );
        assert_eq!(
            Vec::from(df.min_horizontal().unwrap().unwrap().i32().unwrap()),
            &[Some(1), Some(2), Some(3)]
        );
        assert_eq!(
            Vec::from(df.max_horizontal().unwrap().unwrap().i32().unwrap()),
            &[Some(4), Some(2), Some(6)]
        );
    }
}
