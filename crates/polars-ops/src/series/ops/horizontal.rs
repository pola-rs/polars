use std::borrow::Cow;

use polars_arrow::array::PrimitiveArray;
use polars_arrow::compute::utils::combine_validities_or;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::series::arithmetic::coerce_lhs_rhs;
use polars_core::utils::{CustomIterTools, align_chunks_binary, dtypes_to_supertype};
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::broadcast::broadcast_len;
use polars_utils::min_max::MinMax;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

pub trait MinMaxHorizontal {
    /// Aggregate the column horizontally to their min values.
    fn min_horizontal(&self) -> PolarsResult<Option<Column>>;
    /// Aggregate the column horizontally to their max values.
    fn max_horizontal(&self) -> PolarsResult<Option<Column>>;
}

impl MinMaxHorizontal for DataFrame {
    fn min_horizontal(&self) -> PolarsResult<Option<Column>> {
        min_horizontal(self.columns())
    }
    fn max_horizontal(&self) -> PolarsResult<Option<Column>> {
        max_horizontal(self.columns())
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
        sum_horizontal(self.columns(), null_strategy)
    }
    fn mean_horizontal(&self, null_strategy: NullStrategy) -> PolarsResult<Option<Column>> {
        mean_horizontal(self.columns(), null_strategy)
    }
}

/// Replace the nulls in `s` with the values of `other` at the same position.
fn fill_null_from<'a>(s: &'a Series, other: &Series) -> PolarsResult<Cow<'a, Series>> {
    if s.has_nulls() {
        s.zip_with_same_type(&s.is_not_null(), other)
            .map(Cow::Owned)
    } else {
        Ok(Cow::Borrowed(s))
    }
}

/// Elementwise `op` where a null on one side takes the value of the other side.
fn binary_ignore_nulls<T, F>(
    left: &ChunkedArray<T>,
    right: &ChunkedArray<T>,
    op: F,
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    F: Fn(T::Native, T::Native) -> T::Native,
{
    let (left, right) = align_chunks_binary(left, right);
    let chunks = left
        .downcast_iter()
        .zip(right.downcast_iter())
        .map(|(l, r)| {
            let lv = l.values().iter().copied();
            let rv = r.values().iter().copied();
            let values: Vec<T::Native> = match (l.validity(), r.validity()) {
                (Some(lm), Some(rm)) => lv
                    .zip(rv)
                    .zip(lm.iter().zip(rm.iter()))
                    .map(|((a, b), (a_valid, b_valid))| {
                        op(if a_valid { a } else { b }, if b_valid { b } else { a })
                    })
                    .collect_trusted(),
                (Some(lm), None) => lv
                    .zip(rv)
                    .zip(lm.iter())
                    .map(|((a, b), a_valid)| op(if a_valid { a } else { b }, b))
                    .collect_trusted(),
                (None, Some(rm)) => lv
                    .zip(rv)
                    .zip(rm.iter())
                    .map(|((a, b), b_valid)| op(a, if b_valid { b } else { a }))
                    .collect_trusted(),
                (None, None) => lv.zip(rv).map(|(a, b)| op(a, b)).collect_trusted(),
            };
            let validity = combine_validities_or(l.validity(), r.validity());
            PrimitiveArray::from_vec(values).with_validity(validity)
        });
    ChunkedArray::from_chunk_iter(left.name().clone(), chunks)
}

fn min_max_binary_columns(left: &Column, right: &Column, min: bool) -> PolarsResult<Column> {
    if left.dtype().to_physical().is_primitive_numeric()
        && right.dtype().to_physical().is_primitive_numeric()
    {
        let height = broadcast_len([left, right])?;
        let left_s = left.as_materialized_series_maintain_scalar();
        let right_s = right.as_materialized_series_maintain_scalar();
        let (lhs, rhs) = coerce_lhs_rhs(&left_s, &right_s)?;
        let logical = lhs.dtype();

        let lhs = lhs.to_physical_repr();
        let rhs = rhs.to_physical_repr();
        let has_nulls = lhs.has_nulls() || rhs.has_nulls();
        let same_len = lhs.len() == rhs.len();
        let (lhs, rhs) = if has_nulls && !same_len {
            (fill_null_from(&lhs, &rhs)?, fill_null_from(&rhs, &lhs)?)
        } else {
            (lhs, rhs)
        };

        with_match_physical_numeric_polars_type!(lhs.dtype(), |$T| {
            let a: &ChunkedArray<$T> = lhs.as_ref().as_ref().as_ref();
            let b: &ChunkedArray<$T> = rhs.as_ref().as_ref().as_ref();

            let out: ChunkedArray<$T> = match (has_nulls && same_len, min) {
                (true, true) => binary_ignore_nulls(a, b, MinMax::min_ignore_nan),
                (true, false) => binary_ignore_nulls(a, b, MinMax::max_ignore_nan),
                (false, true) => {
                    arity::broadcast_binary_elementwise_values(a, b, MinMax::min_ignore_nan)
                },
                (false, false) => {
                    arity::broadcast_binary_elementwise_values(a, b, MinMax::max_ignore_nan)
                },
            };
            unsafe { out.into_series().from_physical_unchecked(logical) }
        })
        .and_then(|s| s.broadcast_owned_to(height))
        .map(Column::from)
    } else {
        let mut mask = if min {
            left.lt(right)?
        } else {
            left.gt(right)?
        };
        if left.has_nulls() {
            mask = mask & left.is_not_null();
        }
        if right.has_nulls() {
            mask = mask | right.is_null();
        }
        left.zip_with(&mask, right)
    }
}

pub fn max_horizontal(columns: &[Column]) -> PolarsResult<Option<Column>> {
    broadcast_len(columns.iter()).context("max_horizontal")?;

    let max_fn = |acc: &Column, s: &Column| min_max_binary_columns(acc, s, false);

    match columns.len() {
        0 => Ok(None),
        1 => Ok(Some(columns[0].clone())),
        2 => max_fn(&columns[0], &columns[1]).map(Some),
        _ => {
            // the try_reduce_with is a bit slower in parallelism,
            // but I don't think it matters here as we parallelize over columns, not over elements
            RAYON.install(|| {
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
    broadcast_len(columns.iter()).context("min_horizontal")?;

    let min_fn = |acc: &Column, s: &Column| min_max_binary_columns(acc, s, true);

    match columns.len() {
        0 => Ok(None),
        1 => Ok(Some(columns[0].clone())),
        2 => min_fn(&columns[0], &columns[1]).map(Some),
        _ => {
            // the try_reduce_with is a bit slower in parallelism,
            // but I don't think it matters here as we parallelize over columns, not over elements
            RAYON.install(|| {
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
    broadcast_len(columns.iter()).context("sum_horizontal")?;
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
        .collect::<Vec<_>>();
    let out_dtype = sum_output_dtype(&dtypes_to_supertype(
        non_null_cols.iter().map(|c| c.dtype()),
    )?);

    // If we have any null columns and null strategy is not `Ignore`, we can return immediately.
    if !ignore_nulls && non_null_cols.len() < columns.len() {
        return Ok(Some(Column::full_null(
            columns[0].name().clone(),
            columns[0].len(),
            &out_dtype,
        )));
    }

    let non_null_cols = non_null_cols
        .into_iter()
        .map(|c| c.as_materialized_series().cast(&out_dtype))
        .collect::<PolarsResult<Vec<_>>>()?;

    match non_null_cols.len() {
        0 => {
            if columns.is_empty() {
                Ok(None)
            } else {
                // all columns are null dtype, so result is null dtype
                Ok(Some(columns[0].clone()))
            }
        },
        1 => {
            let [s] = <[Series; 1]>::try_from(non_null_cols).unwrap();
            Ok(Some(apply_null_strategy(s)?.into()))
        },
        2 => {
            let [a, b] = <[Series; 2]>::try_from(non_null_cols).unwrap();
            sum_fn(a, b).map(Column::from).map(Some)
        },
        _ => {
            // the try_reduce_with is a bit slower in parallelism,
            // but I don't think it matters here as we parallelize over columns, not over elements
            let out = RAYON.install(|| {
                non_null_cols
                    .into_par_iter()
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
    broadcast_len(columns.iter()).context("mean_horizontal")?;

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
            dt if !matches!(dt, DataType::Float16 | DataType::Float32) && !dt.is_decimal() => {
                columns[0].cast(&DataType::Float64)?
            },
            _ => columns[0].clone(),
        })),
        _ => {
            let fast_path = columns.iter().all(|c| {
                c.null_count() == 0
                    && c.dtype().is_primitive_numeric()
                    && !matches!(c.dtype(), DataType::Float16)
            });
            if fast_path {
                let sum = sum_horizontal(columns.as_slice(), null_strategy)?.unwrap();
                let n = num_rows as f64;
                let out = if sum.dtype() == &DataType::Float32 {
                    let n = n as f32;
                    sum.f32()?.apply_values(|v| v / n).into_column()
                } else {
                    let sum = sum.cast(&DataType::Float64)?;
                    sum.f64()?.apply_values(|v| v / n).into_column()
                };
                return Ok(Some(out));
            }

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

            let (sum, null_count) = RAYON.install(|| rayon::join(sum, null_count));
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
            let dt = sum
                .as_ref()
                .map(Column::dtype)
                .filter(|dt| matches!(dt, DataType::Float16 | DataType::Float32))
                .unwrap_or(&DataType::Float64);
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
    // Each iteration recomputes the null mask and allocates a new `out`; a single
    // pass over all inputs would be faster for many inputs with many nulls.
    polars_ensure!(!s.is_empty(), NoData: "cannot coalesce empty list");
    let len = broadcast_len(s)?;
    let mut out = s[0].clone();
    for next in &s[1..] {
        if !out.has_nulls() {
            break;
        }
        if out.null_count() == out.len() {
            out = next.clone().with_name(out.name().clone());
            continue;
        }
        let mask = out.is_not_null();
        out = out
            .as_materialized_series()
            .zip_with_same_type(&mask, next.as_materialized_series())?
            .into();
    }
    // The early exits skip the broadcast `zip_with` would have done.
    if out.len() != len {
        out = out.new_from_index(0, len);
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

        let df = DataFrame::new_infer_height(vec![a, b, c]).unwrap();
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
