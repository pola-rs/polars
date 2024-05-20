use std::ops::{Add, Div, Mul, Sub};

use arrow::array::PrimitiveArray;
use arrow::bitmap::MutableBitmap;
use bytemuck::allocation::zeroed_vec;
use polars_core::export::num::{NumCast, Zero};
use polars_core::prelude::*;

use super::linear_itp;

#[inline]
fn signed_interp_by_sorted<T, F>(y_start: T, y_end: T, x: &[F], out: &mut Vec<T>)
where
    T: Sub<Output = T>
        + Mul<Output = T>
        + Add<Output = T>
        + Div<Output = T>
        + NumCast
        + Copy
        + Zero,
    F: Sub<Output = F> + NumCast + Copy,
{
    let range_y = y_end - y_start;
    let range_x = NumCast::from(x[x.len() - 1] - x[0]).unwrap();
    let slope = range_y / range_x;
    let x_start = x[0];
    for x_i in &x[1..x.len() - 1] {
        let x_delta = NumCast::from(*x_i - x_start).unwrap();
        let v = linear_itp(y_start, x_delta, slope);
        out.push(v)
    }
}

#[inline]
fn signed_interp_by<T, F>(
    y_start: T,
    y_end: T,
    x: &[F],
    out: &mut [T],
    sorting_indices: &[IdxSize],
    low_idx: usize,
) where
    T: Sub<Output = T>
        + Mul<Output = T>
        + Add<Output = T>
        + Div<Output = T>
        + NumCast
        + Copy
        + Zero,
    F: Sub<Output = F> + NumCast + Copy,
{
    let range_y = y_end - y_start;
    let range_x = NumCast::from(x[x.len() - 1] - x[0]).unwrap();
    let slope = range_y / range_x;
    let x_start = x[0];
    for (offset, x_i) in (x[1..x.len() - 1]).iter().enumerate() {
        let x_delta = NumCast::from(*x_i - x_start).unwrap();
        let v = linear_itp(y_start, x_delta, slope);
        let out_idx = unsafe { sorting_indices.get_unchecked(low_idx + offset + 1) };
        unsafe { *out.get_unchecked_mut(*out_idx as usize) = v };
    }
}

fn interpolate_impl_by_sorted<T, F, I>(
    chunked_arr: &ChunkedArray<T>,
    by: &ChunkedArray<F>,
    interpolation_branch: I,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsNumericType,
    F: PolarsIntegerType,
    I: Fn(T::Native, T::Native, &[F::Native], &mut Vec<T::Native>),
{
    // This implementation differs from pandas as that boundary None's are not removed.
    // This prevents a lot of errors due to expressions leading to different lengths.
    if !chunked_arr.has_validity() || chunked_arr.null_count() == chunked_arr.len() {
        return Ok(chunked_arr.clone());
    }

    // At the moment, the algorithm assumes no missing values in the `by` column.
    let by = by.rechunk();
    let by_values = by.cont_slice().map_err(|_| {
        polars_err!(
            InvalidOperation:
            "null values in `by` column are not yet supported in 'interpolate_by' expression"
        )
    })?;

    // We first find the first and last so that we can set the null buffer.
    let first = chunked_arr.first_non_null().unwrap();
    let last = chunked_arr.last_non_null().unwrap() + 1;

    // Fill out with first.
    let mut out = Vec::with_capacity(chunked_arr.len());
    let mut iter = chunked_arr.into_iter().enumerate().skip(first);
    for _ in 0..first {
        out.push(Zero::zero())
    }

    let mut low_val = None;
    let mut low_idx = None;
    loop {
        let next = iter.next();
        match next {
            Some((idx, Some(v))) => {
                out.push(v);
                low_val = Some(v);
                low_idx = Some(idx);
            },
            Some((_idx, None)) => {
                loop {
                    match iter.next() {
                        None => break,         // End of iterator, break.
                        Some((_, None)) => {}, // Another null.
                        Some((high_idx, Some(high))) => {
                            interpolation_branch(
                                low_val.expect("We started iterating at `first`"),
                                high,
                                &by_values[low_idx.unwrap()..=high_idx],
                                &mut out,
                            );
                            out.push(high);
                            low_val = Some(high);
                            low_idx = Some(high_idx);
                            break;
                        },
                    }
                }
            },
            None => {
                break;
            },
        }
    }
    if first != 0 || last != chunked_arr.len() {
        let mut validity = MutableBitmap::with_capacity(chunked_arr.len());
        validity.extend_constant(chunked_arr.len(), true);

        for i in 0..first {
            validity.set(i, false);
        }

        for i in last..chunked_arr.len() {
            validity.set(i, false);
            out.push(Zero::zero())
        }

        let array = PrimitiveArray::new(
            T::get_dtype().to_arrow(true),
            out.into(),
            Some(validity.into()),
        );
        Ok(ChunkedArray::with_chunk(chunked_arr.name(), array))
    } else {
        Ok(ChunkedArray::from_vec(chunked_arr.name(), out))
    }
}

// Sort on behalf of user
fn interpolate_impl_by<T, F, I>(
    ca: &ChunkedArray<T>,
    by: &ChunkedArray<F>,
    interpolation_branch: I,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsNumericType,
    F: PolarsIntegerType,
    I: Fn(T::Native, T::Native, &[F::Native], &mut [T::Native], &[IdxSize], usize),
{
    // This implementation differs from pandas as that boundary None's are not removed.
    // This prevents a lot of errors due to expressions leading to different lengths.
    if !ca.has_validity() || ca.null_count() == ca.len() {
        return Ok(ca.clone());
    }

    let sorting_indices = by.arg_sort(Default::default());
    let sorting_indices = sorting_indices
        .cont_slice()
        .expect("arg sort produces single chunk");
    let by_sorted = unsafe { by.take_unchecked(sorting_indices) };
    let ca_sorted = unsafe { ca.take_unchecked(sorting_indices) };

    // At the moment, the algorithm assumes no missing values in the `by` column.
    let by_sorted = by_sorted.rechunk();
    let by_sorted_values = by_sorted.cont_slice().map_err(|_| {
        polars_err!(
            InvalidOperation:
            "null values in `by` column are not yet supported in 'interpolate_by' expression"
        )
    })?;

    // We first find the first and last so that we can set the null buffer.
    let first = ca_sorted.first_non_null().unwrap();
    let last = ca_sorted.last_non_null().unwrap() + 1;

    let mut out = zeroed_vec(ca_sorted.len());
    let mut iter = ca_sorted.into_iter().enumerate().skip(first);

    let mut low_val = None;
    let mut low_idx = None;
    loop {
        let next = iter.next();
        match next {
            Some((idx, Some(v))) => {
                let out_idx = unsafe { sorting_indices.get_unchecked(idx) };
                unsafe { *out.get_unchecked_mut(*out_idx as usize) = v };
                low_val = Some(v);
                low_idx = Some(idx);
            },
            Some((_idx, None)) => {
                loop {
                    match iter.next() {
                        None => break,         // End of iterator, break.
                        Some((_, None)) => {}, // Another null.
                        Some((high_idx, Some(high))) => {
                            interpolation_branch(
                                low_val.expect("We started iterating at `first`."),
                                high,
                                &by_sorted_values[low_idx.unwrap()..=high_idx],
                                &mut out,
                                sorting_indices,
                                low_idx.unwrap(),
                            );
                            unsafe {
                                let out_idx = sorting_indices.get_unchecked(high_idx);
                                *out.get_unchecked_mut(*out_idx as usize) = high;
                            }
                            low_val = Some(high);
                            low_idx = Some(high_idx);
                            break;
                        },
                    }
                }
            },
            None => {
                break;
            },
        }
    }
    if first != 0 || last != ca_sorted.len() {
        let mut validity = MutableBitmap::with_capacity(ca_sorted.len());
        validity.extend_constant(ca_sorted.len(), true);

        for i in 0..first {
            let out_idx = unsafe { sorting_indices.get_unchecked(i) };
            validity.set(*out_idx as usize, false);
        }

        for i in last..ca_sorted.len() {
            let out_idx = unsafe { sorting_indices.get_unchecked(i) };
            validity.set(*out_idx as usize, false);
        }

        let array = PrimitiveArray::new(
            T::get_dtype().to_arrow(true),
            out.into(),
            Some(validity.into()),
        );
        Ok(ChunkedArray::with_chunk(ca_sorted.name(), array))
    } else {
        Ok(ChunkedArray::from_vec(ca_sorted.name(), out))
    }
}

pub fn interpolate_by(s: &Series, by: &Series, assume_sorted: bool) -> PolarsResult<Series> {
    polars_ensure!(s.len() == by.len(), InvalidOperation: "`by` column must be the same length as Series ({}), got {}", s.len(), by.len());

    fn func<T, F>(
        ca: &ChunkedArray<T>,
        by: &ChunkedArray<F>,
        assume_sorted: bool,
    ) -> PolarsResult<Series>
    where
        T: PolarsNumericType,
        F: PolarsIntegerType,
        ChunkedArray<T>: IntoSeries,
    {
        if assume_sorted {
            interpolate_impl_by_sorted(ca, by, signed_interp_by_sorted).map(|x| x.into_series())
        } else {
            interpolate_impl_by(ca, by, signed_interp_by).map(|x| x.into_series())
        }
    }

    match (s.dtype(), by.dtype()) {
        (DataType::Float64, DataType::Int64) => {
            func(s.f64().unwrap(), by.i64().unwrap(), assume_sorted)
        },
        (DataType::Float64, DataType::Int32) => {
            func(s.f64().unwrap(), by.i32().unwrap(), assume_sorted)
        },
        (DataType::Float64, DataType::UInt64) => {
            func(s.f64().unwrap(), by.u64().unwrap(), assume_sorted)
        },
        (DataType::Float64, DataType::UInt32) => {
            func(s.f64().unwrap(), by.u32().unwrap(), assume_sorted)
        },
        (DataType::Float32, DataType::Int64) => {
            func(s.f32().unwrap(), by.i64().unwrap(), assume_sorted)
        },
        (DataType::Float32, DataType::Int32) => {
            func(s.f32().unwrap(), by.i32().unwrap(), assume_sorted)
        },
        (DataType::Float32, DataType::UInt64) => {
            func(s.f32().unwrap(), by.u64().unwrap(), assume_sorted)
        },
        (DataType::Float32, DataType::UInt32) => {
            func(s.f32().unwrap(), by.u32().unwrap(), assume_sorted)
        },
        #[cfg(feature = "dtype-date")]
        (_, DataType::Date) => {
            interpolate_by(s, &by.cast(&DataType::Int32).unwrap(), assume_sorted)
        },
        #[cfg(feature = "dtype-datetime")]
        (_, DataType::Datetime(_, _)) => {
            interpolate_by(s, &by.cast(&DataType::Int64).unwrap(), assume_sorted)
        },
        (DataType::UInt64 | DataType::UInt32 | DataType::Int64 | DataType::Int32, _) => {
            interpolate_by(&s.cast(&DataType::Float64).unwrap(), by, assume_sorted)
        },
        _ => {
            polars_bail!(InvalidOperation: "expected series to be Float64, Float32, \
                Int64, Int32, UInt64, UInt32, and `by` to be Date, Datetime, Int64, Int32, \
                UInt64, or UInt32")
        },
    }
}
