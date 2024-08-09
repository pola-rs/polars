use std::ops::{Add, Div, Mul, Sub};

use arrow::array::PrimitiveArray;
use arrow::bitmap::MutableBitmap;
use bytemuck::allocation::zeroed_vec;
use polars_core::export::num::{NumCast, Zero};
use polars_core::prelude::*;
use polars_utils::slice::SliceAble;

use super::linear_itp;

/// # Safety
/// - `x` must be non-empty.
#[inline]
unsafe fn signed_interp_by_sorted<T, F>(y_start: T, y_end: T, x: &[F], out: &mut Vec<T>)
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
    let x_start;
    let range_x;
    let iter;
    unsafe {
        x_start = x.get_unchecked(0);
        range_x = NumCast::from(*x.get_unchecked(x.len() - 1) - *x_start).unwrap();
        iter = x.slice_unchecked(1..x.len() - 1).iter();
    }
    let slope = range_y / range_x;
    for x_i in iter {
        let x_delta = NumCast::from(*x_i - *x_start).unwrap();
        let v = linear_itp(y_start, x_delta, slope);
        out.push(v)
    }
}

/// # Safety
/// - `x` must be non-empty.
/// - `sorting_indices` must be the same size as `x`
#[inline]
unsafe fn signed_interp_by<T, F>(
    y_start: T,
    y_end: T,
    x: &[F],
    out: &mut [T],
    sorting_indices: &[IdxSize],
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
    let x_start;
    let range_x;
    let iter;
    unsafe {
        x_start = x.get_unchecked(0);
        range_x = NumCast::from(*x.get_unchecked(x.len() - 1) - *x_start).unwrap();
        iter = x.slice_unchecked(1..x.len() - 1).iter();
    }
    let slope = range_y / range_x;
    for (idx, x_i) in iter.enumerate() {
        let x_delta = NumCast::from(*x_i - *x_start).unwrap();
        let v = linear_itp(y_start, x_delta, slope);
        unsafe {
            let out_idx = sorting_indices.get_unchecked(idx + 1);
            *out.get_unchecked_mut(*out_idx as usize) = v;
        }
    }
}

fn interpolate_impl_by_sorted<T, F, I>(
    chunked_arr: &ChunkedArray<T>,
    by: &ChunkedArray<F>,
    interpolation_branch: I,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsNumericType,
    F: PolarsNumericType,
    I: Fn(T::Native, T::Native, &[F::Native], &mut Vec<T::Native>),
{
    // This implementation differs from pandas as that boundary None's are not removed.
    // This prevents a lot of errors due to expressions leading to different lengths.
    if !chunked_arr.has_nulls() || chunked_arr.null_count() == chunked_arr.len() {
        return Ok(chunked_arr.clone());
    }

    polars_ensure!(by.null_count() == 0, InvalidOperation: "null values in `by` column are not yet supported in 'interpolate_by' expression");
    let by = by.rechunk();
    let by_values = by.cont_slice().unwrap();

    // We first find the first and last so that we can set the null buffer.
    let first = chunked_arr.first_non_null().unwrap();
    let last = chunked_arr.last_non_null().unwrap() + 1;

    // Fill out with `first` nulls.
    let mut out = Vec::with_capacity(chunked_arr.len());
    let mut iter = chunked_arr.iter().enumerate().skip(first);
    for _ in 0..first {
        out.push(Zero::zero());
    }

    // The next element of `iter` is definitely `Some(idx, Some(v))`, because we skipped the first
    // `first` elements and if all values were missing we'd have done an early return.
    let (mut low_idx, opt_low) = iter.next().unwrap();
    let mut low = opt_low.unwrap();
    out.push(low);
    while let Some((idx, next)) = iter.next() {
        if let Some(v) = next {
            out.push(v);
            low = v;
            low_idx = idx;
        } else {
            for (high_idx, next) in iter.by_ref() {
                if let Some(high) = next {
                    // SAFETY: we are in bounds, and `x` is non-empty.
                    unsafe {
                        let x = &by_values.slice_unchecked(low_idx..high_idx + 1);
                        interpolation_branch(low, high, x, &mut out);
                    }
                    out.push(high);
                    low = high;
                    low_idx = high_idx;
                    break;
                }
            }
        }
    }
    if first != 0 || last != chunked_arr.len() {
        let mut validity = MutableBitmap::with_capacity(chunked_arr.len());
        validity.extend_constant(chunked_arr.len(), true);

        for i in 0..first {
            unsafe { validity.set_unchecked(i, false) };
        }

        for i in last..chunked_arr.len() {
            unsafe { validity.set_unchecked(i, false) }
            out.push(Zero::zero());
        }

        let array = PrimitiveArray::new(
            T::get_dtype().to_arrow(CompatLevel::newest()),
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
    F: PolarsNumericType,
    I: Fn(T::Native, T::Native, &[F::Native], &mut [T::Native], &[IdxSize]),
{
    // This implementation differs from pandas as that boundary None's are not removed.
    // This prevents a lot of errors due to expressions leading to different lengths.
    if !ca.has_nulls() || ca.null_count() == ca.len() {
        return Ok(ca.clone());
    }

    polars_ensure!(by.null_count() == 0, InvalidOperation: "null values in `by` column are not yet supported in 'interpolate_by' expression");
    let sorting_indices = by.arg_sort(Default::default());
    let sorting_indices = sorting_indices
        .cont_slice()
        .expect("arg sort produces single chunk");
    let by_sorted = unsafe { by.take_unchecked(sorting_indices) };
    let ca_sorted = unsafe { ca.take_unchecked(sorting_indices) };
    let by_sorted_values = by_sorted
        .cont_slice()
        .expect("We already checked for nulls, and `take_unchecked` produces single chunk");

    // We first find the first and last so that we can set the null buffer.
    let first = ca_sorted.first_non_null().unwrap();
    let last = ca_sorted.last_non_null().unwrap() + 1;

    let mut out = zeroed_vec(ca_sorted.len());
    let mut iter = ca_sorted.iter().enumerate().skip(first);

    // The next element of `iter` is definitely `Some(idx, Some(v))`, because we skipped the first
    // `first` elements and if all values were missing we'd have done an early return.
    let (mut low_idx, opt_low) = iter.next().unwrap();
    let mut low = opt_low.unwrap();
    unsafe {
        let out_idx = sorting_indices.get_unchecked(low_idx);
        *out.get_unchecked_mut(*out_idx as usize) = low;
    }
    while let Some((idx, next)) = iter.next() {
        if let Some(v) = next {
            unsafe {
                let out_idx = sorting_indices.get_unchecked(idx);
                *out.get_unchecked_mut(*out_idx as usize) = v;
            }
            low = v;
            low_idx = idx;
        } else {
            for (high_idx, next) in iter.by_ref() {
                if let Some(high) = next {
                    // SAFETY: we are in bounds, and the slices are the same length (and non-empty).
                    unsafe {
                        interpolation_branch(
                            low,
                            high,
                            by_sorted_values.slice_unchecked(low_idx..high_idx + 1),
                            &mut out,
                            sorting_indices.slice_unchecked(low_idx..high_idx + 1),
                        );
                        let out_idx = sorting_indices.get_unchecked(high_idx);
                        *out.get_unchecked_mut(*out_idx as usize) = high;
                    }
                    low = high;
                    low_idx = high_idx;
                    break;
                }
            }
        }
    }
    if first != 0 || last != ca_sorted.len() {
        let mut validity = MutableBitmap::with_capacity(ca_sorted.len());
        validity.extend_constant(ca_sorted.len(), true);

        for i in 0..first {
            unsafe {
                let out_idx = sorting_indices.get_unchecked(i);
                validity.set_unchecked(*out_idx as usize, false);
            }
        }

        for i in last..ca_sorted.len() {
            unsafe {
                let out_idx = sorting_indices.get_unchecked(i);
                validity.set_unchecked(*out_idx as usize, false);
            }
        }

        let array = PrimitiveArray::new(
            T::get_dtype().to_arrow(CompatLevel::newest()),
            out.into(),
            Some(validity.into()),
        );
        Ok(ChunkedArray::with_chunk(ca_sorted.name(), array))
    } else {
        Ok(ChunkedArray::from_vec(ca_sorted.name(), out))
    }
}

pub fn interpolate_by(s: &Series, by: &Series, by_is_sorted: bool) -> PolarsResult<Series> {
    polars_ensure!(s.len() == by.len(), InvalidOperation: "`by` column must be the same length as Series ({}), got {}", s.len(), by.len());

    fn func<T, F>(
        ca: &ChunkedArray<T>,
        by: &ChunkedArray<F>,
        is_sorted: bool,
    ) -> PolarsResult<Series>
    where
        T: PolarsNumericType,
        F: PolarsNumericType,
        ChunkedArray<T>: IntoSeries,
    {
        if is_sorted {
            interpolate_impl_by_sorted(ca, by, |y_start, y_end, x, out| unsafe {
                signed_interp_by_sorted(y_start, y_end, x, out)
            })
            .map(|x| x.into_series())
        } else {
            interpolate_impl_by(ca, by, |y_start, y_end, x, out, sorting_indices| unsafe {
                signed_interp_by(y_start, y_end, x, out, sorting_indices)
            })
            .map(|x| x.into_series())
        }
    }

    match (s.dtype(), by.dtype()) {
        (DataType::Float64, DataType::Float64) => {
            func(s.f64().unwrap(), by.f64().unwrap(), by_is_sorted)
        },
        (DataType::Float64, DataType::Float32) => {
            func(s.f64().unwrap(), by.f32().unwrap(), by_is_sorted)
        },
        (DataType::Float32, DataType::Float64) => {
            func(s.f32().unwrap(), by.f64().unwrap(), by_is_sorted)
        },
        (DataType::Float32, DataType::Float32) => {
            func(s.f32().unwrap(), by.f32().unwrap(), by_is_sorted)
        },
        (DataType::Float64, DataType::Int64) => {
            func(s.f64().unwrap(), by.i64().unwrap(), by_is_sorted)
        },
        (DataType::Float64, DataType::Int32) => {
            func(s.f64().unwrap(), by.i32().unwrap(), by_is_sorted)
        },
        (DataType::Float64, DataType::UInt64) => {
            func(s.f64().unwrap(), by.u64().unwrap(), by_is_sorted)
        },
        (DataType::Float64, DataType::UInt32) => {
            func(s.f64().unwrap(), by.u32().unwrap(), by_is_sorted)
        },
        (DataType::Float32, DataType::Int64) => {
            func(s.f32().unwrap(), by.i64().unwrap(), by_is_sorted)
        },
        (DataType::Float32, DataType::Int32) => {
            func(s.f32().unwrap(), by.i32().unwrap(), by_is_sorted)
        },
        (DataType::Float32, DataType::UInt64) => {
            func(s.f32().unwrap(), by.u64().unwrap(), by_is_sorted)
        },
        (DataType::Float32, DataType::UInt32) => {
            func(s.f32().unwrap(), by.u32().unwrap(), by_is_sorted)
        },
        #[cfg(feature = "dtype-date")]
        (_, DataType::Date) => interpolate_by(s, &by.cast(&DataType::Int32).unwrap(), by_is_sorted),
        #[cfg(feature = "dtype-datetime")]
        (_, DataType::Datetime(_, _)) => {
            interpolate_by(s, &by.cast(&DataType::Int64).unwrap(), by_is_sorted)
        },
        (DataType::UInt64 | DataType::UInt32 | DataType::Int64 | DataType::Int32, _) => {
            interpolate_by(&s.cast(&DataType::Float64).unwrap(), by, by_is_sorted)
        },
        _ => {
            polars_bail!(InvalidOperation: "expected series to be Float64, Float32, \
                Int64, Int32, UInt64, UInt32, and `by` to be Date, Datetime, Int64, Int32, \
                UInt64, UInt32, Float32 or Float64")
        },
    }
}
