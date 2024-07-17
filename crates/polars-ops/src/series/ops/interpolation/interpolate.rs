use std::ops::{Add, Div, Mul, Sub};

use arrow::array::PrimitiveArray;
use arrow::bitmap::MutableBitmap;
use polars_core::downcast_as_macro_arg_physical;
use polars_core::export::num::{NumCast, Zero};
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{linear_itp, nearest_itp};

fn near_interp<T>(low: T, high: T, steps: IdxSize, steps_n: T, out: &mut Vec<T>)
where
    T: Sub<Output = T>
        + Mul<Output = T>
        + Add<Output = T>
        + Div<Output = T>
        + NumCast
        + Copy
        + PartialOrd,
{
    let diff = high - low;
    for step_i in 1..steps {
        let step_i: T = NumCast::from(step_i).unwrap();
        let v = nearest_itp(low, step_i, diff, steps_n);
        out.push(v)
    }
}

#[inline]
fn signed_interp<T>(low: T, high: T, steps: IdxSize, steps_n: T, out: &mut Vec<T>)
where
    T: Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Div<Output = T> + NumCast + Copy,
{
    let slope = (high - low) / steps_n;
    for step_i in 1..steps {
        let step_i: T = NumCast::from(step_i).unwrap();
        let v = linear_itp(low, step_i, slope);
        out.push(v)
    }
}

fn interpolate_impl<T, I>(chunked_arr: &ChunkedArray<T>, interpolation_branch: I) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    I: Fn(T::Native, T::Native, IdxSize, T::Native, &mut Vec<T::Native>),
{
    // This implementation differs from pandas as that boundary None's are not removed.
    // This prevents a lot of errors due to expressions leading to different lengths.
    if !chunked_arr.has_nulls() || chunked_arr.null_count() == chunked_arr.len() {
        return chunked_arr.clone();
    }

    // We first find the first and last so that we can set the null buffer.
    let first = chunked_arr.first_non_null().unwrap();
    let last = chunked_arr.last_non_null().unwrap() + 1;

    // Fill out with `first` nulls.
    let mut out = Vec::with_capacity(chunked_arr.len());
    let mut iter = chunked_arr.iter().skip(first);
    for _ in 0..first {
        out.push(Zero::zero());
    }

    // The next element of `iter` is definitely `Some(Some(v))`, because we skipped the first
    // elements `first` and if all values were missing we'd have done an early return.
    let mut low = iter.next().unwrap().unwrap();
    out.push(low);
    while let Some(next) = iter.next() {
        if let Some(v) = next {
            out.push(v);
            low = v;
        } else {
            let mut steps = 1 as IdxSize;
            for next in iter.by_ref() {
                steps += 1;
                if let Some(high) = next {
                    let steps_n: T::Native = NumCast::from(steps).unwrap();
                    interpolation_branch(low, high, steps, steps_n, &mut out);
                    out.push(high);
                    low = high;
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
            unsafe { validity.set_unchecked(i, false) };
            out.push(Zero::zero())
        }

        let array = PrimitiveArray::new(
            T::get_dtype().to_arrow(CompatLevel::newest()),
            out.into(),
            Some(validity.into()),
        );
        ChunkedArray::with_chunk(chunked_arr.name(), array)
    } else {
        ChunkedArray::from_vec(chunked_arr.name(), out)
    }
}

fn interpolate_nearest(s: &Series) -> Series {
    match s.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(_, _) | DataType::Enum(_, _) => s.clone(),
        DataType::Binary => s.clone(),
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(_) => s.clone(),
        DataType::List(_) => s.clone(),
        _ => {
            let logical = s.dtype();
            let s = s.to_physical_repr();

            macro_rules! dispatch {
                ($ca:expr) => {{
                    interpolate_impl($ca, near_interp).into_series()
                }};
            }
            let out = downcast_as_macro_arg_physical!(s, dispatch);
            out.cast(logical).unwrap()
        },
    }
}

fn interpolate_linear(s: &Series) -> Series {
    match s.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(_, _) | DataType::Enum(_, _) => s.clone(),
        DataType::Binary => s.clone(),
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(_) => s.clone(),
        DataType::List(_) => s.clone(),
        _ => {
            let logical = s.dtype();

            let s = s.to_physical_repr();

            let out = if matches!(
                logical,
                DataType::Date | DataType::Datetime(_, _) | DataType::Duration(_) | DataType::Time
            ) {
                match s.dtype() {
                    // Datetime, Time, or Duration
                    DataType::Int64 => linear_interp_signed(s.i64().unwrap()),
                    // Date
                    DataType::Int32 => linear_interp_signed(s.i32().unwrap()),
                    _ => unreachable!(),
                }
            } else {
                match s.dtype() {
                    DataType::Float32 => linear_interp_signed(s.f32().unwrap()),
                    DataType::Float64 => linear_interp_signed(s.f64().unwrap()),
                    DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64 => {
                        linear_interp_signed(s.cast(&DataType::Float64).unwrap().f64().unwrap())
                    },
                    _ => s.as_ref().clone(),
                }
            };
            match logical {
                DataType::Date
                | DataType::Datetime(_, _)
                | DataType::Duration(_)
                | DataType::Time => out.cast(logical).unwrap(),
                _ => out,
            }
        },
    }
}

fn linear_interp_signed<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Series
where
    ChunkedArray<T>: IntoSeries,
{
    interpolate_impl(ca, signed_interp::<T::Native>).into_series()
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InterpolationMethod {
    Linear,
    Nearest,
}

pub fn interpolate(s: &Series, method: InterpolationMethod) -> Series {
    match method {
        InterpolationMethod::Linear => interpolate_linear(s),
        InterpolationMethod::Nearest => interpolate_nearest(s),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_interpolate() {
        let ca = UInt32Chunked::new("", &[Some(1), None, None, Some(4), Some(5)]);
        let out = interpolate(&ca.into_series(), InterpolationMethod::Linear);
        let out = out.f64().unwrap();
        assert_eq!(
            Vec::from(out),
            &[Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]
        );

        let ca = UInt32Chunked::new("", &[None, Some(1), None, None, Some(4), Some(5)]);
        let out = interpolate(&ca.into_series(), InterpolationMethod::Linear);
        let out = out.f64().unwrap();
        assert_eq!(
            Vec::from(out),
            &[None, Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]
        );

        let ca = UInt32Chunked::new("", &[None, Some(1), None, None, Some(4), Some(5), None]);
        let out = interpolate(&ca.into_series(), InterpolationMethod::Linear);
        let out = out.f64().unwrap();
        assert_eq!(
            Vec::from(out),
            &[
                None,
                Some(1.0),
                Some(2.0),
                Some(3.0),
                Some(4.0),
                Some(5.0),
                None
            ]
        );
        let ca = UInt32Chunked::new("", &[None, Some(1), None, None, Some(4), Some(5), None]);
        let out = interpolate(&ca.into_series(), InterpolationMethod::Nearest);
        let out = out.u32().unwrap();
        assert_eq!(
            Vec::from(out),
            &[None, Some(1), Some(1), Some(4), Some(4), Some(5), None]
        );
    }

    #[test]
    fn test_interpolate_decreasing_unsigned() {
        let ca = UInt32Chunked::new("", &[Some(4), None, None, Some(1)]);
        let out = interpolate(&ca.into_series(), InterpolationMethod::Linear);
        let out = out.f64().unwrap();
        assert_eq!(
            Vec::from(out),
            &[Some(4.0), Some(3.0), Some(2.0), Some(1.0)]
        )
    }

    #[test]
    fn test_interpolate2() {
        let ca = Float32Chunked::new(
            "",
            &[
                Some(4653f32),
                None,
                None,
                None,
                Some(4657f32),
                None,
                None,
                Some(4657f32),
                None,
                Some(4657f32),
                None,
                None,
                Some(4660f32),
            ],
        );
        let out = interpolate(&ca.into_series(), InterpolationMethod::Linear);
        let out = out.f32().unwrap();

        assert_eq!(
            Vec::from(out),
            &[
                Some(4653.0),
                Some(4654.0),
                Some(4655.0),
                Some(4656.0),
                Some(4657.0),
                Some(4657.0),
                Some(4657.0),
                Some(4657.0),
                Some(4657.0),
                Some(4657.0),
                Some(4658.0),
                Some(4659.0),
                Some(4660.0)
            ]
        );
    }
}
