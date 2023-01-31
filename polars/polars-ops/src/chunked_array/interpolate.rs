use std::ops::{Add, Div, Mul, Sub};

use arrow::array::PrimitiveArray;
use arrow::bitmap::MutableBitmap;
use polars_core::downcast_as_macro_arg_physical;
use polars_core::export::num::{NumCast, Zero};
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

fn linear_itp<T>(low: T, step: T, diff: T, steps_n: T) -> T
where
    T: Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Div<Output = T>,
{
    low + step * diff / steps_n
}

fn nearest_itp<T>(low: T, step: T, diff: T, steps_n: T) -> T
where
    T: Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Div<Output = T> + PartialOrd + Copy,
{
    // 5 - 1 = 5 -> low
    // 5 - 2 = 3 -> low
    // 5 - 3 = 2 -> high
    if (steps_n - step) > step {
        low
    } else {
        low + diff
    }
}

fn near_interp<T>(low: T, high: T, steps: IdxSize, steps_n: T, av: &mut Vec<T>)
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
        av.push(v)
    }
}

#[inline]
fn signed_interp<T>(low: T, high: T, steps: IdxSize, steps_n: T, av: &mut Vec<T>)
where
    T: Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Div<Output = T> + NumCast + Copy,
{
    let diff = high - low;
    for step_i in 1..steps {
        let step_i: T = NumCast::from(step_i).unwrap();
        let v = linear_itp(low, step_i, diff, steps_n);
        av.push(v)
    }
}

#[inline]
fn unsigned_interp<T>(low: T, high: T, steps: IdxSize, steps_n: T, av: &mut Vec<T>)
where
    T: Sub<Output = T>
        + Mul<Output = T>
        + Add<Output = T>
        + Div<Output = T>
        + NumCast
        + PartialOrd
        + Copy,
{
    if high >= low {
        signed_interp::<T>(low, high, steps, steps_n, av)
    } else {
        let diff = low - high;
        for step_i in (1..steps).rev() {
            let step_i: T = NumCast::from(step_i).unwrap();
            let v = linear_itp(high, step_i, diff, steps_n);
            av.push(v)
        }
    }
}

fn interpolate_impl<T, I>(chunked_arr: &ChunkedArray<T>, interpolation_branch: I) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    I: Fn(T::Native, T::Native, IdxSize, T::Native, &mut Vec<T::Native>),
{
    // This implementation differs from pandas as that boundary None's are not removed
    // this prevents a lot of errors due to expressions leading to different lengths
    if !chunked_arr.has_validity() || chunked_arr.null_count() == chunked_arr.len() {
        return chunked_arr.clone();
    }

    // we first find the first and last so that we can set the null buffer
    let first = chunked_arr.first_non_null().unwrap();
    let last = chunked_arr.last_non_null().unwrap() + 1;

    // fill av with first
    let mut av = Vec::with_capacity(chunked_arr.len());
    let mut iter = chunked_arr.into_iter();
    for _ in 0..first {
        av.push(Zero::zero())
    }

    let mut low_val = None;
    loop {
        let next = iter.next();
        match next {
            Some(Some(v)) => {
                av.push(v);
                low_val = Some(v);
            }
            Some(None) => {
                match low_val {
                    // not a non-null value encountered yet
                    // so we skip
                    None => continue,
                    Some(low) => {
                        let mut steps = 1 as IdxSize;
                        loop {
                            steps += 1;
                            match iter.next() {
                                // end of iterator, break
                                None => break,
                                // another null
                                Some(None) => {}
                                Some(Some(high)) => {
                                    let steps_n: T::Native = NumCast::from(steps).unwrap();
                                    interpolation_branch(low, high, steps, steps_n, &mut av);
                                    av.push(high);
                                    low_val = Some(high);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            None => {
                break;
            }
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
            av.push(Zero::zero())
        }

        let array =
            PrimitiveArray::new(T::get_dtype().to_arrow(), av.into(), Some(validity.into()));
        unsafe { ChunkedArray::from_chunks(chunked_arr.name(), vec![Box::new(array)]) }
    } else {
        ChunkedArray::from_vec(chunked_arr.name(), av)
    }
}

fn interpolate_nearest(s: &Series) -> Series {
    match s.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(_) => s.clone(),
        #[cfg(feature = "dtype-binary")]
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
        }
    }
}

fn interpolate_linear(s: &Series) -> Series {
    match s.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(_) => s.clone(),
        #[cfg(feature = "dtype-binary")]
        DataType::Binary => s.clone(),
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(_) => s.clone(),
        DataType::List(_) => s.clone(),
        _ => {
            let logical = s.dtype();

            let s = s.to_physical_repr();
            let out = match s.dtype() {
                #[cfg(feature = "dtype-i8")]
                DataType::Int8 => linear_interp_signed(s.i8().unwrap()),
                #[cfg(feature = "dtype-i16")]
                DataType::Int16 => linear_interp_signed(s.i16().unwrap()),
                DataType::Int32 => linear_interp_signed(s.i32().unwrap()),
                DataType::Int64 => linear_interp_signed(s.i64().unwrap()),
                #[cfg(feature = "dtype-u8")]
                DataType::UInt8 => linear_interp_unsigned(s.u8().unwrap()),
                #[cfg(feature = "dtype-u16")]
                DataType::UInt16 => linear_interp_unsigned(s.u16().unwrap()),
                DataType::UInt32 => linear_interp_unsigned(s.u32().unwrap()),
                DataType::UInt64 => linear_interp_unsigned(s.u64().unwrap()),
                DataType::Float32 => linear_interp_unsigned(s.f32().unwrap()),
                DataType::Float64 => linear_interp_unsigned(s.f64().unwrap()),
                _ => s.as_ref().clone(),
            };
            out.cast(logical).unwrap()
        }
    }
}

fn linear_interp_unsigned<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> Series
where
    ChunkedArray<T>: IntoSeries,
{
    interpolate_impl(ca, unsigned_interp::<T::Native>).into_series()
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
        let out = out.u32().unwrap();
        assert_eq!(
            Vec::from(out),
            &[Some(1), Some(2), Some(3), Some(4), Some(5)]
        );

        let ca = UInt32Chunked::new("", &[None, Some(1), None, None, Some(4), Some(5)]);
        let out = interpolate(&ca.into_series(), InterpolationMethod::Linear);
        let out = out.u32().unwrap();
        assert_eq!(
            Vec::from(out),
            &[None, Some(1), Some(2), Some(3), Some(4), Some(5)]
        );

        let ca = UInt32Chunked::new("", &[None, Some(1), None, None, Some(4), Some(5), None]);
        let out = interpolate(&ca.into_series(), InterpolationMethod::Linear);
        let out = out.u32().unwrap();
        assert_eq!(
            Vec::from(out),
            &[None, Some(1), Some(2), Some(3), Some(4), Some(5), None]
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
        let out = out.u32().unwrap();
        assert_eq!(Vec::from(out), &[Some(4), Some(3), Some(2), Some(1)])
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
