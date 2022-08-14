use std::ops::{Add, Div, Mul, Sub};

use arrow::bitmap::MutableBitmap;
use num::{FromPrimitive, Zero};

use crate::prelude::*;

fn linear_itp<T>(low: T, step: T, diff: T, steps_n: T) -> T
where
    T: Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Div<Output = T>,
{
    low + step * diff / steps_n
}

#[inline]
fn signed_interp<T: PolarsNumericType>(
    low: T::Native,
    high: T::Native,
    steps: u32,
    steps_n: T::Native,
    av: &mut Vec<T::Native>,
) {
    let diff = high - low;
    for step_i in 1..steps {
        let step_i = T::Native::from_u32(step_i).unwrap();
        let v = linear_itp(low, step_i, diff, steps_n);
        av.push(v)
    }
}

#[inline]
fn unsigned_interp<T: PolarsNumericType>(
    low: T::Native,
    high: T::Native,
    steps: u32,
    steps_n: T::Native,
    av: &mut Vec<T::Native>,
) {
    if high >= low {
        signed_interp::<T>(low, high, steps, steps_n, av)
    } else {
        let diff = low - high;
        for step_i in (1..steps).rev() {
            let step_i = T::Native::from_u32(step_i).unwrap();
            let v = linear_itp(high, step_i, diff, steps_n);
            av.push(v)
        }
    }
}

impl<T: PolarsNumericType> ChunkedArray<T> {
    fn interpolate_impl<I>(&self, interpolation_branch: I) -> Self
    where
        I: Fn(T::Native, T::Native, u32, T::Native, &mut Vec<T::Native>),
    {
        // This implementation differs from pandas as that boundary None's are not removed
        // this prevents a lot of errors due to expressions leading to different lengths
        if !self.has_validity() || self.null_count() == self.len() {
            return self.clone();
        }

        // we first find the first and last so that we can set the null buffer
        let mut first = 0;
        let mut last = self.len();
        // find first non None
        for i in 0..self.len() {
            // Safety: we just bound checked
            if unsafe { self.get_unchecked(i).is_some() } {
                first = i;
                break;
            }
        }

        // find last non None
        for i in (0..self.len()).rev() {
            if unsafe { self.get_unchecked(i).is_some() } {
                last = i + 1;
                break;
            }
        }

        // fill av with first
        let mut av = Vec::with_capacity(self.len());
        let mut iter = self.into_iter();
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
                            let mut steps = 1u32;
                            loop {
                                steps += 1;
                                match iter.next() {
                                    // end of iterator, break
                                    None => break,
                                    // another null
                                    Some(None) => {}
                                    Some(Some(high)) => {
                                        let steps_n = T::Native::from_u32(steps).unwrap();
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
        if first != 0 || last != self.len() {
            let mut validity = MutableBitmap::with_capacity(self.len());
            validity.extend_constant(self.len(), true);

            for i in 0..first {
                validity.set(i, false);
            }

            for i in last..self.len() {
                validity.set(i, false);
                av.push(Zero::zero())
            }

            let array = PrimitiveArray::from_data(
                T::get_dtype().to_arrow(),
                av.into(),
                Some(validity.into()),
            );
            Self::from_chunks(self.name(), vec![Box::new(array)])
        } else {
            Self::from_vec(self.name(), av)
        }
    }
}

macro_rules! impl_interpolate {
    ($type:ident, $interpolation_branch:ident) => {
        impl Interpolate for ChunkedArray<$type> {
            fn interpolate(&self) -> Self {
                self.interpolate_impl($interpolation_branch::<$type>)
            }
        }
    };
}

#[cfg(feature = "dtype-u8")]
impl_interpolate!(UInt8Type, unsigned_interp);
#[cfg(feature = "dtype-u16")]
impl_interpolate!(UInt16Type, unsigned_interp);
impl_interpolate!(UInt32Type, unsigned_interp);
impl_interpolate!(UInt64Type, unsigned_interp);

#[cfg(feature = "dtype-i8")]
impl_interpolate!(Int8Type, signed_interp);
#[cfg(feature = "dtype-i16")]
impl_interpolate!(Int16Type, signed_interp);
impl_interpolate!(Int32Type, signed_interp);
impl_interpolate!(Int64Type, signed_interp);
impl_interpolate!(Float32Type, signed_interp);
impl_interpolate!(Float64Type, signed_interp);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_interpolate() {
        let ca = UInt32Chunked::new("", &[Some(1), None, None, Some(4), Some(5)]);
        let out = ca.interpolate();
        assert_eq!(
            Vec::from(&out),
            &[Some(1), Some(2), Some(3), Some(4), Some(5)]
        );

        let ca = UInt32Chunked::new("", &[None, Some(1), None, None, Some(4), Some(5)]);
        let out = ca.interpolate();
        assert_eq!(
            Vec::from(&out),
            &[None, Some(1), Some(2), Some(3), Some(4), Some(5)]
        );

        let ca = UInt32Chunked::new("", &[None, Some(1), None, None, Some(4), Some(5), None]);
        let out = ca.interpolate();
        assert_eq!(
            Vec::from(&out),
            &[None, Some(1), Some(2), Some(3), Some(4), Some(5), None]
        );
    }

    #[test]
    fn test_interpolate_decreasing_unsigned() {
        let ca = UInt32Chunked::new("", &[Some(4), None, None, Some(1)]);
        let out = ca.interpolate();
        assert_eq!(Vec::from(&out), &[Some(4), Some(3), Some(2), Some(1)])
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
        let out = ca.interpolate();

        assert_eq!(
            Vec::from(&out),
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
