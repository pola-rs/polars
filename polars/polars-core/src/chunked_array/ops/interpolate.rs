use crate::prelude::*;
use num::FromPrimitive;
use std::ops::{Add, Div, Mul, Sub};

fn linear_itp<T>(low: T, step: T, diff: T, steps_n: T) -> T
where
    T: Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Div<Output = T>,
{
    low + step * diff / steps_n
}

impl<T> Interpolate for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Add<Output = T::Native>
        + Div<Output = T::Native>
        + FromPrimitive,
{
    fn interpolate(&self) -> Self {
        if self.null_count() == 0 || self.null_count() == self.len() {
            return self.clone();
        }
        let mut av = AlignedVec::with_capacity(self.len());
        let mut iter = self.into_iter();

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
                                        let diff = high - low;
                                        let steps_n = T::Native::from_u32(steps).unwrap();
                                        for step_i in 1..steps {
                                            let step_i = T::Native::from_u32(step_i).unwrap();
                                            let v = linear_itp(low, step_i, diff, steps_n);
                                            av.push(v)
                                        }
                                        av.push(high);
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
        Self::new_from_aligned_vec(self.name(), av)
    }
}

macro_rules! interpolate {
    ($ca:ty) => {
        impl Interpolate for $ca {
            fn interpolate(&self) -> Self {
                if self.null_count() == 0 || self.null_count() == self.len() {
                    return self.clone();
                }
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
                        last = i;
                        break;
                    }
                }
                self.slice(first as i64, last)
            }
        }
    };
}

interpolate!(Utf8Chunked);
interpolate!(ListChunked);
interpolate!(BooleanChunked);
interpolate!(CategoricalChunked);

impl<T: PolarsObject> Interpolate for ObjectChunked<T> {
    fn interpolate(&self) -> Self {
        if self.null_count() == 0 || self.null_count() == self.len() {
            return self.clone();
        }
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
                last = i;
                break;
            }
        }
        self.slice(first as i64, last)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_interpolate() {
        let ca = UInt32Chunked::new_from_opt_slice("", &[Some(1), None, None, Some(4), Some(5)]);
        let out = ca.interpolate();
        assert_eq!(
            Vec::from(&out),
            &[Some(1), Some(2), Some(3), Some(4), Some(5)]
        );

        let ca = Utf8Chunked::new_from_opt_slice(
            "",
            &[None, Some("foo"), None, None, Some("bar"), None, None],
        );

        let out = ca.interpolate();
        assert_eq!(Vec::from(&out), &[Some("foo"), None, None, Some("bar")]);
        dbg!(out);
    }
}
