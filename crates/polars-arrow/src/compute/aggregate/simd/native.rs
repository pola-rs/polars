use std::ops::Add;

use super::super::min_max::SimdOrd;
use super::super::sum::Sum;
use super::{simd_add, simd_ord_int};
use crate::types::simd::*;

simd_add!(u8x64, u8, 64, wrapping_add);
simd_add!(u16x32, u16, 32, wrapping_add);
simd_add!(u32x16, u32, 16, wrapping_add);
simd_add!(u64x8, u64, 8, wrapping_add);
simd_add!(i8x64, i8, 64, wrapping_add);
simd_add!(i16x32, i16, 32, wrapping_add);
simd_add!(i32x16, i32, 16, wrapping_add);
simd_add!(i64x8, i64, 8, wrapping_add);
simd_add!(f32x16, f32, 16, add);
simd_add!(f64x8, f64, 8, add);

macro_rules! simd_ord_float {
    ($simd:tt, $type:ty) => {
        impl SimdOrd<$type> for $simd {
            const MIN: $type = <$type>::NAN;
            const MAX: $type = <$type>::NAN;

            #[inline]
            fn max_element(self) -> $type {
                self.0.iter().copied().fold(Self::MIN, <$type>::max)
            }

            #[inline]
            fn min_element(self) -> $type {
                self.0.iter().copied().fold(Self::MAX, <$type>::min)
            }

            #[inline]
            fn max_lane(self, x: Self) -> Self {
                let mut result = <$simd>::default();
                result
                    .0
                    .iter_mut()
                    .zip(self.0.iter())
                    .zip(x.0.iter())
                    .for_each(|((a, b), c)| *a = (*b).max(*c));
                result
            }

            #[inline]
            fn min_lane(self, x: Self) -> Self {
                let mut result = <$simd>::default();
                result
                    .0
                    .iter_mut()
                    .zip(self.0.iter())
                    .zip(x.0.iter())
                    .for_each(|((a, b), c)| *a = (*b).min(*c));
                result
            }

            #[inline]
            fn new_min() -> Self {
                Self([Self::MAX; <$simd>::LANES])
            }

            #[inline]
            fn new_max() -> Self {
                Self([Self::MIN; <$simd>::LANES])
            }
        }
    };
}

simd_ord_int!(u8x64, u8);
simd_ord_int!(u16x32, u16);
simd_ord_int!(u32x16, u32);
simd_ord_int!(u64x8, u64);
simd_ord_int!(i8x64, i8);
simd_ord_int!(i16x32, i16);
simd_ord_int!(i32x16, i32);
simd_ord_int!(i64x8, i64);
simd_ord_float!(f32x16, f32);
simd_ord_float!(f64x8, f64);
