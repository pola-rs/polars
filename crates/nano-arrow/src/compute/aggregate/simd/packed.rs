use std::simd::{SimdFloat as _, SimdInt as _, SimdOrd as _, SimdUint as _};

use super::super::min_max::SimdOrd;
use super::super::sum::Sum;
use crate::types::simd::*;

macro_rules! simd_sum {
    ($simd:tt, $type:ty, $sum:tt) => {
        impl Sum<$type> for $simd {
            #[inline]
            fn simd_sum(self) -> $type {
                self.$sum()
            }
        }
    };
}

simd_sum!(f32x16, f32, reduce_sum);
simd_sum!(f64x8, f64, reduce_sum);
simd_sum!(u8x64, u8, reduce_sum);
simd_sum!(u16x32, u16, reduce_sum);
simd_sum!(u32x16, u32, reduce_sum);
simd_sum!(u64x8, u64, reduce_sum);
simd_sum!(i8x64, i8, reduce_sum);
simd_sum!(i16x32, i16, reduce_sum);
simd_sum!(i32x16, i32, reduce_sum);
simd_sum!(i64x8, i64, reduce_sum);

macro_rules! simd_ord_int {
    ($simd:tt, $type:ty) => {
        impl SimdOrd<$type> for $simd {
            const MIN: $type = <$type>::MIN;
            const MAX: $type = <$type>::MAX;

            #[inline]
            fn max_element(self) -> $type {
                self.reduce_max()
            }

            #[inline]
            fn min_element(self) -> $type {
                self.reduce_min()
            }

            #[inline]
            fn max_lane(self, x: Self) -> Self {
                self.simd_max(x)
            }

            #[inline]
            fn min_lane(self, x: Self) -> Self {
                self.simd_min(x)
            }

            #[inline]
            fn new_min() -> Self {
                Self::splat(Self::MAX)
            }

            #[inline]
            fn new_max() -> Self {
                Self::splat(Self::MIN)
            }
        }
    };
}

macro_rules! simd_ord_float {
    ($simd:tt, $type:ty) => {
        impl SimdOrd<$type> for $simd {
            const MIN: $type = <$type>::NAN;
            const MAX: $type = <$type>::NAN;

            #[inline]
            fn max_element(self) -> $type {
                self.reduce_max()
            }

            #[inline]
            fn min_element(self) -> $type {
                self.reduce_min()
            }

            #[inline]
            fn max_lane(self, x: Self) -> Self {
                self.simd_max(x)
            }

            #[inline]
            fn min_lane(self, x: Self) -> Self {
                self.simd_min(x)
            }

            #[inline]
            fn new_min() -> Self {
                Self::splat(<$type>::NAN)
            }

            #[inline]
            fn new_max() -> Self {
                Self::splat(<$type>::NAN)
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
