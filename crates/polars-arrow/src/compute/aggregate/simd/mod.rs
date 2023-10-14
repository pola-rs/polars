use std::ops::Add;

use super::{SimdOrd, Sum};
use crate::types::simd::{i128x8, NativeSimd};

macro_rules! simd_add {
    ($simd:tt, $type:ty, $lanes:expr, $add:tt) => {
        impl std::ops::AddAssign for $simd {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                for i in 0..$lanes {
                    self[i] = <$type>::$add(self[i], rhs[i]);
                }
            }
        }

        impl std::ops::Add for $simd {
            type Output = Self;

            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                let mut result = Self::default();
                for i in 0..$lanes {
                    result[i] = <$type>::$add(self[i], rhs[i]);
                }
                result
            }
        }

        impl Sum<$type> for $simd {
            #[inline]
            fn simd_sum(self) -> $type {
                let mut reduced = <$type>::default();
                (0..<$simd>::LANES).for_each(|i| {
                    reduced += self[i];
                });
                reduced
            }
        }
    };
}

macro_rules! simd_ord_int {
    ($simd:tt, $type:ty) => {
        impl SimdOrd<$type> for $simd {
            const MIN: $type = <$type>::MIN;
            const MAX: $type = <$type>::MAX;

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

pub(super) use {simd_add, simd_ord_int};

simd_add!(i128x8, i128, 8, add);
simd_ord_int!(i128x8, i128);

#[cfg(not(feature = "simd"))]
mod native;
#[cfg(not(feature = "simd"))]
pub use native::*;
#[cfg(feature = "simd")]
mod packed;
#[cfg(feature = "simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "simd")))]
pub use packed::*;
