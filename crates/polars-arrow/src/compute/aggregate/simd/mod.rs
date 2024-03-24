use std::ops::Add;

use super::Sum;
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

// #[cfg(not(feature = "simd"))]
// pub(super) use simd_add;

simd_add!(i128x8, i128, 8, add);

#[cfg(not(feature = "simd"))]
mod native;

#[cfg(feature = "simd")]
mod packed;
