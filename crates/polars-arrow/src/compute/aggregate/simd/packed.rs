use std::simd::prelude::{SimdFloat as _, SimdInt as _, SimdUint as _};

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
