use std::ops::Add;

use super::super::sum::Sum;
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
