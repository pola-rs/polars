use super::*;

native_simd!(u8x64, u8, 64, u64);
native_simd!(u16x32, u16, 32, u32);
native_simd!(u32x16, u32, 16, u16);
native_simd!(u64x8, u64, 8, u8);
native_simd!(i8x64, i8, 64, u64);
native_simd!(i16x32, i16, 32, u32);
native_simd!(i32x16, i32, 16, u16);
native_simd!(i64x8, i64, 8, u8);
native_simd!(f16x32, f16, 32, u32);
native_simd!(f32x16, f32, 16, u16);
native_simd!(f64x8, f64, 8, u8);
