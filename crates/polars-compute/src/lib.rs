#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(feature = "simd", feature(avx512_target_feature))]
#![cfg_attr(
    all(feature = "simd", target_arch = "x86_64"),
    feature(stdarch_x86_avx512)
)]

use arrow::types::NativeType;

pub mod arithmetic;
pub mod arity;
pub mod bitwise;
#[cfg(feature = "approx_unique")]
pub mod cardinality;
#[cfg(feature = "cast")]
pub mod cast;
pub mod comparisons;
pub mod filter;
pub mod float_sum;
#[cfg(feature = "gather")]
pub mod gather;
pub mod horizontal_flatten;
#[cfg(feature = "approx_unique")]
pub mod hyperloglogplus;
pub mod if_then_else;
pub mod min_max;
pub mod propagate_dictionary;
pub mod size;
pub mod sum;
pub mod unique;
pub mod var_cov;

// Trait to enable the scalar blanket implementation.
pub trait NotSimdPrimitive: NativeType {}

#[cfg(not(feature = "simd"))]
impl<T: NativeType> NotSimdPrimitive for T {}

#[cfg(feature = "simd")]
impl NotSimdPrimitive for u128 {}
#[cfg(feature = "simd")]
impl NotSimdPrimitive for i128 {}

// Trait to allow blanket impl for all SIMD types when simd is enabled.
#[cfg(feature = "simd")]
mod _simd_primitive {
    use std::simd::SimdElement;
    pub trait SimdPrimitive: SimdElement {}
    impl SimdPrimitive for u8 {}
    impl SimdPrimitive for u16 {}
    impl SimdPrimitive for u32 {}
    impl SimdPrimitive for u64 {}
    impl SimdPrimitive for usize {}
    impl SimdPrimitive for i8 {}
    impl SimdPrimitive for i16 {}
    impl SimdPrimitive for i32 {}
    impl SimdPrimitive for i64 {}
    impl SimdPrimitive for isize {}
    impl SimdPrimitive for f32 {}
    impl SimdPrimitive for f64 {}
}

#[cfg(feature = "simd")]
pub use _simd_primitive::SimdPrimitive;
