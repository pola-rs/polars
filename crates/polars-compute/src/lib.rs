#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(feature = "simd", feature(avx512_target_feature))]
#![cfg_attr(
    all(feature = "simd", target_arch = "x86_64"),
    feature(stdarch_x86_avx512)
)]

use arrow::types::NativeType;

pub mod arithmetic;
pub mod arity;
pub mod comparisons;
pub mod filter;
pub mod float_sum;
pub mod if_then_else;
pub mod min_max;
pub mod size;
pub mod unique;

// Trait to enable the scalar blanket implementation.
pub trait NotSimdPrimitive: NativeType {}

#[cfg(not(feature = "simd"))]
impl<T: NativeType> NotSimdPrimitive for T {}

#[cfg(feature = "simd")]
impl NotSimdPrimitive for u128 {}
#[cfg(feature = "simd")]
impl NotSimdPrimitive for i128 {}
