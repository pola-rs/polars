#![cfg_attr(feature = "simd", feature(portable_simd))]

use arrow::types::NativeType;

pub mod arithmetic;
pub mod arity;
pub mod binview_index_map;
pub mod bitwise;
pub mod boolean;
#[cfg(feature = "approx_unique")]
pub mod cardinality;
#[cfg(feature = "cast")]
pub mod cast;
pub mod comparisons;
#[cfg(feature = "dtype-decimal")]
pub mod decimal;
pub mod ewm;
pub mod filter;
#[cfg(feature = "cast")]
pub mod find_validity_mismatch;
pub mod float_sum;
#[cfg(feature = "gather")]
pub mod gather;
pub mod horizontal_flatten;
#[cfg(feature = "approx_unique")]
pub mod hyperloglogplus;
pub mod if_then_else;
pub mod min_max;
pub mod moment;
pub mod nan;
pub mod propagate_dictionary;
pub mod propagate_nulls;
pub mod rolling;
pub mod size;
pub mod sum;
pub mod trim_lists_to_normalized_offsets;
pub mod unique;

// Trait to enable the scalar blanket implementation.
pub trait NotSimdPrimitive: NativeType {}

#[cfg(not(feature = "simd"))]
impl<T: NativeType> NotSimdPrimitive for T {}

#[cfg(feature = "simd")]
impl NotSimdPrimitive for u128 {}
#[cfg(feature = "simd")]
impl NotSimdPrimitive for i128 {}
#[cfg(feature = "simd")]
impl NotSimdPrimitive for pf16 {}

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
#[cfg(feature = "simd")]
use polars_utils::float16::pf16;
