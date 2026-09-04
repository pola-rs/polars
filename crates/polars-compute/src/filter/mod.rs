//! Contains operators to filter arrays such as [`filter`].
mod boolean;
mod dyn_array;
mod pl_array;
mod primitive;
mod scalar;

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
mod avx512;

pub use boolean::filter_boolean_kernel;
pub use dyn_array::filter_arrow_with_bitmap;
pub use pl_array::{filter, filter_with_bitmap};
