use arrow::array::Array;
use polars_utils::min_max::MinMax;

// Low-level min/max kernel.
pub trait MinMaxKernel: Sized + Array {
    type Scalar: MinMax;

    fn min_ignore_nan(&self) -> Option<Self::Scalar>;
    fn max_ignore_nan(&self) -> Option<Self::Scalar>;
    fn min_propagate_nan(&self) -> Option<Self::Scalar>;
    fn max_propagate_nan(&self) -> Option<Self::Scalar>;
}


// Trait to enable the scalar blanket implementation.
trait NotSimdPrimitive {}

#[cfg(not(feature = "simd"))]
impl<T> NotSimdPrimitive for T {}

#[cfg(feature = "simd")]
impl NotSimdPrimitive for u128 {}
#[cfg(feature = "simd")]
impl NotSimdPrimitive for i128 {}

mod scalar;

#[cfg(feature = "simd")]
mod simd;