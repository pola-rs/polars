use polars_utils::min_max::MinMax;

pub use self::dyn_array::{
    dyn_array_max_ignore_nan, dyn_array_max_propagate_nan, dyn_array_min_ignore_nan,
    dyn_array_min_max_propagate_nan, dyn_array_min_propagate_nan,
};

/// Low-level min/max kernel.
pub trait MinMaxKernel {
    type Scalar<'a>: MinMax
    where
        Self: 'a;

    fn min_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>>;
    fn max_ignore_nan_kernel(&self) -> Option<Self::Scalar<'_>>;
    fn min_max_ignore_nan_kernel(&self) -> Option<(Self::Scalar<'_>, Self::Scalar<'_>)> {
        Some((self.min_ignore_nan_kernel()?, self.max_ignore_nan_kernel()?))
    }

    fn min_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>>;
    fn max_propagate_nan_kernel(&self) -> Option<Self::Scalar<'_>>;
    fn min_max_propagate_nan_kernel(&self) -> Option<(Self::Scalar<'_>, Self::Scalar<'_>)> {
        Some((
            self.min_propagate_nan_kernel()?,
            self.max_propagate_nan_kernel()?,
        ))
    }
}

// Trait to enable the scalar blanket implementation.
trait NotSimdPrimitive {}

#[cfg(not(feature = "simd"))]
impl<T> NotSimdPrimitive for T {}

#[cfg(feature = "simd")]
impl NotSimdPrimitive for u128 {}
#[cfg(feature = "simd")]
impl NotSimdPrimitive for i128 {}

mod dyn_array;
mod scalar;

#[cfg(feature = "simd")]
mod simd;
