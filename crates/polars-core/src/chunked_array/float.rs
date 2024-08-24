use arrow::legacy::kernels::float::*;
use arrow::legacy::kernels::set::set_at_nulls;
use num_traits::Float;
use polars_utils::total_ord::{canonical_f32, canonical_f64};

use crate::prelude::arity::unary_elementwise_values;
use crate::prelude::*;

impl<T> ChunkedArray<T>
where
    T: PolarsFloatType,
    T::Native: Float,
{
    pub fn is_nan(&self) -> BooleanChunked {
        self.apply_kernel_cast(&is_nan::<T::Native>)
    }
    pub fn is_not_nan(&self) -> BooleanChunked {
        self.apply_kernel_cast(&is_not_nan::<T::Native>)
    }
    pub fn is_finite(&self) -> BooleanChunked {
        self.apply_kernel_cast(&is_finite)
    }
    pub fn is_infinite(&self) -> BooleanChunked {
        self.apply_kernel_cast(&is_infinite)
    }

    #[must_use]
    /// Convert missing values to `NaN` values.
    pub fn none_to_nan(&self) -> Self {
        let chunks = self
            .downcast_iter()
            .map(|arr| set_at_nulls(arr, T::Native::nan()));
        ChunkedArray::from_chunk_iter(self.name(), chunks)
    }
}

pub trait Canonical {
    fn canonical(self) -> Self;
}

impl Canonical for f32 {
    #[inline]
    fn canonical(self) -> Self {
        canonical_f32(self)
    }
}

impl Canonical for f64 {
    #[inline]
    fn canonical(self) -> Self {
        canonical_f64(self)
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsFloatType,
    T::Native: Float + Canonical,
{
    pub fn to_canonical(&self) -> Self {
        unary_elementwise_values(self, |v| v.canonical())
    }
}
