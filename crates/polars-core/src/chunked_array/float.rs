use arrow::legacy::kernels::float::*;
use arrow::legacy::kernels::set::set_at_nulls;
use num_traits::Float;

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
