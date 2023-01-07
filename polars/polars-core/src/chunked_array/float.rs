use num::Float;
use polars_arrow::kernels::float::*;
use polars_arrow::kernels::set::set_at_nulls;

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
            .map(|arr| Box::new(set_at_nulls(arr, T::Native::nan())) as ArrayRef)
            .collect();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }
}
