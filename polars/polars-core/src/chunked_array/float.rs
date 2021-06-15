use crate::chunked_array::kernels::{is_finite, is_infinite, is_nan, is_not_nan};
use crate::prelude::*;
use num::Float;

impl<T> ChunkedArray<T>
where
    T: PolarsFloatType,
    T::Native: Float,
{
    pub fn is_nan(&self) -> BooleanChunked {
        self.apply_kernel_cast(is_nan::<T::Native>)
    }
    pub fn is_not_nan(&self) -> BooleanChunked {
        self.apply_kernel_cast(is_not_nan::<T::Native>)
    }
    pub fn is_finite(&self) -> BooleanChunked {
        self.apply_kernel_cast(is_finite)
    }
    pub fn is_infinite(&self) -> BooleanChunked {
        self.apply_kernel_cast(is_infinite)
    }
}
