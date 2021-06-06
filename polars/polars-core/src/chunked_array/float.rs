use crate::chunked_array::kernels::{is_finite, is_infinite, is_nan, is_not_nan};
use crate::prelude::*;
use num::Float;

pub trait IsNan {
    fn is_nan(&self) -> BooleanChunked;
    fn is_not_nan(&self) -> BooleanChunked;
    fn is_finite(&self) -> BooleanChunked;
    fn is_infinite(&self) -> BooleanChunked;
}

impl<T> IsNan for ChunkedArray<T>
where
    T: PolarsFloatType,
    T::Native: Float,
{
    fn is_nan(&self) -> BooleanChunked {
        self.apply_kernel_cast(is_nan)
    }
    fn is_not_nan(&self) -> BooleanChunked {
        self.apply_kernel_cast(is_not_nan)
    }
    fn is_finite(&self) -> BooleanChunked {
        self.apply_kernel_cast(is_finite)
    }
    fn is_infinite(&self) -> BooleanChunked {
        self.apply_kernel_cast(is_infinite)
    }
}
