use crate::prelude::*;
use arrow::compute::comparison::Simd8;
use num::Zero;

impl<T> ChunkPeaks for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumComp + Zero + Simd8,
{
    /// Get a boolean mask of the local maximum peaks.
    fn peak_max(&self) -> BooleanChunked {
        (self.shift_and_fill(1, Some(Zero::zero())).lt(self))
            & (self.shift_and_fill(-1, Some(Zero::zero())).lt(self))
    }

    /// Get a boolean mask of the local minimum peaks.
    fn peak_min(&self) -> BooleanChunked {
        (self.shift_and_fill(1, Some(Zero::zero())).gt(self))
            & (self.shift_and_fill(-1, Some(Zero::zero())).gt(self))
    }
}

impl ChunkPeaks for BooleanChunked {}
impl ChunkPeaks for Utf8Chunked {}
impl ChunkPeaks for CategoricalChunked {}
impl ChunkPeaks for ListChunked {}

#[cfg(feature = "object")]
impl<T> ChunkPeaks for ObjectChunked<T> where
    T: 'static + std::fmt::Debug + Clone + Send + Sync + Default
{
}
