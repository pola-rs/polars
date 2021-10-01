use crate::prelude::*;
use num::Zero;

impl<T> ChunkPeaks for ChunkedArray<T>
where
    T: PolarsNumericType,
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
