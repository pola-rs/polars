use num_traits::Zero;
use polars_core::prelude::*;

/// Get a boolean mask of the local maximum peaks.
pub fn peak_max<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> BooleanChunked
where
    ChunkedArray<T>: for<'a> ChunkCompare<&'a ChunkedArray<T>, Item = BooleanChunked>,
{
    let shift_left = ca.shift_and_fill(1, Some(Zero::zero()));
    let shift_right = ca.shift_and_fill(-1, Some(Zero::zero()));
    ChunkedArray::lt(&shift_left, ca) & ChunkedArray::lt(&shift_right, ca)
}

/// Get a boolean mask of the local minimum peaks.
pub fn peak_min<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> BooleanChunked
where
    ChunkedArray<T>: for<'a> ChunkCompare<&'a ChunkedArray<T>, Item = BooleanChunked>,
{
    let shift_left = ca.shift_and_fill(1, Some(Zero::zero()));
    let shift_right = ca.shift_and_fill(-1, Some(Zero::zero()));
    ChunkedArray::gt(&shift_left, ca) & ChunkedArray::gt(&shift_right, ca)
}
