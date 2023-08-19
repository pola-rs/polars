use arrow::array::Array;

use crate::datatypes::{HasUnderlyingArray, StaticallyMatchesPolarsType};
use crate::prelude::{ChunkedArray, PolarsDataType};
use crate::utils::align_chunks_binary;

/// Applies a kernel that produces `Array` types.
pub fn binary_mut<T, U, V, F, Arr>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
) -> ChunkedArray<V>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsDataType,
    ChunkedArray<T>: HasUnderlyingArray,
    ChunkedArray<U>: HasUnderlyingArray,
    Arr: Array + StaticallyMatchesPolarsType<V>,
    F: FnMut(
        &<ChunkedArray<T> as HasUnderlyingArray>::ArrayT,
        &<ChunkedArray<U> as HasUnderlyingArray>::ArrayT,
    ) -> Arr,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let iter = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| op(lhs_arr, rhs_arr));
    ChunkedArray::from_chunk_iter(lhs.name(), iter)
}

/// Applies a kernel that produces `ArrayRef` of the same type.
///
/// # Safety
/// Caller must ensure that the returned `ArrayRef` belongs to `T: PolarsDataType`.
pub unsafe fn binary_mut_unchecked_same_type<T, U, F>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
    keep_sorted: bool,
    keep_fast_explode: bool,
) -> ChunkedArray<T>
where
    T: PolarsDataType,
    U: PolarsDataType,
    ChunkedArray<T>: HasUnderlyingArray,
    ChunkedArray<U>: HasUnderlyingArray,
    F: FnMut(
        &<ChunkedArray<T> as HasUnderlyingArray>::ArrayT,
        &<ChunkedArray<U> as HasUnderlyingArray>::ArrayT,
    ) -> Box<dyn Array>,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let chunks = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| op(lhs_arr, rhs_arr))
        .collect();
    lhs.copy_with_chunks(chunks, keep_sorted, keep_fast_explode)
}
