use arrow::array::{Array, PrimitiveArray};
use polars_arrow::utils::combine_validities_and;

use crate::chunked_array::ops::apply::collect_array;
use crate::datatypes::{
    HasUnderlyingArray, PolarsNumericType, StaticArray, StaticallyMatchesPolarsType,
};
use crate::prelude::{ChunkedArray, PolarsDataType};
use crate::utils::align_chunks_binary;

#[inline]
pub fn binary_elementwise<T, U, V, F>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
) -> ChunkedArray<V>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsNumericType,
    ChunkedArray<T>: HasUnderlyingArray,
    ChunkedArray<U>: HasUnderlyingArray,
    F: for<'a> FnMut(
        Option<<<ChunkedArray<T> as HasUnderlyingArray>::ArrayT as StaticArray>::ValueT<'a>>,
        Option<<<ChunkedArray<U> as HasUnderlyingArray>::ArrayT as StaticArray>::ValueT<'a>>,
    ) -> Option<V::Native>,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let iter = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| {
            lhs_arr
                .iter()
                .zip(rhs_arr.iter())
                .map(|(lhs_opt_val, rhs_opt_val)| op(lhs_opt_val, rhs_opt_val))
                .collect::<PrimitiveArray<V::Native>>()
        });
    ChunkedArray::from_chunk_iter(lhs.name(), iter)
}

#[inline]
pub fn binary_elementwise_values<T, U, V, F>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
) -> ChunkedArray<V>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsNumericType,
    ChunkedArray<T>: HasUnderlyingArray,
    ChunkedArray<U>: HasUnderlyingArray,
    F: for<'a> FnMut(
        <<ChunkedArray<T> as HasUnderlyingArray>::ArrayT as StaticArray>::ValueT<'a>,
        <<ChunkedArray<U> as HasUnderlyingArray>::ArrayT as StaticArray>::ValueT<'a>,
    ) -> V::Native,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let iter = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| {
            let validity = combine_validities_and(lhs_arr.validity(), rhs_arr.validity());

            let iter = lhs_arr
                .values_iter()
                .zip(rhs_arr.values_iter())
                .map(|(lhs_val, rhs_val)| op(lhs_val, rhs_val));
            collect_array(iter, validity)
        });
    ChunkedArray::from_chunk_iter(lhs.name(), iter)
}

/// Applies a kernel that produces `Array` types.
#[inline]
pub fn binary_mut_with_options<T, U, V, F, Arr>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
    name: &str,
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
    ChunkedArray::from_chunk_iter(name, iter)
}

/// Applies a kernel that produces `Array` types.
pub fn binary_mut<T, U, V, F, Arr>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    op: F,
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
    binary_mut_with_options(lhs, rhs, op, lhs.name())
}

/// Applies a kernel that produces `ArrayRef` of the same type.
///
/// # Safety
/// Caller must ensure that the returned `ArrayRef` belongs to `T: PolarsDataType`.
#[inline]
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
