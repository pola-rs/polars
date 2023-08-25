use std::error::Error;

use arrow::array::Array;
use polars_arrow::utils::combine_validities_and;

use crate::datatypes::{
    ArrayFromElementIter, HasUnderlyingArray, PolarsNumericType, StaticArray,
    StaticallyMatchesPolarsType,
};
use crate::prelude::{ChunkedArray, PolarsDataType};
use crate::utils::align_chunks_binary;

#[inline]
pub fn binary_elementwise<T, U, V, F, K>(
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
    F: for<'a> FnMut(
        Option<<<ChunkedArray<T> as HasUnderlyingArray>::ArrayT as StaticArray>::ValueT<'a>>,
        Option<<<ChunkedArray<U> as HasUnderlyingArray>::ArrayT as StaticArray>::ValueT<'a>>,
    ) -> Option<K>,
    K: ArrayFromElementIter,
    K::ArrayType: StaticallyMatchesPolarsType<V>,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let iter = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| {
            let element_iter = lhs_arr
                .iter()
                .zip(rhs_arr.iter())
                .map(|(lhs_opt_val, rhs_opt_val)| op(lhs_opt_val, rhs_opt_val));
            K::array_from_iter(element_iter)
        });
    ChunkedArray::from_chunk_iter(lhs.name(), iter)
}

#[inline]
pub fn try_binary_elementwise<T, U, V, F, K, E>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
) -> Result<ChunkedArray<V>, E>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsDataType,
    ChunkedArray<T>: HasUnderlyingArray,
    ChunkedArray<U>: HasUnderlyingArray,
    F: for<'a> FnMut(
        Option<<<ChunkedArray<T> as HasUnderlyingArray>::ArrayT as StaticArray>::ValueT<'a>>,
        Option<<<ChunkedArray<U> as HasUnderlyingArray>::ArrayT as StaticArray>::ValueT<'a>>,
    ) -> Result<Option<K>, E>,
    K: ArrayFromElementIter,
    K::ArrayType: StaticallyMatchesPolarsType<V>,
    E: Error,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let iter = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| {
            let element_iter = lhs_arr
                .iter()
                .zip(rhs_arr.iter())
                .map(|(lhs_opt_val, rhs_opt_val)| op(lhs_opt_val, rhs_opt_val));
            K::try_array_from_iter(element_iter)
        });
    ChunkedArray::try_from_chunk_iter(lhs.name(), iter)
}

#[inline]
pub fn binary_elementwise_values<T, U, V, F, K>(
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
    ) -> K,
    K: ArrayFromElementIter,
    K::ArrayType: StaticallyMatchesPolarsType<V>,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let iter = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| {
            let validity = combine_validities_and(lhs_arr.validity(), rhs_arr.validity());

            let element_iter = lhs_arr
                .values_iter()
                .zip(rhs_arr.values_iter())
                .map(|(lhs_val, rhs_val)| op(lhs_val, rhs_val));

            let array = K::array_from_values_iter(element_iter);
            array.with_validity_typed(validity)
        });
    ChunkedArray::from_chunk_iter(lhs.name(), iter)
}

#[inline]
pub fn try_binary_elementwise_values<T, U, V, F, K, E>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
) -> Result<ChunkedArray<V>, E>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsNumericType,
    ChunkedArray<T>: HasUnderlyingArray,
    ChunkedArray<U>: HasUnderlyingArray,
    F: for<'a> FnMut(
        <<ChunkedArray<T> as HasUnderlyingArray>::ArrayT as StaticArray>::ValueT<'a>,
        <<ChunkedArray<U> as HasUnderlyingArray>::ArrayT as StaticArray>::ValueT<'a>,
    ) -> Result<K, E>,
    K: ArrayFromElementIter,
    K::ArrayType: StaticallyMatchesPolarsType<V>,
    E: Error,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let iter = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| {
            let validity = combine_validities_and(lhs_arr.validity(), rhs_arr.validity());

            let element_iter = lhs_arr
                .values_iter()
                .zip(rhs_arr.values_iter())
                .map(|(lhs_val, rhs_val)| op(lhs_val, rhs_val));

            let array = K::try_array_from_values_iter(element_iter)?;
            Ok(array.with_validity_typed(validity))
        });
    ChunkedArray::try_from_chunk_iter(lhs.name(), iter)
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
pub fn binary<T, U, V, F, Arr>(
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

/// Applies a kernel that produces `Array` types.
pub fn try_binary<T, U, V, F, Arr, E>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
) -> Result<ChunkedArray<V>, E>
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
    ) -> Result<Arr, E>,
    E: Error,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let iter = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| op(lhs_arr, rhs_arr));
    ChunkedArray::try_from_chunk_iter(lhs.name(), iter)
}

/// Applies a kernel that produces `ArrayRef` of the same type.
///
/// # Safety
/// Caller must ensure that the returned `ArrayRef` belongs to `T: PolarsDataType`.
#[inline]
pub unsafe fn binary_unchecked_same_type<T, U, F>(
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

/// Applies a kernel that produces `ArrayRef` of the same type.
///
/// # Safety
/// Caller must ensure that the returned `ArrayRef` belongs to `T: PolarsDataType`.
#[inline]
pub unsafe fn try_binary_unchecked_same_type<T, U, F, E>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
    keep_sorted: bool,
    keep_fast_explode: bool,
) -> Result<ChunkedArray<T>, E>
where
    T: PolarsDataType,
    U: PolarsDataType,
    ChunkedArray<T>: HasUnderlyingArray,
    ChunkedArray<U>: HasUnderlyingArray,
    F: FnMut(
        &<ChunkedArray<T> as HasUnderlyingArray>::ArrayT,
        &<ChunkedArray<U> as HasUnderlyingArray>::ArrayT,
    ) -> Result<Box<dyn Array>, E>,
    E: Error,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let chunks = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| op(lhs_arr, rhs_arr))
        .collect::<Result<Vec<_>, E>>()?;
    Ok(lhs.copy_with_chunks(chunks, keep_sorted, keep_fast_explode))
}
