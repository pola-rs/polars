use std::error::Error;

use arrow::array::Array;
use arrow::legacy::utils::combine_validities_and;

use crate::datatypes::{ArrayCollectIterExt, ArrayFromIter, StaticArray};
use crate::prelude::{ChunkedArray, PolarsDataType};
use crate::utils::{align_chunks_binary, align_chunks_ternary};

// We need this helper because for<'a> notation can't yet be applied properly
// on the return type.
pub trait TernaryFnMut<A1, A2, A3>: FnMut(A1, A2, A3) -> Self::Ret {
    type Ret;
}

impl<A1, A2, A3, R, T: FnMut(A1, A2, A3) -> R> TernaryFnMut<A1, A2, A3> for T {
    type Ret = R;
}

// We need this helper because for<'a> notation can't yet be applied properly
// on the return type.
pub trait BinaryFnMut<A1, A2>: FnMut(A1, A2) -> Self::Ret {
    type Ret;
}

impl<A1, A2, R, T: FnMut(A1, A2) -> R> BinaryFnMut<A1, A2> for T {
    type Ret = R;
}

/// Applies a kernel that produces `Array` types.
///
/// Intended for kernels that apply on values, this function will apply the
/// validity mask afterwards.
#[inline]
pub fn unary_mut_values<T, V, F, Arr>(ca: &ChunkedArray<T>, mut op: F) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: Array + StaticArray,
    F: FnMut(&T::Array) -> Arr,
{
    let iter = ca
        .downcast_iter()
        .map(|arr| op(arr).with_validity_typed(arr.validity().cloned()));
    ChunkedArray::from_chunk_iter(ca.name(), iter)
}

/// Applies a kernel that produces `Array` types.
#[inline]
pub fn unary_mut_with_options<T, V, F, Arr>(ca: &ChunkedArray<T>, mut op: F) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: Array + StaticArray,
    F: FnMut(&T::Array) -> Arr,
{
    let iter = ca.downcast_iter().map(|arr| op(arr));
    ChunkedArray::from_chunk_iter(ca.name(), iter)
}

#[inline]
pub fn binary_elementwise<T, U, V, F>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
) -> ChunkedArray<V>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsDataType,
    F: for<'a> BinaryFnMut<Option<T::Physical<'a>>, Option<U::Physical<'a>>>,
    V::Array: for<'a> ArrayFromIter<
        <F as BinaryFnMut<Option<T::Physical<'a>>, Option<U::Physical<'a>>>>::Ret,
    >,
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
            element_iter.collect_arr()
        });
    ChunkedArray::from_chunk_iter(lhs.name(), iter)
}

#[inline]
pub fn binary_elementwise_for_each<'a, 'b, T, U, F>(
    lhs: &'a ChunkedArray<T>,
    rhs: &'b ChunkedArray<U>,
    mut op: F,
) where
    T: PolarsDataType,
    U: PolarsDataType,
    F: FnMut(Option<T::Physical<'a>>, Option<U::Physical<'b>>),
{
    let mut lhs_arr_iter = lhs.downcast_iter();
    let mut rhs_arr_iter = rhs.downcast_iter();

    let lhs_arr = lhs_arr_iter.next().unwrap();
    let rhs_arr = rhs_arr_iter.next().unwrap();

    let mut lhs_remaining = lhs_arr.len();
    let mut rhs_remaining = rhs_arr.len();
    let mut lhs_iter = lhs_arr.iter();
    let mut rhs_iter = rhs_arr.iter();

    loop {
        let range = std::cmp::min(lhs_remaining, rhs_remaining);

        for _ in 0..range {
            // SAFETY: we loop until the smaller iter is exhausted.
            let lhs_opt_val = unsafe { lhs_iter.next().unwrap_unchecked() };
            let rhs_opt_val = unsafe { rhs_iter.next().unwrap_unchecked() };
            op(lhs_opt_val, rhs_opt_val)
        }
        lhs_remaining -= range;
        rhs_remaining -= range;

        if lhs_remaining == 0 {
            let Some(new_arr) = lhs_arr_iter.next() else {
                return;
            };
            lhs_remaining = new_arr.len();
            lhs_iter = new_arr.iter();
        }
        if rhs_remaining == 0 {
            let Some(new_arr) = rhs_arr_iter.next() else {
                return;
            };
            rhs_remaining = new_arr.len();
            rhs_iter = new_arr.iter();
        }
    }
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
    F: for<'a> FnMut(Option<T::Physical<'a>>, Option<U::Physical<'a>>) -> Result<Option<K>, E>,
    V::Array: ArrayFromIter<Option<K>>,
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
            element_iter.try_collect_arr()
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
    V: PolarsDataType,
    F: for<'a> FnMut(T::Physical<'a>, U::Physical<'a>) -> K,
    V::Array: ArrayFromIter<K>,
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

            let array: V::Array = element_iter.collect_arr();
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
    V: PolarsDataType,
    F: for<'a> FnMut(T::Physical<'a>, U::Physical<'a>) -> Result<K, E>,
    V::Array: ArrayFromIter<K>,
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

            let array: V::Array = element_iter.try_collect_arr()?;
            Ok(array.with_validity_typed(validity))
        });
    ChunkedArray::try_from_chunk_iter(lhs.name(), iter)
}

/// Applies a kernel that produces `Array` types.
///
/// Intended for kernels that apply on values, this function will filter out any
/// results which do not have two non-null inputs.
#[inline]
pub fn binary_mut_values<T, U, V, F, Arr>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
    name: &str,
) -> ChunkedArray<V>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: Array + StaticArray,
    F: FnMut(&T::Array, &U::Array) -> Arr,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let iter = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| {
            let ret = op(lhs_arr, rhs_arr);
            let inp_val = combine_validities_and(lhs_arr.validity(), rhs_arr.validity());
            let val = combine_validities_and(inp_val.as_ref(), ret.validity());
            ret.with_validity_typed(val)
        });
    ChunkedArray::from_chunk_iter(name, iter)
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
    V: PolarsDataType<Array = Arr>,
    Arr: Array,
    F: FnMut(&T::Array, &U::Array) -> Arr,
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
    V: PolarsDataType<Array = Arr>,
    Arr: Array,
    F: FnMut(&T::Array, &U::Array) -> Arr,
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
    V: PolarsDataType<Array = Arr>,
    Arr: Array,
    F: FnMut(&T::Array, &U::Array) -> Result<Arr, E>,
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
    F: FnMut(&T::Array, &U::Array) -> Box<dyn Array>,
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
    F: FnMut(&T::Array, &U::Array) -> Result<Box<dyn Array>, E>,
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

#[inline]
pub fn try_ternary_elementwise<T, U, V, G, F, K, E>(
    ca1: &ChunkedArray<T>,
    ca2: &ChunkedArray<U>,
    ca3: &ChunkedArray<G>,
    mut op: F,
) -> Result<ChunkedArray<V>, E>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsDataType,
    G: PolarsDataType,
    F: for<'a> FnMut(
        Option<T::Physical<'a>>,
        Option<U::Physical<'a>>,
        Option<G::Physical<'a>>,
    ) -> Result<Option<K>, E>,
    V::Array: ArrayFromIter<Option<K>>,
{
    let (ca1, ca2, ca3) = align_chunks_ternary(ca1, ca2, ca3);
    let iter = ca1
        .downcast_iter()
        .zip(ca2.downcast_iter())
        .zip(ca3.downcast_iter())
        .map(|((ca1_arr, ca2_arr), ca3_arr)| {
            let element_iter = ca1_arr.iter().zip(ca2_arr.iter()).zip(ca3_arr.iter()).map(
                |((ca1_opt_val, ca2_opt_val), ca3_opt_val)| {
                    op(ca1_opt_val, ca2_opt_val, ca3_opt_val)
                },
            );
            element_iter.try_collect_arr()
        });
    ChunkedArray::try_from_chunk_iter(ca1.name(), iter)
}

#[inline]
pub fn ternary_elementwise<T, U, V, G, F>(
    ca1: &ChunkedArray<T>,
    ca2: &ChunkedArray<U>,
    ca3: &ChunkedArray<G>,
    mut op: F,
) -> ChunkedArray<V>
where
    T: PolarsDataType,
    U: PolarsDataType,
    G: PolarsDataType,
    V: PolarsDataType,
    F: for<'a> TernaryFnMut<
        Option<T::Physical<'a>>,
        Option<U::Physical<'a>>,
        Option<G::Physical<'a>>,
    >,
    V::Array: for<'a> ArrayFromIter<
        <F as TernaryFnMut<
            Option<T::Physical<'a>>,
            Option<U::Physical<'a>>,
            Option<G::Physical<'a>>,
        >>::Ret,
    >,
{
    let (ca1, ca2, ca3) = align_chunks_ternary(ca1, ca2, ca3);
    let iter = ca1
        .downcast_iter()
        .zip(ca2.downcast_iter())
        .zip(ca3.downcast_iter())
        .map(|((ca1_arr, ca2_arr), ca3_arr)| {
            let element_iter = ca1_arr.iter().zip(ca2_arr.iter()).zip(ca3_arr.iter()).map(
                |((ca1_opt_val, ca2_opt_val), ca3_opt_val)| {
                    op(ca1_opt_val, ca2_opt_val, ca3_opt_val)
                },
            );
            element_iter.collect_arr()
        });
    ChunkedArray::from_chunk_iter(ca1.name(), iter)
}
