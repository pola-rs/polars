use std::error::Error;

use arrow::array::{Array, MutablePlString, StaticArray};
use arrow::compute::utils::combine_validities_and;
use polars_error::PolarsResult;

use crate::chunked_array::metadata::MetadataProperties;
use crate::datatypes::{ArrayCollectIterExt, ArrayFromIter};
use crate::prelude::{ChunkedArray, CompatLevel, PolarsDataType, Series, StringChunked};
use crate::utils::{align_chunks_binary, align_chunks_binary_owned, align_chunks_ternary};

// We need this helper because for<'a> notation can't yet be applied properly
// on the return type.
pub trait UnaryFnMut<A1>: FnMut(A1) -> Self::Ret {
    type Ret;
}

impl<A1, R, T: FnMut(A1) -> R> UnaryFnMut<A1> for T {
    type Ret = R;
}

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
#[inline]
pub fn unary_kernel<T, V, F, Arr>(ca: &ChunkedArray<T>, op: F) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: Array,
    F: FnMut(&T::Array) -> Arr,
{
    let iter = ca.downcast_iter().map(op);
    ChunkedArray::from_chunk_iter(ca.name(), iter)
}

/// Applies a kernel that produces `Array` types.
#[inline]
pub fn unary_kernel_owned<T, V, F, Arr>(ca: ChunkedArray<T>, op: F) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: Array,
    F: FnMut(T::Array) -> Arr,
{
    let name = ca.name().to_owned();
    let iter = ca.downcast_into_iter().map(op);
    ChunkedArray::from_chunk_iter(&name, iter)
}

#[inline]
pub fn unary_elementwise<'a, T, V, F>(ca: &'a ChunkedArray<T>, mut op: F) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType,
    F: UnaryFnMut<Option<T::Physical<'a>>>,
    V::Array: ArrayFromIter<<F as UnaryFnMut<Option<T::Physical<'a>>>>::Ret>,
{
    let iter = ca
        .downcast_iter()
        .map(|arr| arr.iter().map(&mut op).collect_arr());
    ChunkedArray::from_chunk_iter(ca.name(), iter)
}

#[inline]
pub fn try_unary_elementwise<'a, T, V, F, K, E>(
    ca: &'a ChunkedArray<T>,
    mut op: F,
) -> Result<ChunkedArray<V>, E>
where
    T: PolarsDataType,
    V: PolarsDataType,
    F: FnMut(Option<T::Physical<'a>>) -> Result<Option<K>, E>,
    V::Array: ArrayFromIter<Option<K>>,
{
    let iter = ca
        .downcast_iter()
        .map(|arr| arr.iter().map(&mut op).try_collect_arr());
    ChunkedArray::try_from_chunk_iter(ca.name(), iter)
}

#[inline]
pub fn unary_elementwise_values<'a, T, V, F>(ca: &'a ChunkedArray<T>, mut op: F) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType,
    F: UnaryFnMut<T::Physical<'a>>,
    V::Array: ArrayFromIter<<F as UnaryFnMut<T::Physical<'a>>>::Ret>,
{
    if ca.null_count() == ca.len() {
        let arr = V::Array::full_null(ca.len(), V::get_dtype().to_arrow(CompatLevel::newest()));
        return ChunkedArray::with_chunk(ca.name(), arr);
    }

    let iter = ca.downcast_iter().map(|arr| {
        let validity = arr.validity().cloned();
        let arr: V::Array = arr.values_iter().map(&mut op).collect_arr();
        arr.with_validity_typed(validity)
    });
    ChunkedArray::from_chunk_iter(ca.name(), iter)
}

#[inline]
pub fn try_unary_elementwise_values<'a, T, V, F, K, E>(
    ca: &'a ChunkedArray<T>,
    mut op: F,
) -> Result<ChunkedArray<V>, E>
where
    T: PolarsDataType,
    V: PolarsDataType,
    F: FnMut(T::Physical<'a>) -> Result<K, E>,
    V::Array: ArrayFromIter<K>,
{
    if ca.null_count() == ca.len() {
        let arr = V::Array::full_null(ca.len(), V::get_dtype().to_arrow(CompatLevel::newest()));
        return Ok(ChunkedArray::with_chunk(ca.name(), arr));
    }

    let iter = ca.downcast_iter().map(|arr| {
        let validity = arr.validity().cloned();
        let arr: V::Array = arr.values_iter().map(&mut op).try_collect_arr()?;
        Ok(arr.with_validity_typed(validity))
    });
    ChunkedArray::try_from_chunk_iter(ca.name(), iter)
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
pub fn unary_mut_with_options<T, V, F, Arr>(ca: &ChunkedArray<T>, op: F) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: Array + StaticArray,
    F: FnMut(&T::Array) -> Arr,
{
    ChunkedArray::from_chunk_iter(ca.name(), ca.downcast_iter().map(op))
}

#[inline]
pub fn try_unary_mut_with_options<T, V, F, Arr, E>(
    ca: &ChunkedArray<T>,
    op: F,
) -> Result<ChunkedArray<V>, E>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: Array + StaticArray,
    F: FnMut(&T::Array) -> Result<Arr, E>,
    E: Error,
{
    ChunkedArray::try_from_chunk_iter(ca.name(), ca.downcast_iter().map(op))
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
    if lhs.null_count() == lhs.len() || rhs.null_count() == rhs.len() {
        let len = lhs.len().min(rhs.len());
        let arr = V::Array::full_null(len, V::get_dtype().to_arrow(CompatLevel::newest()));

        return ChunkedArray::with_chunk(lhs.name(), arr);
    }

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

pub fn binary_elementwise_into_string_amortized<T, U, F>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
) -> StringChunked
where
    T: PolarsDataType,
    U: PolarsDataType,
    F: for<'a> FnMut(T::Physical<'a>, U::Physical<'a>, &mut String),
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let mut buf = String::new();
    let iter = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| {
            let mut mutarr = MutablePlString::with_capacity(lhs_arr.len());
            lhs_arr
                .iter()
                .zip(rhs_arr.iter())
                .for_each(|(lhs_opt, rhs_opt)| match (lhs_opt, rhs_opt) {
                    (None, _) | (_, None) => mutarr.push_null(),
                    (Some(lhs_val), Some(rhs_val)) => {
                        buf.clear();
                        op(lhs_val, rhs_val, &mut buf);
                        mutarr.push_value(&buf)
                    },
                });
            mutarr.freeze()
        });
    ChunkedArray::from_chunk_iter(lhs.name(), iter)
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

#[inline]
pub fn try_binary_mut_with_options<T, U, V, F, Arr, E>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
    name: &str,
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
    ChunkedArray::try_from_chunk_iter(name, iter)
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
pub fn binary_owned<L, R, V, F, Arr>(
    lhs: ChunkedArray<L>,
    rhs: ChunkedArray<R>,
    mut op: F,
) -> ChunkedArray<V>
where
    L: PolarsDataType,
    R: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: Array,
    F: FnMut(L::Array, R::Array) -> Arr,
{
    let name = lhs.name().to_owned();
    let (lhs, rhs) = align_chunks_binary_owned(lhs, rhs);
    let iter = lhs
        .downcast_into_iter()
        .zip(rhs.downcast_into_iter())
        .map(|(lhs_arr, rhs_arr)| op(lhs_arr, rhs_arr));
    ChunkedArray::from_chunk_iter(&name, iter)
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

    let mut ca = lhs.copy_with_chunks(chunks);

    use MetadataProperties as P;

    let mut properties = P::empty();
    properties.set(P::SORTED, keep_sorted);
    properties.set(P::FAST_EXPLODE_LIST, keep_fast_explode);
    ca.copy_metadata(&lhs, properties);

    ca
}

#[inline]
pub fn binary_to_series<T, U, F>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
) -> PolarsResult<Series>
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
        .collect::<Vec<_>>();
    Series::try_from((lhs.name(), chunks))
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
    let mut ca = lhs.copy_with_chunks(chunks);

    use MetadataProperties as P;
    let mut properties = P::empty();
    properties.set(P::SORTED, keep_sorted);
    properties.set(P::FAST_EXPLODE_LIST, keep_fast_explode);
    ca.copy_metadata(&lhs, properties);

    Ok(ca)
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

pub fn broadcast_binary_elementwise<T, U, V, F>(
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
    match (lhs.len(), rhs.len()) {
        (1, _) => {
            let a = unsafe { lhs.get_unchecked(0) };
            unary_elementwise(rhs, |b| op(a.clone(), b)).with_name(lhs.name())
        },
        (_, 1) => {
            let b = unsafe { rhs.get_unchecked(0) };
            unary_elementwise(lhs, |a| op(a, b.clone()))
        },
        _ => binary_elementwise(lhs, rhs, op),
    }
}

pub fn broadcast_try_binary_elementwise<T, U, V, F, K, E>(
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
    match (lhs.len(), rhs.len()) {
        (1, _) => {
            let a = unsafe { lhs.get_unchecked(0) };
            Ok(try_unary_elementwise(rhs, |b| op(a.clone(), b))?.with_name(lhs.name()))
        },
        (_, 1) => {
            let b = unsafe { rhs.get_unchecked(0) };
            try_unary_elementwise(lhs, |a| op(a, b.clone()))
        },
        _ => try_binary_elementwise(lhs, rhs, op),
    }
}

pub fn broadcast_binary_elementwise_values<T, U, V, F, K>(
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
    if lhs.null_count() == lhs.len() || rhs.null_count() == rhs.len() {
        let min = lhs.len().min(rhs.len());
        let max = lhs.len().max(rhs.len());
        let len = if min == 1 { max } else { min };
        let arr = V::Array::full_null(len, V::get_dtype().to_arrow(CompatLevel::newest()));

        return ChunkedArray::with_chunk(lhs.name(), arr);
    }

    match (lhs.len(), rhs.len()) {
        (1, _) => {
            let a = unsafe { lhs.value_unchecked(0) };
            unary_elementwise_values(rhs, |b| op(a.clone(), b)).with_name(lhs.name())
        },
        (_, 1) => {
            let b = unsafe { rhs.value_unchecked(0) };
            unary_elementwise_values(lhs, |a| op(a, b.clone()))
        },
        _ => binary_elementwise_values(lhs, rhs, op),
    }
}

pub fn apply_binary_kernel_broadcast<'l, 'r, L, R, O, K, LK, RK>(
    lhs: &'l ChunkedArray<L>,
    rhs: &'r ChunkedArray<R>,
    kernel: K,
    lhs_broadcast_kernel: LK,
    rhs_broadcast_kernel: RK,
) -> ChunkedArray<O>
where
    L: PolarsDataType,
    R: PolarsDataType,
    O: PolarsDataType,
    K: Fn(&L::Array, &R::Array) -> O::Array,
    LK: Fn(L::Physical<'l>, &R::Array) -> O::Array,
    RK: Fn(&L::Array, R::Physical<'r>) -> O::Array,
{
    let name = lhs.name();
    let out = match (lhs.len(), rhs.len()) {
        (a, b) if a == b => binary(lhs, rhs, |lhs, rhs| kernel(lhs, rhs)),
        // broadcast right path
        (_, 1) => {
            let opt_rhs = rhs.get(0);
            match opt_rhs {
                None => {
                    let arr = O::Array::full_null(
                        lhs.len(),
                        O::get_dtype().to_arrow(CompatLevel::newest()),
                    );
                    ChunkedArray::<O>::with_chunk(lhs.name(), arr)
                },
                Some(rhs) => unary_kernel(lhs, |arr| rhs_broadcast_kernel(arr, rhs.clone())),
            }
        },
        (1, _) => {
            let opt_lhs = lhs.get(0);
            match opt_lhs {
                None => {
                    let arr = O::Array::full_null(
                        rhs.len(),
                        O::get_dtype().to_arrow(CompatLevel::newest()),
                    );
                    ChunkedArray::<O>::with_chunk(lhs.name(), arr)
                },
                Some(lhs) => unary_kernel(rhs, |arr| lhs_broadcast_kernel(lhs.clone(), arr)),
            }
        },
        _ => panic!("Cannot apply operation on arrays of different lengths"),
    };
    out.with_name(name)
}

pub fn apply_binary_kernel_broadcast_owned<L, R, O, K, LK, RK>(
    lhs: ChunkedArray<L>,
    rhs: ChunkedArray<R>,
    kernel: K,
    lhs_broadcast_kernel: LK,
    rhs_broadcast_kernel: RK,
) -> ChunkedArray<O>
where
    L: PolarsDataType,
    R: PolarsDataType,
    O: PolarsDataType,
    K: Fn(L::Array, R::Array) -> O::Array,
    for<'a> LK: Fn(L::Physical<'a>, R::Array) -> O::Array,
    for<'a> RK: Fn(L::Array, R::Physical<'a>) -> O::Array,
{
    let name = lhs.name().to_owned();
    let out = match (lhs.len(), rhs.len()) {
        (a, b) if a == b => binary_owned(lhs, rhs, kernel),
        // broadcast right path
        (_, 1) => {
            let opt_rhs = rhs.get(0);
            match opt_rhs {
                None => {
                    let arr = O::Array::full_null(
                        lhs.len(),
                        O::get_dtype().to_arrow(CompatLevel::newest()),
                    );
                    ChunkedArray::<O>::with_chunk(lhs.name(), arr)
                },
                Some(rhs) => unary_kernel_owned(lhs, |arr| rhs_broadcast_kernel(arr, rhs.clone())),
            }
        },
        (1, _) => {
            let opt_lhs = lhs.get(0);
            match opt_lhs {
                None => {
                    let arr = O::Array::full_null(
                        rhs.len(),
                        O::get_dtype().to_arrow(CompatLevel::newest()),
                    );
                    ChunkedArray::<O>::with_chunk(lhs.name(), arr)
                },
                Some(lhs) => unary_kernel_owned(rhs, |arr| lhs_broadcast_kernel(lhs.clone(), arr)),
            }
        },
        _ => panic!("Cannot apply operation on arrays of different lengths"),
    };
    out.with_name(&name)
}
