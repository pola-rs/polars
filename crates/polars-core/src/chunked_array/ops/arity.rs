#![allow(unsafe_op_in_unsafe_fn)]
use std::error::Error;

use polars_array::bitmap::combine_validities_and;
use polars_array::builder::StaticArrayBuilder;
use polars_array::{Flat, PlArray, PlBitmap, PlBitmapRef, PlUtf8ViewArrayBuilder, StaticArray};
use polars_utils::pl_str::PlSmallStr;

use crate::chunked_array::flags::StatisticsFlags;
use crate::datatypes::{ArrayCollectIterExt, ArrayFromIter};
use crate::prelude::{ChunkedArray, PlArrayRef, PolarsDataType, StringChunked};
use crate::utils::{align_chunks_binary, align_chunks_binary_owned, align_chunks_ternary};

/// Returns `ret` masked off wherever either input has a null, on top of its own mask. A scalar
/// mask among the three is combined as the single bit it stands for, not written out first.
#[inline]
fn mask_with_inputs<A: StaticArray>(
    ret: A,
    lhs: Option<PlBitmapRef<'_>>,
    rhs: Option<PlBitmapRef<'_>>,
) -> A {
    // The combined mask covers the inputs' elements, which is what `ret` holds too: a kernel that
    // handed back a result of a different height than its inputs panics here.
    let inputs = combine_validities_and(lhs, rhs);
    let validity = combine_validities_and(inputs.as_ref().map(PlBitmap::as_ref), ret.validity());
    ret.with_validity_broadcast_typed(validity.map(PlBitmap::into_flat_or_scalar))
}

/// The height of the output of an elementwise operation over two columns of these lengths, or
/// `None` if the two do not broadcast. The operations here answer a mismatch with a panic.
#[inline]
pub fn broadcast_height(lhs: usize, rhs: usize) -> Option<usize> {
    match (lhs, rhs) {
        (lhs, rhs) if lhs == rhs => Some(lhs),
        (length, 1) | (1, length) => Some(length),
        _ => None,
    }
}

#[macro_export]
macro_rules! binary_output_height {
    ($a:expr, $b:expr, op = $op:expr) => {
        match ($a.len(), $b.len()) {
            (a, 1) | (1, a) => Ok(a),
            (a, b) if a == b => Ok(a),
            (a, b) => Err(polars_err!(
                ShapeMismatch:
                "{} got differing lengths \
                ({}: {}, {}: {})",
                $op,
                stringify!($a.len()), $a.len(),
                stringify!($b.len()), $b.len(),
            )),
        }
    }
}

#[macro_export]
macro_rules! ternary_output_height {
    ($a:expr, $b:expr, $c:expr, op = $op:expr) => {
        match ($a.len(), $b.len(), $c.len()) {
            (a, 1, 1) | (1, a, 1) | (1, 1, a) => Ok(a),
            (a, b, 1) | (a, 1, b) | (1, a, b) if a == b => Ok(a),
            (a, b, c) if a == b && b == c => Ok(a),
            (a, b, c) => Err(polars_err!(
                ShapeMismatch:
                "{} got differing lengths \
                ({}: {}, {}: {}, {}: {})",
                $op,
                stringify!($a.len()), $a.len(),
                stringify!($b.len()), $b.len(),
                stringify!($c.len()), $c.len(),
            )),
        }
    }
}

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
    Arr: StaticArray,
    F: FnMut(&T::Array) -> Arr,
{
    let iter = ca.downcast_iter().map(op);
    ChunkedArray::from_chunk_iter(ca.name().clone(), iter)
}

/// Applies an elementwise kernel written against the [`flat`](polars_array::broadcast)
/// representation to one chunk, leaving a [`scalar`](polars_array::broadcast) chunk scalar.
///
/// A flat chunk reaches `op` as it is. A scalar chunk is not written out: `op` is handed the
/// single element it repeats, and the one element that comes back stands for the whole chunk.
///
/// `op` must be elementwise — every element of the result a function of the element at the same
/// index alone — which is what makes the answer for one element the answer for every element.
#[inline]
fn elementwise_flat<A, Arr, F>(arr: &A, op: &mut F) -> Arr
where
    A: StaticArray,
    Arr: StaticArray,
    F: FnMut(&Flat<A>) -> Arr,
{
    if let Some(flat) = arr.as_flat() {
        return op(flat);
    }

    let length = arr.len();
    if !PlArray::is_scalar(arr) || length < 2 {
        // Only some of the buffers hold a single slot, so there is no one element standing for
        // the rest; a chunk of a single element has nothing to save either.
        return op(&arr.to_flat());
    }

    // Sliced down to the one element the chunk repeats, which leaves every buffer holding the
    // single slot it already held, and is therefore flat as well as `O(1)`.
    let mut single = arr.clone();
    single.slice(0, 1);

    let out = op(&single.to_flat());
    debug_assert_eq!(
        out.len(),
        1,
        "an elementwise kernel answers one element with one"
    );

    out.new_from_index_typed(0, length)
}

/// [`elementwise_flat`] for a kernel that reads two chunks of the same height at once.
///
/// The shortcut is taken only when both chunks are scalar: a kernel reading one flat side cannot
/// be answered by a single element of the other.
#[inline]
fn elementwise_binary_flat<A, B, Arr, F>(lhs: &A, rhs: &B, op: &mut F) -> Arr
where
    A: StaticArray,
    B: StaticArray,
    Arr: StaticArray,
    F: FnMut(&Flat<A>, &Flat<B>) -> Arr,
{
    if let (Some(lhs), Some(rhs)) = (lhs.as_flat(), rhs.as_flat()) {
        return op(lhs, rhs);
    }

    // The chunks are aligned, so the two lengths are the same one.
    let length = lhs.len();
    if length < 2 || !PlArray::is_scalar(lhs) || !PlArray::is_scalar(rhs) {
        return op(&lhs.to_flat(), &rhs.to_flat());
    }

    let (mut lhs, mut rhs) = (lhs.clone(), rhs.clone());
    lhs.slice(0, 1);
    rhs.slice(0, 1);

    let out = op(&lhs.to_flat(), &rhs.to_flat());
    debug_assert_eq!(
        out.len(),
        1,
        "an elementwise kernel answers one element with one"
    );

    out.new_from_index_typed(0, length)
}

/// Applies an elementwise kernel written against the [`flat`](polars_array::broadcast)
/// representation: this is [`unary_kernel`] over the backing buffers.
///
/// `op` must be elementwise; a [`scalar`](polars_array::broadcast) chunk reaches it as the single
/// element it repeats, and the result is repeated in turn. See [`elementwise_flat`].
#[inline]
pub fn unary_elementwise_kernel_flat<T, V, F, Arr>(
    ca: &ChunkedArray<T>,
    mut op: F,
) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
    F: FnMut(&Flat<T::Array>) -> Arr,
{
    let iter = ca.downcast_iter().map(|arr| elementwise_flat(arr, &mut op));
    ChunkedArray::from_chunk_iter(ca.name().clone(), iter)
}

/// Applies a kernel that produces `Array` types.
#[inline]
pub fn unary_kernel_owned<T, V, F, Arr>(ca: ChunkedArray<T>, op: F) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
    F: FnMut(T::Array) -> Arr,
{
    let name = ca.name().clone();
    let iter = ca.downcast_into_iter().map(op);
    ChunkedArray::from_chunk_iter(name, iter)
}

#[inline]
pub fn unary_elementwise<'a, T, V, F>(ca: &'a ChunkedArray<T>, mut op: F) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType,
    F: UnaryFnMut<Option<T::Physical<'a>>>,
    V::Array: ArrayFromIter<<F as UnaryFnMut<Option<T::Physical<'a>>>>::Ret>,
{
    if ca.has_nulls() {
        let iter = ca
            .downcast_iter()
            .map(|arr| arr.iter().map(&mut op).collect_arr());
        ChunkedArray::from_chunk_iter(ca.name().clone(), iter)
    } else {
        let iter = ca
            .downcast_iter()
            .map(|arr| arr.values_iter().map(|x| op(Some(x))).collect_arr());
        ChunkedArray::from_chunk_iter(ca.name().clone(), iter)
    }
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
    ChunkedArray::try_from_chunk_iter(ca.name().clone(), iter)
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
        return ChunkedArray::with_chunk(ca.name().clone(), V::full_null_array(ca.len()));
    }

    let iter = ca.downcast_iter().map(|arr| {
        let validity = arr.validity().map(|v| v.to_flat_or_scalar());
        let arr: V::Array = arr.values_iter().map(&mut op).collect_arr();
        arr.with_validity_broadcast_typed(validity)
    });
    ChunkedArray::from_chunk_iter(ca.name().clone(), iter)
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
        return Ok(ChunkedArray::with_chunk(
            ca.name().clone(),
            V::full_null_array(ca.len()),
        ));
    }

    let iter = ca.downcast_iter().map(|arr| {
        let validity = arr.validity().map(|v| v.to_flat_or_scalar());
        let arr: V::Array = arr.values_iter().map(&mut op).try_collect_arr()?;
        Ok(arr.with_validity_broadcast_typed(validity))
    });
    ChunkedArray::try_from_chunk_iter(ca.name().clone(), iter)
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
    Arr: StaticArray,
    F: FnMut(&T::Array) -> Arr,
{
    let iter = ca.downcast_iter().map(|arr| {
        op(arr).with_validity_broadcast_typed(arr.validity().map(|v| v.to_flat_or_scalar()))
    });
    ChunkedArray::from_chunk_iter(ca.name().clone(), iter)
}

/// Applies an elementwise kernel written against the [`flat`](polars_array::broadcast)
/// representation, putting the input's validity mask back on the result: [`unary_mut_values`]
/// over the buffers.
///
/// `op` must be elementwise; a [`scalar`](polars_array::broadcast) chunk reaches it as the single
/// element it repeats, and the result is repeated in turn. See [`elementwise_flat`].
#[inline]
pub fn unary_elementwise_mut_values_flat<T, V, F, Arr>(
    ca: &ChunkedArray<T>,
    mut op: F,
) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
    F: FnMut(&Flat<T::Array>) -> Arr,
{
    let iter = ca.downcast_iter().map(|arr| {
        elementwise_flat(arr, &mut op)
            .with_validity_broadcast_typed(arr.validity().map(|v| v.to_flat_or_scalar()))
    });
    ChunkedArray::from_chunk_iter(ca.name().clone(), iter)
}

/// Applies a kernel that produces `Array` types.
#[inline]
pub fn unary_mut_with_options<T, V, F, Arr>(ca: &ChunkedArray<T>, op: F) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
    F: FnMut(&T::Array) -> Arr,
{
    ChunkedArray::from_chunk_iter(ca.name().clone(), ca.downcast_iter().map(op))
}

/// Applies an elementwise kernel written against the [`flat`](polars_array::broadcast)
/// representation, leaving the result's own validity mask alone: [`unary_mut_with_options`] over
/// the buffers.
///
/// `op` must be elementwise; a [`scalar`](polars_array::broadcast) chunk reaches it as the single
/// element it repeats, and the result is repeated in turn. See [`elementwise_flat`].
#[inline]
pub fn unary_elementwise_mut_with_options_flat<T, V, F, Arr>(
    ca: &ChunkedArray<T>,
    mut op: F,
) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
    F: FnMut(&Flat<T::Array>) -> Arr,
{
    let iter = ca.downcast_iter().map(|arr| elementwise_flat(arr, &mut op));
    ChunkedArray::from_chunk_iter(ca.name().clone(), iter)
}

#[inline]
pub fn try_unary_mut_with_options<T, V, F, Arr, E>(
    ca: &ChunkedArray<T>,
    op: F,
) -> Result<ChunkedArray<V>, E>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
    F: FnMut(&T::Array) -> Result<Arr, E>,
    E: Error,
{
    ChunkedArray::try_from_chunk_iter(ca.name().clone(), ca.downcast_iter().map(op))
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
    ChunkedArray::from_chunk_iter(lhs.name().clone(), iter)
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
    ChunkedArray::try_from_chunk_iter(lhs.name().clone(), iter)
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
        return ChunkedArray::with_chunk(lhs.name().clone(), V::full_null_array(len));
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
            array.with_validity_broadcast_typed(validity.map(PlBitmap::into_flat_or_scalar))
        });
    ChunkedArray::from_chunk_iter(lhs.name().clone(), iter)
}

/// Apply elementwise binary function which produces string, amortising allocations.
///
/// Currently unused within Polars itself, but it's a useful utility for plugin authors.
#[inline]
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
            let mut builder = PlUtf8ViewArrayBuilder::with_capacity(lhs_arr.len());
            lhs_arr
                .iter()
                .zip(rhs_arr.iter())
                .for_each(|(lhs_opt, rhs_opt)| match (lhs_opt, rhs_opt) {
                    // SAFETY: every value pushed is the UTF-8 of the `String` it was built in.
                    (None, _) | (_, None) => unsafe { builder.inner_mut() }.push_null(),
                    (Some(lhs_val), Some(rhs_val)) => {
                        buf.clear();
                        op(lhs_val, rhs_val, &mut buf);
                        unsafe { builder.inner_mut() }.push_value(buf.as_bytes())
                    },
                });
            builder.freeze()
        });
    ChunkedArray::from_chunk_iter(lhs.name().clone(), iter)
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
    name: PlSmallStr,
) -> ChunkedArray<V>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
    F: FnMut(&T::Array, &U::Array) -> Arr,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let iter = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| {
            let ret = op(lhs_arr, rhs_arr);
            mask_with_inputs(ret, lhs_arr.validity(), rhs_arr.validity())
        });
    ChunkedArray::from_chunk_iter(name, iter)
}

/// Applies an elementwise binary kernel written against the [`flat`](polars_array::broadcast)
/// representation, masking off every element that either side has a null at:
/// [`binary_mut_values`] over buffers.
///
/// `op` must be elementwise; two [`scalar`](polars_array::broadcast) chunks reach it as the single
/// element each repeats, and the result is repeated in turn. See [`elementwise_binary_flat`].
#[inline]
pub fn binary_elementwise_mut_values_flat<T, U, V, F, Arr>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
    name: PlSmallStr,
) -> ChunkedArray<V>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
    F: FnMut(&Flat<T::Array>, &Flat<U::Array>) -> Arr,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let iter = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| {
            let ret = elementwise_binary_flat(lhs_arr, rhs_arr, &mut op);
            mask_with_inputs(ret, lhs_arr.validity(), rhs_arr.validity())
        });
    ChunkedArray::from_chunk_iter(name, iter)
}

/// Applies a kernel that produces `Array` types.
#[inline]
pub fn binary_mut_with_options<T, U, V, F, Arr>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
    name: PlSmallStr,
) -> ChunkedArray<V>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
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
    name: PlSmallStr,
) -> Result<ChunkedArray<V>, E>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
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

/// Applies an elementwise binary kernel written against the [`flat`](polars_array::broadcast)
/// representation. This is the `(flat, flat)` path of [`apply_binary_kernel_broadcast`].
///
/// `op` must be elementwise; two [`scalar`](polars_array::broadcast) chunks reach it as the single
/// element each repeats, and the result is repeated in turn. See [`elementwise_binary_flat`].
pub fn binary_elementwise_kernel_flat<T, U, V, F, Arr>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
    name: PlSmallStr,
) -> ChunkedArray<V>
where
    T: PolarsDataType,
    U: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
    F: FnMut(&Flat<T::Array>, &Flat<U::Array>) -> Arr,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let iter = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| elementwise_binary_flat(lhs_arr, rhs_arr, &mut op));
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
    Arr: StaticArray,
    F: FnMut(&T::Array, &U::Array) -> Arr,
{
    binary_mut_with_options(lhs, rhs, op, lhs.name().clone())
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
    Arr: StaticArray,
    F: FnMut(L::Array, R::Array) -> Arr,
{
    let name = lhs.name().clone();
    let (lhs, rhs) = align_chunks_binary_owned(lhs, rhs);
    let iter = lhs
        .downcast_into_iter()
        .zip(rhs.downcast_into_iter())
        .map(|(lhs_arr, rhs_arr)| op(lhs_arr, rhs_arr));
    ChunkedArray::from_chunk_iter(name, iter)
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
    Arr: StaticArray,
    F: FnMut(&T::Array, &U::Array) -> Result<Arr, E>,
    E: Error,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let iter = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| op(lhs_arr, rhs_arr));
    ChunkedArray::try_from_chunk_iter(lhs.name().clone(), iter)
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
    F: FnMut(&T::Array, &U::Array) -> PlArrayRef,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let chunks = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| op(lhs_arr, rhs_arr))
        .collect();

    let mut ca = lhs.copy_with_chunks(chunks);

    let mut retain_flags = StatisticsFlags::empty();
    use StatisticsFlags as F;
    retain_flags.set(F::IS_SORTED_ANY, keep_sorted);
    retain_flags.set(F::CAN_FAST_EXPLODE_LIST, keep_fast_explode);
    ca.retain_flags_from(lhs.as_ref(), retain_flags);

    ca
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
    F: FnMut(&T::Array, &U::Array) -> Result<PlArrayRef, E>,
    E: Error,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let chunks = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| op(lhs_arr, rhs_arr))
        .collect::<Result<Vec<_>, E>>()?;
    let mut ca = lhs.copy_with_chunks(chunks);

    let mut retain_flags = StatisticsFlags::empty();
    use StatisticsFlags as F;
    retain_flags.set(F::IS_SORTED_ANY, keep_sorted);
    retain_flags.set(F::CAN_FAST_EXPLODE_LIST, keep_fast_explode);
    ca.retain_flags_from(lhs.as_ref(), retain_flags);

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
    ChunkedArray::try_from_chunk_iter(ca1.name().clone(), iter)
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
    ChunkedArray::from_chunk_iter(ca1.name().clone(), iter)
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
    let length = broadcast_height(lhs.len(), rhs.len())
        .expect("cannot apply operation on arrays of different lengths");

    // A side that repeats one element is read once and handed to the unary walk over the other.
    // A column of one element repeats it by definition, so this subsumes the length-one case.
    match (lhs.scalar_value(), rhs.scalar_value()) {
        (Some(a), _) if rhs.len() == length => {
            unary_elementwise(rhs, |b| op(a.clone(), b)).with_name(lhs.name().clone())
        },
        (_, Some(b)) => unary_elementwise(lhs, |a| op(a, b.clone())),
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
    let length = broadcast_height(lhs.len(), rhs.len())
        .expect("cannot apply operation on arrays of different lengths");

    // See [`broadcast_binary_elementwise`] for what the two scalar arms are.
    match (lhs.scalar_value(), rhs.scalar_value()) {
        (Some(a), _) if rhs.len() == length => {
            Ok(try_unary_elementwise(rhs, |b| op(a.clone(), b))?.with_name(lhs.name().clone()))
        },
        (_, Some(b)) => try_unary_elementwise(lhs, |a| op(a, b.clone())),
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
    let length = broadcast_height(lhs.len(), rhs.len())
        .expect("cannot apply operation on arrays of different lengths");

    if lhs.null_count() == lhs.len() || rhs.null_count() == rhs.len() {
        return ChunkedArray::with_chunk(lhs.name().clone(), V::full_null_array(length));
    }

    // See [`broadcast_binary_elementwise`] for what the two scalar arms are. Neither side is
    // fully null here, so a side that repeats one element repeats a value rather than a null.
    match (lhs.scalar_value(), rhs.scalar_value()) {
        (Some(Some(a)), _) if rhs.len() == length => {
            unary_elementwise_values(rhs, |b| op(a.clone(), b)).with_name(lhs.name().clone())
        },
        (_, Some(Some(b))) => unary_elementwise_values(lhs, |a| op(a, b.clone())),
        _ => binary_elementwise_values(lhs, rhs, op),
    }
}

/// Applies a binary kernel to the chunks of `lhs` and `rhs`, handing a side that is one value
/// repeated to the kernel written for a repeated operand.
///
/// The chunks reach the kernel in whatever representation they are in, for it to read as it sees
/// fit: this is the helper for a kernel that handles the [`scalar`](polars_array::broadcast)
/// representation itself. See [`apply_binary_kernel_broadcast_flat`] for one that does not.
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
    let length = broadcast_height(lhs.len(), rhs.len())
        .expect("cannot apply operation on arrays of different lengths");

    // The broadcast paths come first, so that a side that is one value repeated reaches the
    // kernel as that value. What is left is the path over two chunks of the same length.
    let out = match (lhs.scalar_value(), rhs.scalar_value()) {
        // broadcast right path
        (_, Some(rhs)) if lhs.len() == length => match rhs {
            None => ChunkedArray::<O>::with_chunk(name.clone(), O::full_null_array(length)),
            Some(rhs) => unary_kernel(lhs, |arr| rhs_broadcast_kernel(arr, rhs.clone())),
        },
        (Some(lhs), _) => match lhs {
            None => ChunkedArray::<O>::with_chunk(name.clone(), O::full_null_array(length)),
            Some(lhs) => unary_kernel(rhs, |arr| lhs_broadcast_kernel(lhs.clone(), arr)),
        },
        _ => binary_mut_with_options(lhs, rhs, |lhs, rhs| kernel(lhs, rhs), name.clone()),
    };
    out.with_name(name.clone())
}

/// [`apply_binary_kernel_broadcast`] for a kernel that takes its chunks by value.
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
    let length = broadcast_height(lhs.len(), rhs.len())
        .expect("cannot apply operation on arrays of different lengths");

    // See [`apply_binary_kernel_broadcast`] for what the two broadcast paths are. The scalar
    // check is a borrow, so it is taken before either side is moved into a kernel.
    let lhs_repeats = lhs.scalar_value().is_some();
    let rhs_repeats = rhs.scalar_value().is_some();

    // broadcast right path
    let out = if rhs_repeats && lhs.len() == length {
        match rhs.scalar_value().unwrap() {
            None => ChunkedArray::<O>::with_chunk(name.clone(), O::full_null_array(length)),
            Some(rhs) => unary_kernel_owned(lhs, |arr| rhs_broadcast_kernel(arr, rhs.clone())),
        }
    } else if lhs_repeats {
        match lhs.scalar_value().unwrap() {
            None => ChunkedArray::<O>::with_chunk(name.clone(), O::full_null_array(length)),
            Some(lhs) => unary_kernel_owned(rhs, |arr| lhs_broadcast_kernel(lhs.clone(), arr)),
        }
    } else {
        binary_owned(lhs, rhs, kernel)
    };
    out.with_name(name)
}

/// [`apply_binary_kernel_broadcast`] for a kernel written against the
/// [`flat`](polars_array::broadcast) representation, which a [`scalar`](polars_array::broadcast)
/// chunk is written out to reach.
///
/// The kernel is elementwise, so two scalar chunks reach it as the single element each repeats
/// and the answer is repeated in turn; a scalar chunk that meets a flat one is what gets written
/// out. Prefer [`apply_binary_kernel_broadcast`] for a kernel that reads its chunks itself.
pub fn apply_binary_kernel_broadcast_flat<'l, 'r, L, R, O, K, LK, RK>(
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
    K: Fn(&Flat<L::Array>, &Flat<R::Array>) -> O::Array,
    LK: Fn(L::Physical<'l>, &Flat<R::Array>) -> O::Array,
    RK: Fn(&Flat<L::Array>, R::Physical<'r>) -> O::Array,
{
    let name = lhs.name();
    let length = broadcast_height(lhs.len(), rhs.len())
        .expect("cannot apply operation on arrays of different lengths");

    // The broadcast paths come first, so that a side repeating a single value reaches the kernel
    // as that value rather than being written out. What is left is the `(flat, flat)` path.
    let out = match (lhs.scalar_value(), rhs.scalar_value()) {
        // broadcast right path
        (_, Some(rhs)) if lhs.len() == length => match rhs {
            None => ChunkedArray::<O>::with_chunk(name.clone(), O::full_null_array(length)),
            Some(rhs) => {
                unary_elementwise_kernel_flat(lhs, |arr| rhs_broadcast_kernel(arr, rhs.clone()))
            },
        },
        (Some(lhs), _) => match lhs {
            None => ChunkedArray::<O>::with_chunk(name.clone(), O::full_null_array(length)),
            Some(lhs) => {
                unary_elementwise_kernel_flat(rhs, |arr| lhs_broadcast_kernel(lhs.clone(), arr))
            },
        },
        _ => binary_elementwise_kernel_flat(lhs, rhs, |lhs, rhs| kernel(lhs, rhs), name.clone()),
    };
    out.with_name(name.clone())
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;
    use polars_array::{PlBooleanArray, PlPrimitiveArray};

    use super::*;
    use crate::prelude::{BooleanChunked, Int32Chunked};

    fn chunk(arr: PlPrimitiveArray<i32>) -> Int32Chunked {
        Int32Chunked::with_chunk(PlSmallStr::EMPTY, arr)
    }

    /// The kernel below is elementwise, so it may be handed a single element; it counts its calls
    /// to show that a scalar chunk reaches it once rather than over a written-out buffer.
    fn is_seven(ca: &Int32Chunked, calls: &mut usize) -> BooleanChunked {
        unary_elementwise_kernel_flat(ca, |arr| {
            *calls += 1;
            PlBooleanArray::from_iter(arr.values_iter().map(|v| *v == 7))
        })
    }

    #[test]
    fn a_scalar_chunk_reaches_an_elementwise_kernel_as_one_element() {
        let mut calls = 0;
        let out = is_seven(&chunk(PlPrimitiveArray::new_scalar(7i32, 5)), &mut calls);

        assert_eq!(calls, 1);
        assert_eq!(out.len(), 5);
        assert!(out.downcast_get(0).unwrap().is_scalar(), "{out:?}");
        assert!(out.iter().all(|v| v == Some(true)));

        // The single element is the one the kernel answers for, whatever the answer is.
        let mut calls = 0;
        let out = is_seven(&chunk(PlPrimitiveArray::new_scalar(3i32, 5)), &mut calls);
        assert_eq!(calls, 1);
        assert!(out.iter().all(|v| v == Some(false)));
    }

    /// A kernel that carries the input's mask over, which is what makes the null of an all-null
    /// scalar chunk the answer for every element of the result.
    fn is_seven_keeping_nulls(ca: &Int32Chunked, calls: &mut usize) -> BooleanChunked {
        unary_elementwise_kernel_flat(ca, |arr| {
            *calls += 1;
            let values = Bitmap::from_iter(arr.values_iter().map(|v| *v == 7));
            PlBooleanArray::new(values, arr.len(), arr.validity().cloned())
        })
    }

    #[test]
    fn a_scalar_null_chunk_answers_null_for_every_element() {
        let mut calls = 0;
        let out = is_seven_keeping_nulls(
            &chunk(PlPrimitiveArray::<i32>::new_full_null(5)),
            &mut calls,
        );

        assert_eq!(calls, 1);
        assert_eq!(out.len(), 5);
        assert_eq!(out.null_count(), 5);
        assert!(out.downcast_get(0).unwrap().is_scalar(), "{out:?}");

        // A scalar chunk of a value keeps its mask over too.
        let mut calls = 0;
        let out = is_seven_keeping_nulls(&chunk(PlPrimitiveArray::new_scalar(7i32, 5)), &mut calls);
        assert_eq!(calls, 1);
        assert_eq!(out.null_count(), 0);
        assert!(out.iter().all(|v| v == Some(true)));
    }

    #[test]
    fn a_flat_chunk_is_handed_over_element_by_element() {
        let mut calls = 0;
        let out = is_seven(
            &chunk(PlPrimitiveArray::from_vec(vec![7i32, 3, 7])),
            &mut calls,
        );

        assert_eq!(calls, 1);
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            [Some(true), Some(false), Some(true)],
        );
    }

    /// A chunk with a scalar values buffer but a flat validity mask is neither: there is no one
    /// element standing for the rest, so it is written out as before.
    #[test]
    fn a_half_scalar_chunk_is_written_out() {
        let arr = PlPrimitiveArray::new_scalar(7i32, 4)
            .with_validity(Some(Bitmap::from_iter([true, false, true, true])));
        assert!(!arr.is_scalar() && !arr.is_flat());

        let mut calls = 0;
        let out = is_seven(&chunk(arr), &mut calls);

        assert_eq!(calls, 1);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v == Some(true)));
    }

    #[test]
    fn two_scalar_chunks_reach_a_binary_kernel_as_one_element_each() {
        let lhs = chunk(PlPrimitiveArray::new_scalar(7i32, 5));
        let rhs = chunk(PlPrimitiveArray::new_scalar(3i32, 5));

        let mut calls = 0;
        let out: BooleanChunked = binary_elementwise_kernel_flat(
            &lhs,
            &rhs,
            |a, b| {
                calls += 1;
                PlBooleanArray::from_iter(a.values_iter().zip(b.values_iter()).map(|(l, r)| l > r))
            },
            PlSmallStr::EMPTY,
        );

        assert_eq!(calls, 1);
        assert_eq!(out.len(), 5);
        assert!(out.downcast_get(0).unwrap().is_scalar(), "{out:?}");
        assert!(out.iter().all(|v| v == Some(true)));

        // One flat side leaves nothing to repeat: the kernel sees every element of both.
        let rhs = chunk(PlPrimitiveArray::from_vec(vec![3i32, 9, 3, 9, 3]));
        let out: BooleanChunked = binary_elementwise_kernel_flat(
            &lhs,
            &rhs,
            |a, b| {
                PlBooleanArray::from_iter(a.values_iter().zip(b.values_iter()).map(|(l, r)| l > r))
            },
            PlSmallStr::EMPTY,
        );
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            [Some(true), Some(false), Some(true), Some(false), Some(true)],
        );
    }
}
