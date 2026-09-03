#![allow(unsafe_op_in_unsafe_fn)]
use std::error::Error;

use polars_array::bitmap::combine_validities_and;
use polars_array::builder::StaticArrayBuilder;
use polars_array::{
    Flat, PlArray, PlBitmap, PlBitmapRef, PlUtf8ViewArrayBuilder, StaticArray, as_flat,
};

use crate::prelude::PlArrayRef;

/// An Arrow array, which is what the kernels that hand a [`Series`] back produce: the [`DataType`]
/// of the result is read off the Arrow data type when the chunks are imported.
///
/// [`DataType`]: crate::prelude::DataType
type ArrowArrayRef = arrow::array::ArrayRef;
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;

use crate::chunked_array::flags::StatisticsFlags;
use crate::datatypes::{ArrayCollectIterExt, ArrayFromIter};
use crate::prelude::{ChunkedArray, PolarsDataType, Series, StringChunked};
use crate::utils::{align_chunks_binary, align_chunks_binary_owned, align_chunks_ternary};

/// Returns `ret` masked off wherever either input has a null, on top of its own mask.
///
/// The three masks all cover the same elements. A scalar one among them is combined as the single
/// bit it stands for rather than written out first, so a kernel over two chunks that are fully
/// null hands back a result that is still `O(1)` in memory.
#[inline]
fn mask_with_inputs<A: StaticArray>(
    ret: A,
    lhs: Option<PlBitmapRef<'_>>,
    rhs: Option<PlBitmapRef<'_>>,
) -> A {
    // The combined mask covers the inputs' elements, which is what `ret` holds too: a kernel that
    // handed back a result of a different height than its inputs panics here, or in the setter
    // below.
    let inputs = combine_validities_and(lhs, rhs);
    let validity = combine_validities_and(inputs.as_ref().map(PlBitmap::as_ref), ret.validity());
    ret.with_validity_broadcast_typed(validity.map(PlBitmap::into_flat_or_scalar))
}

/// The height of the output of an elementwise operation over two columns of these lengths, or
/// `None` if the two do not broadcast.
///
/// A column either has the height of the output or is a single element repeated to meet it. This
/// is [`binary_output_height`](crate::binary_output_height) without the error, leaving the caller
/// to say what a length mismatch is — the operations here answer it with a panic, as they always
/// have.
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

/// Applies a kernel written against the [`flat`](polars_array::broadcast) representation.
///
/// This is [`unary_kernel`] for a kernel that reads the backing buffers directly: every chunk
/// reaches it as a [`Flat`] array, written out first if it was not laid out flat — see
/// [`as_flat`].
#[inline]
pub fn unary_kernel_flat<T, V, F, Arr>(ca: &ChunkedArray<T>, mut op: F) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
    F: FnMut(&Flat<T::Array>) -> Arr,
{
    let iter = ca.downcast_iter().map(|arr| op(&as_flat(arr)));
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

/// Applies an owned kernel written against the [`flat`](polars_array::broadcast) representation.
///
/// This is [`unary_kernel_owned`] for a kernel that reads the backing buffers directly — see
/// [`unary_kernel_flat`].
#[inline]
pub fn unary_kernel_owned_flat<T, V, F, Arr>(ca: ChunkedArray<T>, mut op: F) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
    F: FnMut(Flat<T::Array>) -> Arr,
{
    let name = ca.name().clone();
    let iter = ca
        .downcast_into_iter()
        .map(|arr| op(StaticArray::to_flat(&arr)));
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

/// Applies a kernel written against the [`flat`](polars_array::broadcast) representation, putting
/// the input's validity mask back on the result.
///
/// This is [`unary_mut_values`] for a kernel that reads the backing buffers directly.
#[inline]
pub fn unary_mut_values_flat<T, V, F, Arr>(ca: &ChunkedArray<T>, mut op: F) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
    F: FnMut(&Flat<T::Array>) -> Arr,
{
    let iter = ca.downcast_iter().map(|arr| {
        op(&as_flat(arr))
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

/// Applies a kernel written against the [`flat`](polars_array::broadcast) representation,
/// leaving the result's own validity mask alone.
///
/// This is [`unary_mut_with_options`] for a kernel that reads the backing buffers directly.
#[inline]
pub fn unary_mut_with_options_flat<T, V, F, Arr>(ca: &ChunkedArray<T>, mut op: F) -> ChunkedArray<V>
where
    T: PolarsDataType,
    V: PolarsDataType<Array = Arr>,
    Arr: StaticArray,
    F: FnMut(&Flat<T::Array>) -> Arr,
{
    let iter = ca.downcast_iter().map(|arr| op(&as_flat(arr)));
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

/// Applies a binary kernel written against the [`flat`](polars_array::broadcast) representation,
/// masking off every element that either side has a null at.
///
/// This is [`binary_mut_values`] for a kernel that reads the backing buffers directly — see
/// [`binary_kernel_flat`].
#[inline]
pub fn binary_mut_values_flat<T, U, V, F, Arr>(
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
            let ret = op(&as_flat(lhs_arr), &as_flat(rhs_arr));
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

/// Applies a binary kernel written against the [`flat`](polars_array::broadcast) representation.
///
/// This is the `(flat, flat)` path of a broadcasting operation — see
/// [`apply_binary_kernel_broadcast`], which dispatches to it once the two sides are known to be
/// of the same length.
pub fn binary_kernel_flat<T, U, V, F, Arr>(
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
        .map(|(lhs_arr, rhs_arr)| op(&as_flat(lhs_arr), &as_flat(rhs_arr)));
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

pub fn try_unary_to_series<T, F>(ca: &ChunkedArray<T>, op: F) -> PolarsResult<Series>
where
    T: PolarsDataType,
    F: FnMut(&T::Array) -> PolarsResult<ArrowArrayRef>,
{
    let chunks = ca
        .downcast_iter()
        .map(op)
        .collect::<PolarsResult<Vec<_>>>()?;
    Series::try_from((ca.name().clone(), chunks))
}

pub fn binary_to_series<T, U, F>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
) -> PolarsResult<Series>
where
    T: PolarsDataType,
    U: PolarsDataType,
    F: FnMut(&T::Array, &U::Array) -> ArrowArrayRef,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let chunks = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| op(lhs_arr, rhs_arr))
        .collect::<Vec<_>>();
    Series::try_from((lhs.name().clone(), chunks))
}

pub fn try_binary_to_series<T, U, F>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<U>,
    mut op: F,
) -> PolarsResult<Series>
where
    T: PolarsDataType,
    U: PolarsDataType,
    F: FnMut(&T::Array, &U::Array) -> PolarsResult<ArrowArrayRef>,
{
    let (lhs, rhs) = align_chunks_binary(lhs, rhs);
    let chunks = lhs
        .downcast_iter()
        .zip(rhs.downcast_iter())
        .map(|(lhs_arr, rhs_arr)| op(lhs_arr, rhs_arr))
        .collect::<PolarsResult<Vec<_>>>()?;
    Series::try_from((lhs.name().clone(), chunks))
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

    // A side that repeats one element is read once and handed to the unary walk over the other,
    // which is what keeps a column that stands for a single value from being read `length` times.
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
    K: Fn(&Flat<L::Array>, &Flat<R::Array>) -> O::Array,
    LK: Fn(L::Physical<'l>, &Flat<R::Array>) -> O::Array,
    RK: Fn(&Flat<L::Array>, R::Physical<'r>) -> O::Array,
{
    let name = lhs.name();
    let length = broadcast_height(lhs.len(), rhs.len())
        .expect("cannot apply operation on arrays of different lengths");

    // The broadcast paths come first: a side that repeats a single value is handed to the kernel
    // as the value it is, so that the other side reaches a kernel whose second argument is flat
    // without that repeated value ever being written out. A column of one element repeats it by
    // definition, and one whose only chunk is in the [`scalar`](polars_array::broadcast)
    // representation repeats it over its whole height — so this subsumes the length-one case
    // rather than sitting beside it, and `as_flat` no longer has a scalar chunk to expand.
    //
    // The `(flat, flat)` path is what is left: two columns of the same height, neither of which
    // stands for a single value, which is what the specialized kernels are written for.
    let out = match (lhs.scalar_value(), rhs.scalar_value()) {
        // broadcast right path
        (_, Some(rhs)) if lhs.len() == length => match rhs {
            None => ChunkedArray::<O>::with_chunk(name.clone(), O::full_null_array(length)),
            Some(rhs) => unary_kernel_flat(lhs, |arr| rhs_broadcast_kernel(arr, rhs.clone())),
        },
        (Some(lhs), _) => match lhs {
            None => ChunkedArray::<O>::with_chunk(name.clone(), O::full_null_array(length)),
            Some(lhs) => unary_kernel_flat(rhs, |arr| lhs_broadcast_kernel(lhs.clone(), arr)),
        },
        _ => binary_kernel_flat(lhs, rhs, |lhs, rhs| kernel(lhs, rhs), name.clone()),
    };
    out.with_name(name.clone())
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
    K: Fn(Flat<L::Array>, Flat<R::Array>) -> O::Array,
    for<'a> LK: Fn(L::Physical<'a>, Flat<R::Array>) -> O::Array,
    for<'a> RK: Fn(Flat<L::Array>, R::Physical<'a>) -> O::Array,
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
            Some(rhs) => unary_kernel_owned(lhs, |arr| {
                rhs_broadcast_kernel(StaticArray::to_flat(&arr), rhs.clone())
            }),
        }
    } else if lhs_repeats {
        match lhs.scalar_value().unwrap() {
            None => ChunkedArray::<O>::with_chunk(name.clone(), O::full_null_array(length)),
            Some(lhs) => unary_kernel_owned(rhs, |arr| {
                lhs_broadcast_kernel(lhs.clone(), StaticArray::to_flat(&arr))
            }),
        }
    } else {
        binary_owned(lhs, rhs, |lhs, rhs| {
            kernel(StaticArray::to_flat(&lhs), StaticArray::to_flat(&rhs))
        })
    };
    out.with_name(name)
}
