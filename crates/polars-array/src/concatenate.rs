//! Concatenation of arrays into a single array of the same physical representation.

use std::ops::Deref;

use arrow::array::View;
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_utils::slice_broadcast_iter::SliceBroadcastIter;

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::{
    PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlFixedSizeBinaryArray, PlFixedSizeListArray,
    PlListArray, PlNullArray, PlPrimitiveArray, PlStructArray, PlUtf8ViewArray,
    with_match_pl_primitive_array_type,
};

/// The elements of a slice, in order, or one element over and over, each as what it stands for.
struct MappedBroadcastIter<'a, 'm, T: ?Sized, S> {
    /// The elements left to yield, each of which is mapped to what it stands for.
    inner: SliceBroadcastIter<'a, S>,
    /// What an element of the slice stands for, which is yielded in its place.
    ///
    /// This is borrowed rather than held, since a map composed onto another one — the fields of
    /// the arrays this yields, say — closes over that one, and a closure has to live somewhere
    /// for as long as the repetition it maps.
    map: &'m dyn Fn(&'a S) -> &'a T,
}

impl<'a, 'm, T: ?Sized, S> MappedBroadcastIter<'a, 'm, T, S> {
    /// What the elements of `slice` stand for, in order, once.
    fn new(slice: &'a [S], map: &'m dyn Fn(&'a S) -> &'a T) -> Self {
        Self {
            inner: SliceBroadcastIter::new(slice),
            map,
        }
    }

    /// What `element` stands for, `repeats` times over.
    ///
    /// # Panics
    /// Panics if `repeats` is more than half a `usize`, which no concatenation has the memory to
    /// back: an element of it is at least a bit wide.
    fn repeat(element: &'a S, repeats: usize, map: &'m dyn Fn(&'a S) -> &'a T) -> Self {
        assert!(
            repeats <= usize::MAX >> 1,
            "the number of arrays to concatenate overflows a `usize`",
        );
        Self {
            inner: SliceBroadcastIter::repeat(element, repeats),
            map,
        }
    }

    /// What the distinct elements left to yield stand for, in the order they are repeated in.
    ///
    /// A repeated element is one element however often it is left to be yielded, including no
    /// times at all: what is distinct about it does not depend on the repetition.
    fn distinct(&self) -> impl ExactSizeIterator<Item = &'a T> + use<'a, 'm, T, S> {
        let distinct = match self.inner.clone().split() {
            Ok(slice) => slice,
            Err((element, _)) => std::slice::from_ref(element),
        };
        distinct.iter().map(self.map)
    }

    /// What one element of the slice stands for, which is where a map composed onto this one
    /// starts.
    fn get(&self, element: &'a S) -> &'a T {
        (self.map)(element)
    }

    /// How many times over the elements left to yield repeat the distinct ones.
    ///
    /// This is the `repeats` a repetition was built with, until iteration starts eating into it,
    /// and one for a slice that is walked once — zero when it holds nothing to walk, which keeps
    /// the total length this is multiplied into at zero.
    fn repeats(&self) -> usize {
        match self.inner.clone().split() {
            Ok(slice) => usize::from(!slice.is_empty()),
            Err((_, repeats)) => repeats,
        }
    }

    /// The same elements left to yield, standing for something else — the arrays this yields
    /// downcast to their concrete type, say, or one of their fields.
    fn mapped<'n, U: ?Sized>(
        &self,
        map: &'n dyn Fn(&'a S) -> &'a U,
    ) -> MappedBroadcastIter<'a, 'n, U, S> {
        MappedBroadcastIter {
            inner: self.inner.clone(),
            map,
        }
    }
}

impl<T: ?Sized, S> Clone for MappedBroadcastIter<'_, '_, T, S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            map: self.map,
        }
    }
}

impl<'a, T: ?Sized, S> Iterator for MappedBroadcastIter<'a, '_, T, S> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(self.map)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    /// Hoists the branch on how the elements are walked out of the loop, the way the inner
    /// iterator does; `for_each` and `collect` route through here.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let map = self.map;
        self.inner.fold(init, |acc, element| f(acc, map(element)))
    }
}

impl<T: ?Sized, S> ExactSizeIterator for MappedBroadcastIter<'_, '_, T, S> {}

/// An element of a slice of pointers to arrays stands for the array it points to: the array
/// itself where the slice is one of arrays, and the one in the box where it is one of boxes.
fn pointee<S: Deref<Target = T>, T: ?Sized>(element: &S) -> &T {
    element
}

/// Concatenates `arrays`, in order, into a single array of their common [`PlArrayType`].
///
/// This dispatches to the typed function for that array type; see the [module docs](self) for when
/// the result keeps the scalar representation instead of being materialized.
///
/// Only the array types of the arrays themselves are checked, since that is all a
/// [`PlArrayType`] carries: what a nested array is taken over is checked where it is concatenated,
/// which is not reached when a fast path hands back one of the arrays unchanged.
///
/// # Errors
/// This function errors if `arrays` is empty, if the arrays do not all have the same
/// [`PlArrayType`], if primitive arrays of that type do not all have the same element type, or if
/// the values of nested arrays do not concatenate.
///
/// # Example
/// ```
/// use polars_array::concatenate::concatenate;
/// use polars_array::{PlArray, PlPrimitiveArray};
///
/// let lhs = PlPrimitiveArray::from_vec(vec![1i32, 2]);
/// let rhs = PlPrimitiveArray::from_iter([Some(3i32), None]);
/// let concatenated = concatenate(&[&lhs, &rhs]).unwrap();
///
/// assert_eq!(concatenated.len(), 4);
/// assert_eq!(concatenated.null_count(), 1);
/// assert_eq!(
///     concatenated
///         .as_any()
///         .downcast_ref::<PlPrimitiveArray<i32>>()
///         .unwrap()
///         .iter()
///         .collect::<Vec<_>>(),
///     [Some(1), Some(2), Some(3), None],
/// );
/// ```
pub fn concatenate(arrays: &[&dyn PlArray]) -> PolarsResult<Box<dyn PlArray>> {
    concatenate_impl(MappedBroadcastIter::new(arrays, &pointee))
}

/// Concatenates `repeats` copies of `array` into a single array of its [`PlArrayType`].
///
/// This is what repeating the elements of an array as a whole means, as opposed to repeating one
/// of them: [`PlArray::new_from_index`] is the latter, and is what this dispatches to when there
/// is only the one element to repeat. Repeating no copies, or an array that holds no elements,
/// yields an empty array; a single copy is the array itself.
///
/// Outside those cases this is `O(len * repeats)`: the elements of the result come from more than
/// one copy, so what a copy stands for has to be written out — see the [module docs](self) for
/// when it does not.
///
/// # Errors
/// This function errors if the values of a nested array do not concatenate with themselves, which
/// they always do unless an outside implementation of [`PlArray`] misreports its array type.
///
/// # Example
/// ```
/// use polars_array::concatenate::concatenate_repeated;
/// use polars_array::{PlArray, PlPrimitiveArray};
///
/// let arr = PlPrimitiveArray::from_vec(vec![1i32, 2]);
/// let repeated = concatenate_repeated(&arr, 3).unwrap();
///
/// assert_eq!(repeated.len(), 6);
/// assert_eq!(
///     repeated
///         .as_any()
///         .downcast_ref::<PlPrimitiveArray<i32>>()
///         .unwrap()
///         .iter()
///         .collect::<Vec<_>>(),
///     [Some(1), Some(2), Some(1), Some(2), Some(1), Some(2)],
/// );
///
/// // A single element is repeated without being written out: a billion copies of it are `O(1)`.
/// let one = PlPrimitiveArray::from_vec(vec![7i32]);
/// let repeated = concatenate_repeated(&one, 1_000_000_000).unwrap();
///
/// assert_eq!(repeated.len(), 1_000_000_000);
/// ```
pub fn concatenate_repeated(array: &dyn PlArray, repeats: usize) -> PolarsResult<Box<dyn PlArray>> {
    // No copy holds an element, so the result is empty. It is still sliced out of the array
    // itself, which is what carries anything a `PlArrayType` does not — the values of a list
    // array, the fields of a struct array.
    if repeats == 0 || array.is_empty() {
        return Ok(array.sliced(0, 0));
    }

    // There is nothing to repeat the array over: the one copy is the array.
    if repeats == 1 {
        return Ok(array.to_boxed());
    }

    // Copies of a single element are copies of the array, which every array repeats in `O(1)` but
    // a struct array, and none of them writes the element out.
    if array.len() == 1 {
        return Ok(array.new_from_index(0, repeats));
    }

    concatenate_impl(MappedBroadcastIter::repeat(&array, repeats, &pointee))
}

/// Concatenates the arrays `iter` yields, in order, into a single array of their common
/// [`PlArrayType`], which is what [`concatenate`] and [`concatenate_repeated`] both come down to.
fn concatenate_impl<'a, S>(
    iter: MappedBroadcastIter<'a, '_, dyn PlArray, S>,
) -> PolarsResult<Box<dyn PlArray>> {
    let mut distinct = iter.distinct();
    let Some(first) = distinct.next() else {
        polars_bail!(InvalidOperation: "cannot concatenate an empty list of arrays");
    };

    let array_type = first.array_type();
    for array in distinct {
        polars_ensure!(
            array.array_type() == array_type,
            InvalidOperation:
            "cannot concatenate an array of type {:?} with one of type {:?}",
            array_type, array.array_type(),
        );
    }

    match array_type {
        PlArrayType::Boolean => {
            let map = downcast_map::<PlBooleanArray, _>(&iter, array_type)?;
            Ok(Box::new(concatenate_boolean_impl(iter.mapped(&map))))
        },
        PlArrayType::Primitive(primitive) => {
            with_match_pl_primitive_array_type!(first, |T| concatenate_primitive_as::<T, _>(&iter))
                .flatten()
                .ok_or_else(|| {
                    polars_err!(
                        InvalidOperation:
                        "cannot concatenate arrays of primitive type {:?} that do not all have \
                         the same element type",
                        primitive,
                    )
                })
        },
        PlArrayType::Binary => {
            let map = downcast_map::<PlBinaryArray, _>(&iter, array_type)?;
            Ok(Box::new(concatenate_binary_impl(iter.mapped(&map))))
        },
        PlArrayType::BinaryView => {
            let map = downcast_map::<PlBinaryViewArray, _>(&iter, array_type)?;
            Ok(Box::new(concatenate_binview_impl(iter.mapped(&map))))
        },
        PlArrayType::Utf8View => {
            // A string array is a binary view array whose bytes are UTF-8, so it concatenates the
            // same way; what the wrapper adds is the promise, which is re-established below.
            let map = downcast_map::<PlUtf8ViewArray, _>(&iter, array_type)?;
            let bytes_map = |array: &'a S| map(array).as_binview();
            let concatenated = concatenate_binview_impl(iter.mapped(&bytes_map));
            // SAFETY: every element came from a `PlUtf8ViewArray`, so every one is valid UTF-8.
            Ok(Box::new(unsafe {
                PlUtf8ViewArray::from_binview_unchecked(concatenated)
            }))
        },
        PlArrayType::FixedSizeBinary => {
            let map = downcast_map::<PlFixedSizeBinaryArray, _>(&iter, array_type)?;
            Ok(Box::new(concatenate_fixed_size_binary_impl(
                iter.mapped(&map),
            )?))
        },
        PlArrayType::Struct => {
            let map = downcast_map::<PlStructArray, _>(&iter, array_type)?;
            Ok(Box::new(concatenate_struct_impl(iter.mapped(&map))?))
        },
        PlArrayType::List => {
            let map = downcast_map::<PlListArray, _>(&iter, array_type)?;
            Ok(Box::new(concatenate_list_impl(iter.mapped(&map))?))
        },
        PlArrayType::FixedSizeList => {
            let map = downcast_map::<PlFixedSizeListArray, _>(&iter, array_type)?;
            Ok(Box::new(concatenate_fixed_size_list_impl(
                iter.mapped(&map),
            )?))
        },
        PlArrayType::Null => {
            let map = downcast_map::<PlNullArray, _>(&iter, array_type)?;
            Ok(Box::new(concatenate_null_impl(iter.mapped(&map))))
        },
        x @ PlArrayType::Object { .. } => {
            panic!("polars-array: no concatenate impl for {x:?}")
        },
    }
}

/// Concatenates the validity masks of `arrays`, in order, into the mask of their concatenation.
///
/// Returns `None` when no element of any array is null, which is the mask of a fully valid array,
/// and a scalar (one-bit) mask when every element of every array is null. Otherwise the mask is
/// materialized, one bit per element, which is `O(total length)`.
///
/// # Example
/// ```
/// use polars_array::concatenate::concatenate_validities;
/// use polars_array::PlPrimitiveArray;
///
/// // Nothing is null, so there is no mask to build.
/// let valid = PlPrimitiveArray::from_vec(vec![1i32, 2]);
/// assert!(concatenate_validities(&[&valid, &valid]).is_none());
///
/// // Everything is null, so a single bit stands for a billion elements.
/// let null = PlPrimitiveArray::<i32>::new_full_null(1_000_000_000);
/// let validity = concatenate_validities(&[&null, &null]).unwrap();
/// assert_eq!(validity.len(), 1);
/// ```
pub fn concatenate_validities<A: PlArray + ?Sized>(arrays: &[&A]) -> Option<Bitmap> {
    let map = &pointee;
    let iter = MappedBroadcastIter::new(arrays, map);
    let (length, null_count) = total_length_and_null_count(&iter);
    concatenate_validities_with(iter, length, null_count)
}

/// [`concatenate_validities`], for a caller that has already counted the elements and the nulls.
fn concatenate_validities_with<A: PlArray + ?Sized, S>(
    iter: MappedBroadcastIter<'_, '_, A, S>,
    length: usize,
    null_count: usize,
) -> Option<Bitmap> {
    if null_count == 0 {
        return None;
    }
    if null_count == length {
        // Every element is null, so a single shared bit is the whole mask.
        return Some(Bitmap::new_zeroed(1));
    }

    let mut validity = BitmapBuilder::with_capacity(length);
    for array in iter {
        let null_count = array.null_count();
        if null_count == 0 {
            validity.extend_constant(array.len(), true);
        } else if null_count == array.len() {
            validity.extend_constant(array.len(), false);
        } else {
            // The mask is neither all-set nor all-unset, so it cannot be scalar: this is a clone.
            validity.extend_from_bitmap(&array.validity().unwrap().to_flat());
        }
    }
    validity.into_opt_validity()
}

/// Concatenates `arrays`, in order, into a single [`PlPrimitiveArray`].
///
/// See the [module docs](self) for when the result keeps the scalar representation instead of
/// being materialized. Concatenating no arrays yields an empty array.
///
/// # Example
/// ```
/// use polars_array::concatenate::concatenate_primitive;
/// use polars_array::PlPrimitiveArray;
///
/// // Two arrays standing for the same repeated element concatenate in `O(1)`.
/// let lhs = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);
/// let rhs = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);
/// let concatenated = concatenate_primitive(&[&lhs, &rhs]);
///
/// assert_eq!(concatenated.len(), 2_000_000_000);
/// assert_eq!(concatenated.scalar_values(), Some(7));
/// assert!(concatenated.is_scalar());
/// ```
pub fn concatenate_primitive<T: NativeType>(
    arrays: &[&PlPrimitiveArray<T>],
) -> PlPrimitiveArray<T> {
    concatenate_primitive_impl(MappedBroadcastIter::new(arrays, &pointee))
}

/// [`concatenate_primitive`], over the arrays `iter` yields.
fn concatenate_primitive_impl<T: NativeType, S>(
    iter: MappedBroadcastIter<'_, '_, PlPrimitiveArray<T>, S>,
) -> PlPrimitiveArray<T> {
    if let Some(array) = only_non_empty(&iter) {
        return array.clone();
    }

    let (length, null_count) = total_length_and_null_count(&iter);

    // Every element is null, so every value is undetermined and none of them has to be written.
    if length > 0 && null_count == length {
        return PlPrimitiveArray::new_full_null(length);
    }

    // Every array stands for the same repeated element, which the result repeats over all of them.
    if let Some(element) = shared_element(&iter, PlPrimitiveArray::scalar_value) {
        return match element {
            Some(value) => PlPrimitiveArray::new_scalar(value, length),
            None => PlPrimitiveArray::new_full_null(length),
        };
    }

    let validity = concatenate_validities_with(iter.clone(), length, null_count);

    let mut values = Vec::with_capacity(length);
    for array in iter.filter(|array| !array.is_empty()) {
        if let Some(array_values) = array.flat_values() {
            values.extend_from_slice(array_values.as_slice());
        } else if let Some(value) = array.scalar_values() {
            values.extend(std::iter::repeat_n(value, array.len()));
        }
    }

    // SAFETY: the values hold one slot per element of the concatenation, and the mask is the one
    // `concatenate_validities_with` built for that many elements.
    unsafe { PlPrimitiveArray::new_unchecked(Buffer::from(values), length, validity) }
}

/// Concatenates `arrays`, in order, into a single [`PlBooleanArray`].
///
/// See the [module docs](self) for when the result keeps the scalar representation instead of
/// being materialized. Concatenating no arrays yields an empty array.
pub fn concatenate_boolean(arrays: &[&PlBooleanArray]) -> PlBooleanArray {
    concatenate_boolean_impl(MappedBroadcastIter::new(arrays, &pointee))
}

/// [`concatenate_boolean`], over the arrays `iter` yields.
fn concatenate_boolean_impl<S>(
    iter: MappedBroadcastIter<'_, '_, PlBooleanArray, S>,
) -> PlBooleanArray {
    if let Some(array) = only_non_empty(&iter) {
        return array.clone();
    }

    let (length, null_count) = total_length_and_null_count(&iter);

    // Every element is null, so every value is undetermined and none of them has to be written.
    if length > 0 && null_count == length {
        return PlBooleanArray::new_full_null(length);
    }

    // Every array stands for the same repeated element, which the result repeats over all of them.
    if let Some(element) = shared_element(&iter, PlBooleanArray::scalar_value) {
        return match element {
            Some(value) => PlBooleanArray::new_scalar(value, length),
            None => PlBooleanArray::new_full_null(length),
        };
    }

    let validity = concatenate_validities_with(iter.clone(), length, null_count);

    let mut values = BitmapBuilder::with_capacity(length);
    for array in iter.filter(|array| !array.is_empty()) {
        if let Some(array_values) = array.flat_values() {
            values.extend_from_bitmap(array_values);
        } else if let Some(value) = array.scalar_values() {
            values.extend_constant(array.len(), value);
        }
    }

    // SAFETY: the values hold one bit per element of the concatenation, and the mask is the one
    // `concatenate_validities_with` built for that many elements.
    unsafe { PlBooleanArray::new_unchecked(values.freeze(), length, validity) }
}

/// Concatenates `arrays`, in order, into a single [`PlBinaryArray`] over the bytes their elements
/// cover.
///
/// The offsets are rebased onto the concatenated bytes, and the bytes an input holds outside its
/// own offsets — which slicing leaves behind — are dropped. See the [module docs](self) for when
/// the result keeps the scalar representation; outside those cases the offsets of the result are
/// flat, so an input whose own offsets are scalar has its element written out once per element it
/// stands for, since no two elements of a flat binary array can cover the same range.
/// Concatenating no arrays yields an empty array.
///
/// # Example
/// ```
/// use polars_array::concatenate::concatenate_binary;
/// use polars_array::PlBinaryArray;
///
/// let lhs = PlBinaryArray::from_values_iter([b"foo".as_slice()]);
/// let rhs = PlBinaryArray::from_values_iter([b"bar".as_slice(), b"baz"]);
/// let concatenated = concatenate_binary(&[&lhs, &rhs]);
///
/// assert_eq!(concatenated.len(), 3);
/// assert_eq!(concatenated.value(2), b"baz");
///
/// // Two arrays standing for the same repeated element concatenate in `O(value.len())`.
/// let arr = PlBinaryArray::new_scalar(b"ab", 1_000_000_000);
/// let concatenated = concatenate_binary(&[&arr, &arr]);
///
/// assert_eq!(concatenated.len(), 2_000_000_000);
/// assert_eq!(concatenated.scalar_offsets(), Some(0..2));
/// assert!(concatenated.is_scalar());
/// ```
pub fn concatenate_binary(arrays: &[&PlBinaryArray]) -> PlBinaryArray {
    concatenate_binary_impl(MappedBroadcastIter::new(arrays, &pointee))
}

/// [`concatenate_binary`], over the arrays `iter` yields.
fn concatenate_binary_impl<S>(
    iter: MappedBroadcastIter<'_, '_, PlBinaryArray, S>,
) -> PlBinaryArray {
    if let Some(array) = only_non_empty(&iter) {
        return array.clone();
    }

    let (length, null_count) = total_length_and_null_count(&iter);

    // There is no element to concatenate, however many arrays there are, so there are no bytes to
    // keep around for one either.
    if length == 0 {
        return PlBinaryArray::new_empty();
    }

    // Every element is null, so every value is undetermined and none of them has to be written out.
    if null_count == length {
        return PlBinaryArray::new_full_null(length);
    }

    // Every array stands for the same repeated element, which the result repeats over all of them.
    if let Some(element) = shared_element(&iter, PlBinaryArray::scalar_value) {
        return match element {
            Some(value) => PlBinaryArray::new_scalar(value, length),
            // An element shared as null makes every element of every array null, which the branch
            // above has already returned.
            None => PlBinaryArray::new_full_null(length),
        };
    }

    let validity = concatenate_validities_with(iter.clone(), length, null_count);

    // One pass over the distinct arrays is what the concatenation is made of: the copies of them
    // cover the very same bytes over again, so the pass is repeated rather than built twice.
    let mut values: Vec<u8> = Vec::new();
    let mut offsets: Vec<u64> = Vec::with_capacity(length + 1);
    offsets.push(0);

    for array in iter.distinct().filter(|array| !array.is_empty()) {
        let end = offsets[offsets.len() - 1];

        if let Some(array_offsets) = array.flat_offsets() {
            let (first, last) = (array_offsets[0], array_offsets[array.len()]);
            // SAFETY: the offsets are ordered and bounded by the length of the values.
            values.extend_from_slice(unsafe {
                array.values().get_unchecked(first as usize..last as usize)
            });
            offsets.extend(
                array_offsets[1..]
                    .iter()
                    .map(|offset| end + (offset - first)),
            );
        } else if let Some(element) = array.scalar_values() {
            // Every element of the array covers the same bytes, which the result writes out once
            // per element: no two elements of a flat binary array can share a range.
            values.reserve(element.len() * array.len());
            for _ in 0..array.len() {
                values.extend_from_slice(element);
            }

            let value_length = element.len() as u64;
            offsets.extend((1..=array.len() as u64).map(|i| end + i * value_length));
        }
    }

    // Every copy of the distinct arrays holds the same elements over again, which is the pass that
    // was just built, repeated — its offsets rebased onto the bytes the copies before it wrote.
    let pass_bytes = values.len();
    let pass_elements = offsets.len() - 1;
    for copy in 1..iter.repeats() {
        values.extend_from_within(..pass_bytes);

        let base = pass_bytes as u64 * copy as u64;
        for i in 1..=pass_elements {
            offsets.push(base + offsets[i]);
        }
    }

    // SAFETY: the offsets are the lengths of the byte strings of every array in order, so they are
    // ordered and end at the length of the bytes; the mask covers that many elements.
    unsafe {
        PlBinaryArray::new_unchecked(
            Buffer::from(values),
            Buffer::from(offsets),
            length,
            validity,
        )
    }
}

/// Concatenates `arrays`, in order, into a single [`PlBinaryViewArray`] over the data buffers of
/// all of them.
///
/// The bytes of the elements are never copied: the data buffers of the arrays are appended to one
/// another and the views are rebased onto them, so this costs a view per element rather than the
/// bytes of one. See the [module docs](self) for when the result keeps the scalar representation
/// instead of being materialized. Concatenating no arrays yields an empty array.
///
/// # Panics
/// Panics if the arrays hold more data buffers between them than a view can index.
///
/// # Example
/// ```
/// use polars_array::concatenate::concatenate_binview;
/// use polars_array::PlBinaryViewArray;
///
/// let lhs = PlBinaryViewArray::from_values_iter([b"foo".as_slice()]);
/// let rhs = PlBinaryViewArray::from_values_iter([b"bar".as_slice(), b"baz"]);
/// let concatenated = concatenate_binview(&[&lhs, &rhs]);
///
/// assert_eq!(concatenated.len(), 3);
/// assert_eq!(concatenated.value(2), b"baz");
/// ```
pub fn concatenate_binview(arrays: &[&PlBinaryViewArray]) -> PlBinaryViewArray {
    concatenate_binview_impl(MappedBroadcastIter::new(arrays, &pointee))
}

/// [`concatenate_binview`], over the arrays `iter` yields.
fn concatenate_binview_impl<S>(
    iter: MappedBroadcastIter<'_, '_, PlBinaryViewArray, S>,
) -> PlBinaryViewArray {
    if let Some(array) = only_non_empty(&iter) {
        return array.clone();
    }

    let (length, null_count) = total_length_and_null_count(&iter);

    // There is no element to concatenate, however many arrays there are, so there is no data
    // buffer to keep around for one either.
    if length == 0 {
        return PlBinaryViewArray::new_empty();
    }

    // Every element is null, so every value is undetermined and none of them has to be written.
    if null_count == length {
        return PlBinaryViewArray::new_full_null(length);
    }

    // Every array stands for the same repeated element, which the result repeats over all of them.
    if let Some(element) = shared_element(&iter, PlBinaryViewArray::scalar_value) {
        return match element {
            Some(value) => PlBinaryViewArray::new_scalar(value, length),
            None => PlBinaryViewArray::new_full_null(length),
        };
    }

    let validity = concatenate_validities_with(iter.clone(), length, null_count);

    // One pass over the distinct arrays is what the concatenation is made of: the copies of them
    // hold the very same views over the very same data buffers, so neither is built twice.
    let mut buffers: Vec<Buffer<u8>> = Vec::new();
    let mut views: Vec<View> = Vec::with_capacity(length);
    for array in iter.distinct().filter(|array| !array.is_empty()) {
        buffers.extend(array.data_buffers().iter().cloned());
        let end = u32::try_from(buffers.len())
            .expect("the concatenation holds more data buffers than a view can index");
        let buffer_offset = end - array.data_buffers().len() as u32;

        // A view that inlines its bytes reads no data buffer, so it is already what it stands for.
        let rebase = |mut view: View| {
            if !view.is_inline() {
                view.buffer_idx += buffer_offset;
            }
            view
        };

        if let Some(array_views) = array.flat_views() {
            views.extend(array_views.iter().copied().map(rebase));
        } else if let Some(view) = array.scalar_views() {
            views.extend(std::iter::repeat_n(rebase(view), array.len()));
        }
    }

    // Every copy of the distinct arrays holds the same elements over again, which is the pass that
    // was just built, repeated.
    let pass = views.len();
    for _ in 1..iter.repeats() {
        views.extend_from_within(..pass);
    }

    // SAFETY: the views hold one slot per element, each rebased onto the buffers of the array it
    // came from, and the mask covers that many elements.
    unsafe {
        PlBinaryViewArray::new_unchecked(
            Buffer::from(views),
            buffers.into_iter().collect(),
            length,
            validity,
        )
    }
}

/// Concatenates `arrays`, in order, into a single [`PlFixedSizeBinaryArray`] over the bytes their
/// elements cover.
///
/// Every array has to agree on the width, which is what the elements of the result are as wide as.
/// See the [module docs](self) for when the result keeps the scalar representation; outside those
/// cases the values of the result are flat, so an input whose own values are scalar has its element
/// written out once per element it stands for, since the values of a flat fixed size binary array
/// hold one width per element.
///
/// # Errors
/// This function errors if `arrays` is empty, since there is then no width for the elements of the
/// result to have, or if the arrays do not all have the same width.
///
/// # Example
/// ```
/// use polars_array::concatenate::concatenate_fixed_size_binary;
/// use polars_array::PlFixedSizeBinaryArray;
///
/// // Two arrays standing for the same repeated element concatenate in `O(width)`.
/// let arr = PlFixedSizeBinaryArray::new_scalar(b"ab", 1_000_000_000);
/// let concatenated = concatenate_fixed_size_binary(&[&arr, &arr]).unwrap();
///
/// assert_eq!(concatenated.len(), 2_000_000_000);
/// assert_eq!(concatenated.scalar_values(), Some(b"ab".as_slice()));
/// assert!(concatenated.is_scalar());
/// ```
pub fn concatenate_fixed_size_binary(
    arrays: &[&PlFixedSizeBinaryArray],
) -> PolarsResult<PlFixedSizeBinaryArray> {
    concatenate_fixed_size_binary_impl(MappedBroadcastIter::new(arrays, &pointee))
}

/// [`concatenate_fixed_size_binary`], over the arrays `iter` yields.
fn concatenate_fixed_size_binary_impl<S>(
    iter: MappedBroadcastIter<'_, '_, PlFixedSizeBinaryArray, S>,
) -> PolarsResult<PlFixedSizeBinaryArray> {
    let mut distinct = iter.distinct();
    let Some(first) = distinct.next() else {
        polars_bail!(
            InvalidOperation:
            "cannot concatenate an empty list of fixed size binary arrays: there is no width for \
             the elements of the result to have"
        );
    };

    let width = first.width();
    for array in distinct {
        polars_ensure!(
            array.width() == width,
            InvalidOperation:
            "cannot concatenate a fixed size binary array of width {} with one of width {}",
            width, array.width(),
        );
    }

    if let Some(array) = only_non_empty(&iter) {
        return Ok(array.clone());
    }

    let (length, null_count) = total_length_and_null_count(&iter);

    // Every element is null, so every value is undetermined and none of them has to be written
    // out: a zeroed element of the width stands in for the one they all cover.
    if length > 0 && null_count == length {
        return Ok(PlFixedSizeBinaryArray::new_full_null(width, length));
    }

    // Every array stands for the same repeated element, which the result repeats over all of them.
    if let Some(element) = shared_element(&iter, PlFixedSizeBinaryArray::scalar_value) {
        return Ok(match element {
            Some(element) => PlFixedSizeBinaryArray::new_scalar(element, length),
            // An element shared as null makes every element of every array null, which the branch
            // above has already returned.
            None => PlFixedSizeBinaryArray::new_full_null(width, length),
        });
    }

    let validity = concatenate_validities_with(iter.clone(), length, null_count);

    // One pass over the distinct arrays is what the concatenation is made of: the copies of them
    // cover the very same bytes over again, so the pass is repeated rather than built twice.
    let flat_len = length
        .checked_mul(width)
        .expect("the values of the concatenation overflow a `usize`");
    let mut values: Vec<u8> = Vec::with_capacity(flat_len);
    for array in iter.distinct().filter(|array| !array.is_empty()) {
        if let Some(array_values) = array.flat_values() {
            values.extend_from_slice(array_values.as_slice());
        } else if let Some(element) = array.scalar_values() {
            // Scalar values are the one element every element covers, which the result writes out
            // once per element it stands for.
            for _ in 0..array.len() {
                values.extend_from_slice(element);
            }
        }
    }

    let pass = values.len();
    for _ in 1..iter.repeats() {
        values.extend_from_within(..pass);
    }

    // SAFETY: every array contributed the width of each of its elements, so the values hold one
    // width per element of the concatenation, and the mask covers that many elements.
    Ok(unsafe {
        PlFixedSizeBinaryArray::new_unchecked(Buffer::from(values), width, length, validity)
    })
}

/// Concatenates `arrays`, in order, into a single [`PlFixedSizeListArray`] over the concatenation
/// of the values their lists reach.
///
/// Every array has to agree on the width, which is what the lists of the result are as wide as.
/// See the [module docs](self) for when the result keeps the scalar representation; outside those
/// cases the values of the result are flat, so an input whose own values are scalar has its
/// element written out once per element it stands for, since the values of a flat fixed size list
/// array hold one width per element.
///
/// # Errors
/// This function errors if `arrays` is empty, since there is no values array to take the lists of
/// the result over, if the arrays do not all have the same width, or if the values do not
/// concatenate.
pub fn concatenate_fixed_size_list(
    arrays: &[&PlFixedSizeListArray],
) -> PolarsResult<PlFixedSizeListArray> {
    concatenate_fixed_size_list_impl(MappedBroadcastIter::new(arrays, &pointee))
}

/// [`concatenate_fixed_size_list`], over the arrays `iter` yields.
fn concatenate_fixed_size_list_impl<S>(
    iter: MappedBroadcastIter<'_, '_, PlFixedSizeListArray, S>,
) -> PolarsResult<PlFixedSizeListArray> {
    let mut distinct = iter.distinct();
    let Some(first) = distinct.next() else {
        polars_bail!(
            InvalidOperation:
            "cannot concatenate an empty list of fixed size list arrays: there is no values array              to take the lists of the result over"
        );
    };

    let width = first.width();
    for array in distinct {
        polars_ensure!(
            array.width() == width,
            InvalidOperation:
            "cannot concatenate a fixed size list array of width {} with one of width {}",
            width, array.width(),
        );
    }

    if let Some(array) = only_non_empty(&iter) {
        return Ok(array.clone());
    }

    let (length, null_count) = total_length_and_null_count(&iter);

    // The values of some element, which is what the undetermined values of a fully null result are
    // taken from: every array agrees on the width, so any element of any of them is as wide as the
    // result needs. There is one whenever the concatenation holds an element at all.
    let undetermined = || {
        let array = iter
            .distinct()
            .find(|array| !array.is_empty())
            .expect("a concatenation that holds elements has an array that holds one");
        // SAFETY: the array was just seen to hold an element.
        unsafe { array.value_unchecked(0) }
    };

    // Every element is null, so every list is undetermined and none of them has to be written out.
    if length > 0 && null_count == length {
        return Ok(PlFixedSizeListArray::new_full_null(undetermined(), length));
    }

    // Every array stands for the same repeated element, which the result repeats over all of them.
    if let Some(element) = shared_element(&iter, PlFixedSizeListArray::scalar_value) {
        return Ok(match element {
            Some(element) => PlFixedSizeListArray::new_scalar(element, length),
            // An element shared as null makes every element of every array null, which the branch
            // above has already returned.
            None => PlFixedSizeListArray::new_full_null(undetermined(), length),
        });
    }

    // The values of each array are what its elements cover, which is the whole values array of a
    // flat one; scalar values are the one element they share, which the result writes out once per
    // element it stands for.
    let mut values = Vec::with_capacity(iter.len());
    for array in iter.clone() {
        if let Some(array_values) = array.flat_values() {
            values.push(array_values.to_boxed());
        } else if let Some(element) = array.scalar_values() {
            // Concatenating the element with copies of itself is what repeats it, and that keeps
            // the values scalar when the element is itself a single repeated value.
            values.push(concatenate_repeated(element, array.len())?);
        }
    }

    // The values of the arrays are concatenated through the boxes they were sliced into.
    let values = concatenate_impl(MappedBroadcastIter::new(values.as_slice(), &pointee))?;
    let validity = concatenate_validities_with(iter, length, null_count);

    // SAFETY: every array contributed the width of each of its elements, so the values hold one
    // width per element of the concatenation, and the mask covers that many elements.
    Ok(unsafe { PlFixedSizeListArray::new_unchecked(values, width, length, validity) })
}

/// Concatenates `arrays`, in order, into a single [`PlNullArray`].
///
/// A null array is nothing but a length, so this is `O(1)`: the lengths are added up and every
/// element of the result is null, like every element of every input. Concatenating no arrays
/// yields an empty array.
pub fn concatenate_null(arrays: &[&PlNullArray]) -> PlNullArray {
    concatenate_null_impl(MappedBroadcastIter::new(arrays, &pointee))
}

/// [`concatenate_null`], over the arrays `iter` yields.
fn concatenate_null_impl<S>(iter: MappedBroadcastIter<'_, '_, PlNullArray, S>) -> PlNullArray {
    PlNullArray::new(total_length_and_null_count(&iter).0)
}

/// Concatenates `arrays`, in order, into a single [`PlStructArray`], concatenating each field with
/// the field at the same position of every other array.
///
/// The fields carry their own representation, so a struct array of nothing but scalar fields
/// concatenates in `O(1)` exactly when its fields do. Concatenating no arrays yields an empty
/// array without fields.
///
/// # Errors
/// This function errors if the arrays do not all have the same number of fields, or if any of the
/// fields do not concatenate.
pub fn concatenate_struct(arrays: &[&PlStructArray]) -> PolarsResult<PlStructArray> {
    concatenate_struct_impl(MappedBroadcastIter::new(arrays, &pointee))
}

/// [`concatenate_struct`], over the arrays `iter` yields.
fn concatenate_struct_impl<S>(
    iter: MappedBroadcastIter<'_, '_, PlStructArray, S>,
) -> PolarsResult<PlStructArray> {
    let mut distinct = iter.distinct();
    let Some(first) = distinct.next() else {
        return Ok(PlStructArray::new_empty());
    };

    let num_fields = first.num_fields();
    for array in distinct {
        polars_ensure!(
            array.num_fields() == num_fields,
            InvalidOperation:
            "cannot concatenate a struct array of {} fields with one of {} fields",
            num_fields, array.num_fields(),
        );
    }

    if let Some(array) = only_non_empty(&iter) {
        return Ok(array.clone());
    }

    let (length, null_count) = total_length_and_null_count(&iter);

    // The field at each position is concatenated over the very same repetition, so the fields of
    // the result are as long as the concatenation itself.
    let fields = (0..num_fields)
        .map(|i| concatenate_impl(iter.mapped(&|array| iter.get(array).field(i))))
        .collect::<PolarsResult<Vec<_>>>()?;

    // Every row is null, so a single shared bit is the whole mask. The fields are kept as they
    // are: their values are undetermined, and it is the mask that makes every row null.
    if length > 0 && null_count == length {
        return Ok(PlStructArray::new_full_null(fields, length));
    }

    let validity = concatenate_validities_with(iter, length, null_count);

    // SAFETY: every field is the concatenation of the fields at that position, so it is as long as
    // the arrays together, and the mask is the flat one built for that many elements.
    Ok(unsafe { PlStructArray::new_unchecked(fields, length, validity) })
}

/// Concatenates `arrays`, in order, into a single [`PlListArray`] over the concatenation of the
/// values their lists reach.
///
/// The offsets are rebased onto that values array, and the values an input holds outside its own
/// offsets — which slicing leaves behind — are dropped. See the [module docs](self) for when the
/// result keeps the scalar representation; outside those cases the offsets of the result are flat,
/// so an input whose own offsets are scalar has its element written out once per element it stands
/// for, since no two elements of a flat list array can cover the same range.
///
/// # Errors
/// This function errors if `arrays` is empty, since there is no values array to take the lists of
/// the result over, or if the values do not concatenate.
pub fn concatenate_list(arrays: &[&PlListArray]) -> PolarsResult<PlListArray> {
    concatenate_list_impl(MappedBroadcastIter::new(arrays, &pointee))
}

/// [`concatenate_list`], over the arrays `iter` yields.
fn concatenate_list_impl<S>(
    iter: MappedBroadcastIter<'_, '_, PlListArray, S>,
) -> PolarsResult<PlListArray> {
    let Some(first) = iter.distinct().next() else {
        polars_bail!(
            InvalidOperation:
            "cannot concatenate an empty list of list arrays: there is no values array to take \
             the lists of the result over"
        );
    };

    if let Some(array) = only_non_empty(&iter) {
        return Ok(array.clone());
    }

    let (length, null_count) = total_length_and_null_count(&iter);

    // Every element is null, so every list is undetermined and none of them has to be written out;
    // the values array is what determines the type of the lists, of which the first one will do.
    if length > 0 && null_count == length {
        return Ok(PlListArray::new_full_null(
            first.values().to_boxed(),
            length,
        ));
    }

    // Every array stands for the same repeated element, which the result repeats over all of them.
    if let Some(element) = shared_element(&iter, PlListArray::scalar_value) {
        return Ok(match element {
            Some(element) => PlListArray::new_scalar(element, length),
            None => PlListArray::new_full_null(first.values().to_boxed(), length),
        });
    }

    // The values of each array are sliced to what its offsets reach, so that the offsets of the
    // result can be rebased onto their concatenation.
    let mut values = Vec::with_capacity(iter.len());
    let mut offsets = Vec::with_capacity(length + 1);
    offsets.push(0);

    let mut end = 0;
    for array in iter.clone() {
        if array.is_empty() {
            // No element of the array is reachable, but its values are still what the type of its
            // lists is taken from, which every array has to agree on.
            values.push(array.values().sliced(0, 0));
        } else if let Some(array_offsets) = array.flat_offsets() {
            let (first, last) = (array_offsets[0], array_offsets[array.len()]);
            values.push(
                array
                    .values()
                    .sliced(first as usize, (last - first) as usize),
            );
            offsets.extend(
                array_offsets[1..]
                    .iter()
                    .map(|offset| end + (offset - first)),
            );
            end += last - first;
        } else if let Some(range) = array.scalar_offsets() {
            // Every element of the array covers the same range, which the result writes out once
            // per element: concatenating the element with copies of itself is what repeats it, and
            // that keeps the values scalar when the element is itself a single repeated value.
            let element = array.values().sliced(range.start, range.len());
            values.push(concatenate_repeated(&*element, array.len())?);

            let value_length = range.len() as u64;
            offsets.extend((1..=array.len() as u64).map(|i| end + i * value_length));
            end += value_length * array.len() as u64;
        }
    }

    // The values of the arrays are concatenated through the boxes they were sliced into.
    let values = concatenate_impl(MappedBroadcastIter::new(values.as_slice(), &pointee))?;
    let validity = concatenate_validities_with(iter, length, null_count);

    // SAFETY: the offsets are the lengths of the lists of every array in order, so they are
    // ordered and end at the length of the values; the mask covers that many elements.
    Ok(unsafe { PlListArray::new_unchecked(values, Buffer::from(offsets), length, validity) })
}

/// The one array that holds every element of the concatenation, if the others are all empty.
///
/// Returns the first array when they are all empty: the concatenation is empty as well, and the
/// first array is what carries anything a [`PlArrayType`] does not — the values of a list array,
/// the fields of a struct array. Returns `None` when more than one array holds elements, and for
/// no arrays at all, which has no array to hand back.
///
/// An array that is repeated holds every element of the concatenation only if the repetition is of
/// a single copy: the other copies hold the very same elements over again.
fn only_non_empty<'a, A: PlArray + ?Sized, S>(
    iter: &MappedBroadcastIter<'a, '_, A, S>,
) -> Option<&'a A> {
    let mut non_empty = iter.distinct().filter(|array| !array.is_empty());
    match non_empty.next() {
        Some(array) => (non_empty.next().is_none() && iter.repeats() == 1).then_some(array),
        None => iter.distinct().next(),
    }
}

/// The total number of elements the arrays `iter` yields hold, and how many of them are null.
///
/// # Panics
/// Panics if the total length overflows a `usize`, which the scalar representation makes possible
/// without the memory to back it.
fn total_length_and_null_count<A: PlArray + ?Sized, S>(
    iter: &MappedBroadcastIter<'_, '_, A, S>,
) -> (usize, usize) {
    let mut length = 0usize;
    let mut null_count = 0usize;
    for array in iter.distinct() {
        length = length
            .checked_add(array.len())
            .expect("the total length of the concatenation overflows a `usize`");
        null_count += array.null_count();
    }

    // Every copy of the distinct arrays holds the same elements over again.
    let repeats = iter.repeats();
    (
        length
            .checked_mul(repeats)
            .expect("the total length of the concatenation overflows a `usize`"),
        // No more null than long, so this cannot overflow where the length does not.
        null_count * repeats,
    )
}

/// The element every element of every array equals, if `element` sees one for each of them and
/// they all agree.
///
/// The arrays that hold no elements are skipped: they have no element to disagree. Returns `None`
/// when no array holds elements, since there is then no element to repeat. Only the distinct
/// arrays are looked at: a repeated array agrees with itself.
fn shared_element<'a, A: PlArray, T: PartialEq, S>(
    iter: &MappedBroadcastIter<'a, '_, A, S>,
    element: impl Fn(&'a A) -> Option<T>,
) -> Option<T> {
    let mut shared = None;
    for array in iter.distinct().filter(|array| !array.is_empty()) {
        let element = element(array)?;
        match &shared {
            Some(shared) if *shared != element => return None,
            Some(_) => {},
            None => shared = Some(element),
        }
    }
    shared
}

/// What the arrays `iter` yields stand for as an `A`, or `None` if that is not the concrete array
/// type of every one of them.
///
/// The arrays are downcast where they are yielded, so what this maps is the very slice they are
/// already in: walking them as an `A` materializes no slice of its own.
fn try_downcast_map<'a, A: PlArray, S>(
    iter: &MappedBroadcastIter<'a, '_, dyn PlArray, S>,
) -> Option<impl Fn(&'a S) -> &'a A> {
    if !iter.distinct().all(|array| array.as_any().is::<A>()) {
        return None;
    }

    let map = iter.map;
    // Every distinct array was just seen to be an `A`, and the repetition yields nothing else.
    Some(move |element: &'a S| map(element).as_any().downcast_ref::<A>().unwrap())
}

/// [`try_downcast_map`], for the `array_type` every array reports, which guarantees `A` is their
/// concrete array type — unless one of them is an outside implementation of [`PlArray`] reporting
/// an array type that is not its own.
fn downcast_map<'a, A: PlArray, S>(
    iter: &MappedBroadcastIter<'a, '_, dyn PlArray, S>,
    array_type: PlArrayType,
) -> PolarsResult<impl Fn(&'a S) -> &'a A> {
    try_downcast_map(iter).ok_or_else(|| {
        polars_err!(
            InvalidOperation:
            "cannot concatenate arrays that report the array type {:?} but are not all of the \
             same concrete array type",
            array_type,
        )
    })
}

/// Concatenates the arrays `iter` yields as [`PlPrimitiveArray<T>`], or returns `None` if that is
/// not the concrete array type of every one of them.
///
/// The element type this is called with is the one of the first array, which the others have to
/// agree on: their equal [`PlArrayType`] does not make them agree, since a
/// [`PrimitiveType`](crate::PrimitiveType) does not pin an element type down.
fn concatenate_primitive_as<'a, T: NativeType, S>(
    iter: &MappedBroadcastIter<'a, '_, dyn PlArray, S>,
) -> Option<Box<dyn PlArray>> {
    let map = try_downcast_map::<PlPrimitiveArray<T>, _>(iter)?;
    Some(Box::new(concatenate_primitive_impl(iter.mapped(&map))))
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn flat_arrays_are_appended_in_order() {
        let lhs = PlPrimitiveArray::from_vec(vec![1i32, 2]);
        let rhs = PlPrimitiveArray::from_iter([Some(3i32), None, Some(5)]);
        let concatenated = concatenate_primitive(&[&lhs, &rhs]);

        assert_eq!(concatenated.len(), 5);
        assert_eq!(concatenated.null_count(), 1);
        assert!(concatenated.is_flat());
        assert_eq!(
            concatenated.iter().collect::<Vec<_>>(),
            [Some(1), Some(2), Some(3), None, Some(5)],
        );
    }

    #[test]
    fn arrays_of_the_same_repeated_element_stay_scalar() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);
        // A flat array of length one stands for a single repeated element as well.
        let single = PlPrimitiveArray::from_vec(vec![7i32]);
        let concatenated = concatenate_primitive(&[&scalar, &single, &scalar]);

        assert_eq!(concatenated.len(), 2_000_000_001);
        assert!(concatenated.is_scalar());
        assert_eq!(concatenated.scalar_value(), Some(Some(7)));

        // A different element on either side leaves nothing to repeat.
        let sixes = PlPrimitiveArray::new_scalar(6i32, 3);
        let sevens = PlPrimitiveArray::new_scalar(7i32, 2);
        let concatenated = concatenate_primitive(&[&sixes, &sevens]);

        assert!(concatenated.is_flat());
        assert_eq!(
            concatenated.flat_values().unwrap().as_slice(),
            [6, 6, 6, 7, 7]
        );
    }

    #[test]
    fn the_trait_object_dispatches_to_every_array_type() {
        let arrays: Vec<Box<dyn PlArray>> = vec![
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
            Box::new(PlBooleanArray::from_vec(vec![true, false, true])),
            Box::new(PlNullArray::new(3)),
            Box::new(PlBinaryArray::from_values_iter([
                b"foo".as_slice(),
                b"",
                b"bar",
            ])),
            Box::new(PlFixedSizeBinaryArray::from_vec(
                vec![1u8, 2, 3, 4, 5, 6],
                2,
            )),
            Box::new(PlStructArray::from_fields(vec![Box::new(
                PlPrimitiveArray::from_vec(vec![1i32, 2, 3]),
            )])),
            Box::new(PlListArray::from_offsets(
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
                Buffer::from(vec![0u64, 1, 2, 3]),
            )),
        ];

        for array in &arrays {
            let concatenated = concatenate(&[&**array, &**array]).unwrap();

            assert_eq!(concatenated.array_type(), array.array_type());
            assert_eq!(concatenated.len(), 6);
            assert_eq!(&concatenated.sliced(0, 3), array);
            assert_eq!(&concatenated.sliced(3, 3), array);
        }
    }
}
