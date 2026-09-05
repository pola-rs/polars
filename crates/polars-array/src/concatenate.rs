//! Concatenation of arrays into a single array of the same physical representation.

use arrow::array::View;
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::PlBitmap;
use crate::broadcast::ArrayRepr;
use crate::primitive::bytes;
use crate::{
    PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlFixedSizeBinaryArray, PlFixedSizeListArray,
    PlListArray, PlNullArray, PlPrimitiveArray, PlStructArray, PlUtf8ViewArray,
    with_match_pl_primitive_array_type,
};

/// The arrays a concatenation is made of: `count` distinct arrays, read by index, which the
/// concatenation lays down `repeats` times over.
///
/// The arrays are reached through a `dyn Fn` rather than held as a slice so that this type is
/// generic in the array type alone. Every caller points at its arrays in its own way — a slice of
/// references, a slice of boxes, one array repeated, one field of every array of another list —
/// and carrying that shape as a type parameter is what used to fan every concatenation out into a
/// copy per shape, on top of the copy per element type it already costs.
struct ArrayList<'a, 'f, A: ?Sized> {
    /// The array at an index below [`Self::count`].
    get: &'f dyn Fn(usize) -> &'a A,
    /// How many distinct arrays there are to read.
    count: usize,
    /// How many times over the concatenation lays the distinct arrays down.
    repeats: usize,
}

impl<'a, 'f, A: ?Sized> ArrayList<'a, 'f, A> {
    /// The `count` arrays `get` reads, in order, once.
    fn new(get: &'f dyn Fn(usize) -> &'a A, count: usize) -> Self {
        Self {
            get,
            count,
            repeats: 1,
        }
    }

    /// The `count` arrays `get` reads, in order, `repeats` times over.
    fn repeated(get: &'f dyn Fn(usize) -> &'a A, count: usize, repeats: usize) -> Self {
        Self {
            get,
            count,
            repeats,
        }
    }

    /// The distinct arrays, in the order they are repeated in.
    fn distinct(&self) -> impl ExactSizeIterator<Item = &'a A> + Clone + use<'a, 'f, A> {
        let get = self.get;
        (0..self.count).map(get)
    }

    /// Every array of the concatenation, in order, the repeats included.
    fn iter(&self) -> impl ExactSizeIterator<Item = &'a A> + Clone + use<'a, 'f, A> {
        let (get, count) = (self.get, self.count);
        // There is no index to wrap around where there is no array to read, and the empty range
        // below never reaches the remainder.
        (0..self.len()).map(move |index| get(index % count))
    }

    /// The array at `index`, which is where a reader composed onto this one starts.
    fn at(&self, index: usize) -> &'a A {
        (self.get)(index)
    }

    /// How many arrays the concatenation is made of, the repeats included.
    ///
    /// # Panics
    /// Panics if that overflows a `usize`, which no concatenation has the memory to back: an
    /// element of it is at least a bit wide.
    fn len(&self) -> usize {
        self.count
            .checked_mul(self.repeats)
            .expect("the number of arrays to concatenate overflows a `usize`")
    }

    /// The same arrays, read as something else — downcast to their concrete array type, say, or
    /// narrowed to one of their fields.
    fn read_as<'n, B: ?Sized>(&self, get: &'n dyn Fn(usize) -> &'a B) -> ArrayList<'a, 'n, B> {
        ArrayList {
            get,
            count: self.count,
            repeats: self.repeats,
        }
    }
}

impl<A: ?Sized> Clone for ArrayList<'_, '_, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: ?Sized> Copy for ArrayList<'_, '_, A> {}

/// Something that stands in as a [`PlArray`] trait object: every array of this crate, and the
/// trait object itself.
///
/// This is what lets the parts of a concatenation that read nothing but the length, the null
/// count and the validity mask of an array be written once over `dyn PlArray`, whichever concrete
/// array type the caller holds.
trait AsPlArray {
    fn as_pl_array(&self) -> &dyn PlArray;
}

impl<A: PlArray> AsPlArray for A {
    #[inline(always)]
    fn as_pl_array(&self) -> &dyn PlArray {
        self
    }
}

impl AsPlArray for dyn PlArray {
    #[inline(always)]
    fn as_pl_array(&self) -> &dyn PlArray {
        self
    }
}

impl<'a, 'f, A: AsPlArray + ?Sized> ArrayList<'a, 'f, A> {
    /// The distinct arrays, as trait objects.
    fn distinct_erased(&self) -> impl ExactSizeIterator<Item = &'a dyn PlArray> + use<'a, 'f, A> {
        self.distinct().map(AsPlArray::as_pl_array)
    }

    /// Every array of the concatenation, the repeats included, as trait objects.
    fn iter_erased(&self) -> impl ExactSizeIterator<Item = &'a dyn PlArray> + use<'a, 'f, A> {
        self.iter().map(AsPlArray::as_pl_array)
    }

    /// The total number of elements the arrays hold and how many of them are null, per
    /// [`total_length_and_null_count`].
    fn length_and_null_count(&self) -> (usize, usize) {
        total_length_and_null_count(&mut self.distinct_erased(), self.repeats)
    }

    /// The validity mask of the concatenation, per [`concatenate_validities_with`].
    fn validities(&self, length: usize, null_count: usize) -> Option<Bitmap> {
        concatenate_validities_with(&mut self.iter_erased(), length, null_count)
    }
}

/// Concatenates `arrays`, in order, into a single array of their common [`PlArrayType`].
///
/// # Errors
/// This function errors if `arrays` is empty, if the arrays differ in [`PlArrayType`] or element
/// type, or if the values of nested arrays do not concatenate.
pub fn concatenate(arrays: &[&dyn PlArray]) -> PolarsResult<Box<dyn PlArray>> {
    concatenate_impl(ArrayList::new(&|index| arrays[index], arrays.len()))
}

/// Concatenates `repeats` copies of `array` into a single array of its [`PlArrayType`].
///
/// # Errors
/// This function errors if the values of a nested array do not concatenate with themselves, which
/// they always do unless an outside implementation of [`PlArray`] misreports its array type.
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

    concatenate_impl(ArrayList::repeated(&|_| array, 1, repeats))
}

/// Concatenates the arrays of `list`, in order, into a single array of their common
/// [`PlArrayType`], which is what [`concatenate`] and [`concatenate_repeated`] both come down to.
fn concatenate_impl(list: ArrayList<'_, '_, dyn PlArray>) -> PolarsResult<Box<dyn PlArray>> {
    let mut distinct = list.distinct();
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
            let get = downcast_get::<PlBooleanArray>(&list, array_type)?;
            Ok(Box::new(concatenate_boolean_impl(list.read_as(&get))))
        },
        PlArrayType::Primitive(primitive) => {
            with_match_pl_primitive_array_type!(first, |T| concatenate_primitive_as::<T>(&list))
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
            let get = downcast_get::<PlBinaryArray>(&list, array_type)?;
            Ok(Box::new(concatenate_binary_impl(list.read_as(&get))))
        },
        PlArrayType::BinaryView => {
            let get = downcast_get::<PlBinaryViewArray>(&list, array_type)?;
            Ok(Box::new(concatenate_binview_impl(list.read_as(&get))))
        },
        PlArrayType::Utf8View => {
            // A string array is a binary view array whose bytes are UTF-8, so it concatenates the
            // same way; what the wrapper adds is the promise, which is re-established below.
            let get = downcast_get::<PlUtf8ViewArray>(&list, array_type)?;
            let bytes_get = |index| get(index).as_binview();
            let concatenated = concatenate_binview_impl(list.read_as(&bytes_get));
            // SAFETY: every element came from a `PlUtf8ViewArray`, so every one is valid UTF-8.
            Ok(Box::new(unsafe {
                PlUtf8ViewArray::from_binview_unchecked(concatenated)
            }))
        },
        PlArrayType::FixedSizeBinary => {
            let get = downcast_get::<PlFixedSizeBinaryArray>(&list, array_type)?;
            Ok(Box::new(concatenate_fixed_size_binary_impl(
                list.read_as(&get),
            )?))
        },
        PlArrayType::Struct => {
            let get = downcast_get::<PlStructArray>(&list, array_type)?;
            Ok(Box::new(concatenate_struct_impl(list.read_as(&get))?))
        },
        PlArrayType::List => {
            let get = downcast_get::<PlListArray>(&list, array_type)?;
            Ok(Box::new(concatenate_list_impl(list.read_as(&get))?))
        },
        PlArrayType::FixedSizeList => {
            let get = downcast_get::<PlFixedSizeListArray>(&list, array_type)?;
            Ok(Box::new(concatenate_fixed_size_list_impl(
                list.read_as(&get),
            )?))
        },
        PlArrayType::Null => {
            let get = downcast_get::<PlNullArray>(&list, array_type)?;
            Ok(Box::new(concatenate_null_impl(list.read_as(&get))))
        },
        x @ PlArrayType::Object { .. } => {
            panic!("polars-array: no concatenate impl for {x:?}")
        },
    }
}

/// Concatenates the validity masks of `arrays`, in order, into the mask of their concatenation.
#[allow(private_bounds)]
pub fn concatenate_validities<A: PlArray + AsPlArray + ?Sized>(arrays: &[&A]) -> Option<Bitmap> {
    let get = |index: usize| arrays[index];
    let list = ArrayList::new(&get, arrays.len());
    let (length, null_count) = list.length_and_null_count();
    list.validities(length, null_count)
}

/// [`concatenate_validities`], for a caller that has already counted the elements and the nulls.
///
/// The arrays arrive as a trait object rather than as an iterator to be monomorphized over: a
/// mask is built out of whole arrays at a time, so the one indirect call per array this costs is
/// nothing against the copy of this function every caller would otherwise be handed.
fn concatenate_validities_with(
    arrays: &mut dyn Iterator<Item = &dyn PlArray>,
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
    for array in arrays {
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
pub fn concatenate_primitive<T: NativeType>(
    arrays: &[&PlPrimitiveArray<T>],
) -> PlPrimitiveArray<T> {
    concatenate_primitive_impl(ArrayList::new(&|index| arrays[index], arrays.len()))
}

/// [`concatenate_primitive`], over the arrays `iter` yields.
fn concatenate_primitive_impl<T: NativeType>(
    list: ArrayList<'_, '_, PlPrimitiveArray<T>>,
) -> PlPrimitiveArray<T> {
    if let Some(array) = only_non_empty(&list) {
        return array.clone();
    }

    let (length, null_count) = list.length_and_null_count();

    // Every element is null, so every value is undetermined and none of them has to be written.
    if length > 0 && null_count == length {
        return PlPrimitiveArray::new_full_null(length);
    }

    // Every array stands for the same repeated element, which the result repeats over all of them.
    //
    // The elements are compared as bytes rather than with `PartialEq`, which is the one place
    // this crate compares bytes on purpose: an array repeating `-0.0` and one repeating `+0.0`
    // hold equal numbers with different bytes, and collapsing the two onto whichever came first
    // would hand back the wrong zero for the other. Bytes that agree are the same value, so this
    // only ever declines a collapse that would have changed one — a `NaN` repeated by two arrays
    // now collapses in `O(1)` where comparing the floats left it to be written out.
    let scalar_bytes = |array: &PlPrimitiveArray<T>| {
        array
            .scalar_value()
            .map(|element| element.map(bytes::to_bytes))
    };
    if let Some(element) = shared_element(&list, scalar_bytes) {
        return match element {
            Some(value) => PlPrimitiveArray::new_scalar(bytes::from_bytes::<T>(value), length),
            None => PlPrimitiveArray::new_full_null(length),
        };
    }

    let validity = list
        .validities(length, null_count)
        .map(PlBitmap::from_bitmap);

    // Copying the values out is what the concatenation comes down to, and it reads nothing of
    // them but their bytes, so it is taken over the byte class of `T` rather than over `T`.
    let mut values = Vec::with_capacity(length);
    for array in list.iter() {
        bytes::extend_subslice(&mut values, array.values_bytes(), 0, array.len());
    }

    // SAFETY: the values hold one slot per element of the concatenation, and the mask is the one
    // `concatenate_validities_with` built for that many elements.
    unsafe {
        PlPrimitiveArray::new_unchecked(bytes::buffer_from_byte_vec::<T>(values), length, validity)
    }
}

/// Concatenates `arrays`, in order, into a single [`PlBooleanArray`].
pub fn concatenate_boolean(arrays: &[&PlBooleanArray]) -> PlBooleanArray {
    concatenate_boolean_impl(ArrayList::new(&|index| arrays[index], arrays.len()))
}

/// [`concatenate_boolean`], over the arrays `iter` yields.
fn concatenate_boolean_impl(list: ArrayList<'_, '_, PlBooleanArray>) -> PlBooleanArray {
    if let Some(array) = only_non_empty(&list) {
        return array.clone();
    }

    let (length, null_count) = list.length_and_null_count();

    // Every element is null, so every value is undetermined and none of them has to be written.
    if length > 0 && null_count == length {
        return PlBooleanArray::new_full_null(length);
    }

    // Every array stands for the same repeated element, which the result repeats over all of them.
    if let Some(element) = shared_element(&list, PlBooleanArray::scalar_value) {
        return match element {
            Some(value) => PlBooleanArray::new_scalar(value, length),
            None => PlBooleanArray::new_full_null(length),
        };
    }

    let validity = list
        .validities(length, null_count)
        .map(PlBitmap::from_bitmap);

    let mut values = BitmapBuilder::with_capacity(length);
    for array in list.iter().filter(|array| !array.is_empty()) {
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
pub fn concatenate_binary(arrays: &[&PlBinaryArray]) -> PlBinaryArray {
    concatenate_binary_impl(ArrayList::new(&|index| arrays[index], arrays.len()))
}

/// [`concatenate_binary`], over the arrays `iter` yields.
fn concatenate_binary_impl(list: ArrayList<'_, '_, PlBinaryArray>) -> PlBinaryArray {
    if let Some(array) = only_non_empty(&list) {
        return array.clone();
    }

    let (length, null_count) = list.length_and_null_count();

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
    if let Some(element) = shared_element(&list, PlBinaryArray::scalar_value) {
        return match element {
            Some(value) => PlBinaryArray::new_scalar(value, length),
            // An element shared as null makes every element of every array null, which the branch
            // above has already returned.
            None => PlBinaryArray::new_full_null(length),
        };
    }

    let validity = list
        .validities(length, null_count)
        .map(PlBitmap::from_bitmap);

    // One pass over the distinct arrays is what the concatenation is made of: the copies of them
    // cover the very same bytes over again, so the pass is repeated rather than built twice.
    let mut values: Vec<u8> = Vec::new();
    let mut offsets: Vec<u64> = Vec::with_capacity(length + 1);
    offsets.push(0);

    for array in list.distinct().filter(|array| !array.is_empty()) {
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
    for copy in 1..list.repeats {
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
/// # Panics
/// Panics if the arrays hold more data buffers between them than a view can index.
pub fn concatenate_binview(arrays: &[&PlBinaryViewArray]) -> PlBinaryViewArray {
    concatenate_binview_impl(ArrayList::new(&|index| arrays[index], arrays.len()))
}

/// [`concatenate_binview`], over the arrays `iter` yields.
fn concatenate_binview_impl(list: ArrayList<'_, '_, PlBinaryViewArray>) -> PlBinaryViewArray {
    if let Some(array) = only_non_empty(&list) {
        return array.clone();
    }

    let (length, null_count) = list.length_and_null_count();

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
    if let Some(element) = shared_element(&list, PlBinaryViewArray::scalar_value) {
        return match element {
            Some(value) => PlBinaryViewArray::new_scalar(value, length),
            None => PlBinaryViewArray::new_full_null(length),
        };
    }

    let validity = list
        .validities(length, null_count)
        .map(PlBitmap::from_bitmap);

    // One pass over the distinct arrays is what the concatenation is made of: the copies of them
    // hold the very same views over the very same data buffers, so neither is built twice.
    let mut buffers: Vec<Buffer<u8>> = Vec::new();
    let mut views: Vec<View> = Vec::with_capacity(length);
    for array in list.distinct().filter(|array| !array.is_empty()) {
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
    for _ in 1..list.repeats {
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
/// # Errors
/// This function errors if `arrays` is empty, since there is then no width for the elements of the
/// result to have, or if the arrays do not all have the same width.
pub fn concatenate_fixed_size_binary(
    arrays: &[&PlFixedSizeBinaryArray],
) -> PolarsResult<PlFixedSizeBinaryArray> {
    concatenate_fixed_size_binary_impl(ArrayList::new(&|index| arrays[index], arrays.len()))
}

/// [`concatenate_fixed_size_binary`], over the arrays `iter` yields.
fn concatenate_fixed_size_binary_impl(
    list: ArrayList<'_, '_, PlFixedSizeBinaryArray>,
) -> PolarsResult<PlFixedSizeBinaryArray> {
    let mut distinct = list.distinct();
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

    if let Some(array) = only_non_empty(&list) {
        return Ok(array.clone());
    }

    let (length, null_count) = list.length_and_null_count();

    // Every element is null, so every value is undetermined and none of them has to be written
    // out: a zeroed element of the width stands in for the one they all cover.
    if length > 0 && null_count == length {
        return Ok(PlFixedSizeBinaryArray::new_full_null(width, length));
    }

    // Every array stands for the same repeated element, which the result repeats over all of them.
    if let Some(element) = shared_element(&list, PlFixedSizeBinaryArray::scalar_value) {
        return Ok(match element {
            Some(element) => PlFixedSizeBinaryArray::new_scalar(element, length),
            // An element shared as null makes every element of every array null, which the branch
            // above has already returned.
            None => PlFixedSizeBinaryArray::new_full_null(width, length),
        });
    }

    let validity = list
        .validities(length, null_count)
        .map(PlBitmap::from_bitmap);

    // One pass over the distinct arrays is what the concatenation is made of: the copies of them
    // cover the very same bytes over again, so the pass is repeated rather than built twice.
    let flat_len = length
        .checked_mul(width)
        .expect("the values of the concatenation overflow a `usize`");
    let mut values: Vec<u8> = Vec::with_capacity(flat_len);
    for array in list.distinct().filter(|array| !array.is_empty()) {
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
    for _ in 1..list.repeats {
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
/// # Errors
/// This function errors if `arrays` is empty, if the arrays do not all have the same width, or if
/// the values do not concatenate.
pub fn concatenate_fixed_size_list(
    arrays: &[&PlFixedSizeListArray],
) -> PolarsResult<PlFixedSizeListArray> {
    concatenate_fixed_size_list_impl(ArrayList::new(&|index| arrays[index], arrays.len()))
}

/// [`concatenate_fixed_size_list`], over the arrays `iter` yields.
fn concatenate_fixed_size_list_impl(
    list: ArrayList<'_, '_, PlFixedSizeListArray>,
) -> PolarsResult<PlFixedSizeListArray> {
    let mut distinct = list.distinct();
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

    if let Some(array) = only_non_empty(&list) {
        return Ok(array.clone());
    }

    let (length, null_count) = list.length_and_null_count();

    // The values of some element, which is what the undetermined values of a fully null result are
    // taken from: every array agrees on the width, so any element of any of them is as wide as the
    // result needs. There is one whenever the concatenation holds an element at all.
    let undetermined = || {
        let array = list
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
    if let Some(element) = shared_element(&list, PlFixedSizeListArray::scalar_value) {
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
    let mut values = Vec::with_capacity(list.len());
    for array in list.iter() {
        if let Some(array_values) = array.flat_values() {
            values.push(array_values.to_boxed());
        } else if let Some(element) = array.scalar_values() {
            // Concatenating the element with copies of itself is what repeats it, and that keeps
            // the values scalar when the element is itself a single repeated value.
            values.push(concatenate_repeated(element, array.len())?);
        }
    }

    // The values of the arrays are concatenated through the boxes they were sliced into.
    let values = concatenate_impl(ArrayList::new(&|index| &*values[index], values.len()))?;
    let validity = list
        .validities(length, null_count)
        .map(PlBitmap::from_bitmap);

    // SAFETY: every array contributed the width of each of its elements, so the values hold one
    // width per element of the concatenation, and the mask covers that many elements.
    Ok(unsafe { PlFixedSizeListArray::new_unchecked(values, width, length, validity) })
}

/// Concatenates `arrays`, in order, into a single [`PlNullArray`].
pub fn concatenate_null(arrays: &[&PlNullArray]) -> PlNullArray {
    concatenate_null_impl(ArrayList::new(&|index| arrays[index], arrays.len()))
}

/// [`concatenate_null`], over the arrays `iter` yields.
fn concatenate_null_impl(list: ArrayList<'_, '_, PlNullArray>) -> PlNullArray {
    PlNullArray::new(list.length_and_null_count().0)
}

/// Concatenates `arrays`, in order, into a single [`PlStructArray`], concatenating each field with
/// the field at the same position of every other array.
///
/// # Errors
/// This function errors if the arrays do not all have the same number of fields, or if any of the
/// fields do not concatenate.
pub fn concatenate_struct(arrays: &[&PlStructArray]) -> PolarsResult<PlStructArray> {
    concatenate_struct_impl(ArrayList::new(&|index| arrays[index], arrays.len()))
}

/// [`concatenate_struct`], over the arrays `iter` yields.
fn concatenate_struct_impl(list: ArrayList<'_, '_, PlStructArray>) -> PolarsResult<PlStructArray> {
    let mut distinct = list.distinct();
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

    if let Some(array) = only_non_empty(&list) {
        return Ok(array.clone());
    }

    let (length, null_count) = list.length_and_null_count();

    // The field at each position is concatenated over the very same repetition, so the fields of
    // the result are as long as the concatenation itself.
    let fields = (0..num_fields)
        .map(|i| concatenate_impl(list.read_as(&|index| list.at(index).field(i))))
        .collect::<PolarsResult<Vec<_>>>()?;

    // Every row is null, so a single shared bit is the whole mask. The fields are kept as they
    // are: their values are undetermined, and it is the mask that makes every row null.
    if length > 0 && null_count == length {
        return Ok(PlStructArray::new_full_null(fields, length));
    }

    let validity = list
        .validities(length, null_count)
        .map(PlBitmap::from_bitmap);

    // SAFETY: every field is the concatenation of the fields at that position, so it is as long as
    // the arrays together, and the mask is the flat one built for that many elements.
    Ok(unsafe { PlStructArray::new_unchecked(fields, length, validity) })
}

/// Concatenates `arrays`, in order, into a single [`PlListArray`] over the concatenation of the
/// values their lists reach.
///
/// # Errors
/// This function errors if `arrays` is empty, since there is no values array to take the lists of
/// the result over, or if the values do not concatenate.
pub fn concatenate_list(arrays: &[&PlListArray]) -> PolarsResult<PlListArray> {
    concatenate_list_impl(ArrayList::new(&|index| arrays[index], arrays.len()))
}

/// [`concatenate_list`], over the arrays `iter` yields.
fn concatenate_list_impl(list: ArrayList<'_, '_, PlListArray>) -> PolarsResult<PlListArray> {
    let Some(first) = list.distinct().next() else {
        polars_bail!(
            InvalidOperation:
            "cannot concatenate an empty list of list arrays: there is no values array to take \
             the lists of the result over"
        );
    };

    if let Some(array) = only_non_empty(&list) {
        return Ok(array.clone());
    }

    let (length, null_count) = list.length_and_null_count();

    // Every element is null, so every list is undetermined and none of them has to be written out;
    // the values array is what determines the type of the lists, of which the first one will do.
    if length > 0 && null_count == length {
        return Ok(PlListArray::new_full_null(
            first.values().to_boxed(),
            length,
        ));
    }

    // Every array stands for the same repeated element, which the result repeats over all of them.
    if let Some(element) = shared_element(&list, PlListArray::scalar_value) {
        return Ok(match element {
            Some(element) => PlListArray::new_scalar(element, length),
            None => PlListArray::new_full_null(first.values().to_boxed(), length),
        });
    }

    // The values of each array are sliced to what its offsets reach, so that the offsets of the
    // result can be rebased onto their concatenation.
    let mut values = Vec::with_capacity(list.len());
    let mut offsets = Vec::with_capacity(length + 1);
    offsets.push(0);

    let mut end = 0;
    for array in list.iter() {
        if array.is_empty() {
            // No element of the array is reachable, but its values are still what the type of its
            // lists is taken from, which every array has to agree on.
            values.push(array.values().sliced(0, 0));
            continue;
        }

        match array.offsets_repr() {
            ArrayRepr::Flat(array_offsets) => {
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
            },
            // Every element of the array covers the same range, which the result writes out once
            // per element: concatenating the element with copies of itself is what repeats it, and
            // that keeps the values scalar when the element is itself a single repeated value.
            ArrayRepr::Scalar(range) => {
                let value_length = range.end - range.start;
                let element = array
                    .values()
                    .sliced(range.start as usize, value_length as usize);
                values.push(concatenate_repeated(&*element, array.len())?);

                offsets.extend((1..=array.len() as u64).map(|i| end + i * value_length));
                end += value_length * array.len() as u64;
            },
        }
    }

    // The values of the arrays are concatenated through the boxes they were sliced into.
    let values = concatenate_impl(ArrayList::new(&|index| &*values[index], values.len()))?;
    let validity = list
        .validities(length, null_count)
        .map(PlBitmap::from_bitmap);

    // SAFETY: the offsets are the lengths of the lists of every array in order, so they are
    // ordered and end at the length of the values; the mask covers that many elements.
    Ok(unsafe { PlListArray::new_unchecked(values, Buffer::from(offsets), length, validity) })
}

/// The one array that holds every element of the concatenation, if the others are all empty.
fn only_non_empty<'a, A: ?Sized + PlArray>(list: &ArrayList<'a, '_, A>) -> Option<&'a A> {
    let mut non_empty = list.distinct().filter(|array| !array.is_empty());
    match non_empty.next() {
        Some(array) => (non_empty.next().is_none() && list.repeats == 1).then_some(array),
        None => list.distinct().next(),
    }
}

/// The total number of elements `distinct` arrays hold `repeats` times over, and how many of those
/// elements are null.
///
/// The arrays arrive as a trait object rather than as an iterator to be monomorphized over, for
/// the reason [`concatenate_validities_with`] gives.
///
/// # Panics
/// Panics if the total length overflows a `usize`, which the scalar representation makes possible
/// without the memory to back it.
fn total_length_and_null_count(
    distinct: &mut dyn Iterator<Item = &dyn PlArray>,
    repeats: usize,
) -> (usize, usize) {
    let mut length = 0usize;
    let mut null_count = 0usize;
    for array in distinct {
        length = length
            .checked_add(array.len())
            .expect("the total length of the concatenation overflows a `usize`");
        null_count += array.null_count();
    }

    // Every copy of the distinct arrays holds the same elements over again.
    (
        length
            .checked_mul(repeats)
            .expect("the total length of the concatenation overflows a `usize`"),
        // No more null than long, so this cannot overflow where the length does not.
        null_count * repeats,
    )
}

/// The element every element of every array equals, if `element` sees one for each of them and they
/// all agree.
fn shared_element<'a, A: PlArray, T: PartialEq>(
    list: &ArrayList<'a, '_, A>,
    element: impl Fn(&'a A) -> Option<T>,
) -> Option<T> {
    let mut shared = None;
    for array in list.distinct().filter(|array| !array.is_empty()) {
        let element = element(array)?;
        match &shared {
            Some(shared) if *shared != element => return None,
            Some(_) => {},
            None => shared = Some(element),
        }
    }
    shared
}

/// Reads the arrays of `list` as an `A`, or `None` if that is not the concrete array type of every
/// one of them.
fn try_downcast_get<'a, 'f, A: PlArray>(
    list: &ArrayList<'a, 'f, dyn PlArray>,
) -> Option<impl Fn(usize) -> &'a A + use<'a, 'f, A>> {
    if !list.distinct().all(|array| array.as_any().is::<A>()) {
        return None;
    }

    let get = list.get;
    // Every distinct array was just seen to be an `A`, and the repetition reaches nothing else.
    Some(move |index: usize| get(index).as_any().downcast_ref::<A>().unwrap())
}

/// [`try_downcast_get`], for the `array_type` every array reports, which guarantees `A` is their
/// concrete array type.
fn downcast_get<'a, 'f, A: PlArray>(
    list: &ArrayList<'a, 'f, dyn PlArray>,
    array_type: PlArrayType,
) -> PolarsResult<impl Fn(usize) -> &'a A + use<'a, 'f, A>> {
    try_downcast_get(list).ok_or_else(|| {
        polars_err!(
            InvalidOperation:
            "cannot concatenate arrays that report the array type {:?} but are not all of the \
             same concrete array type",
            array_type,
        )
    })
}

/// Concatenates the arrays of `list` as [`PlPrimitiveArray<T>`], or returns `None` if that is not
/// the concrete array type of every one of them.
fn concatenate_primitive_as<T: NativeType>(
    list: &ArrayList<'_, '_, dyn PlArray>,
) -> Option<Box<dyn PlArray>> {
    let get = try_downcast_get::<PlPrimitiveArray<T>>(list)?;
    Some(Box::new(concatenate_primitive_impl(list.read_as(&get))))
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

    /// Repeated elements are collapsed onto one when their *bytes* agree, not when `==` does.
    ///
    /// `-0.0 == 0.0`, so comparing the floats would collapse an array of each onto whichever came
    /// first and hand back the wrong zero for the other — a sign that survives a division. `NaN`
    /// is the other way round: it equals nothing, itself included, so comparing the floats would
    /// write out a repeat that the bytes collapse in `O(1)`.
    #[test]
    fn the_two_zeroes_are_kept_apart_and_repeated_nans_are_not_written_out() {
        let negative = PlPrimitiveArray::new_scalar(-0.0f64, 2);
        let positive = PlPrimitiveArray::new_scalar(0.0f64, 2);
        let concatenated = concatenate_primitive(&[&negative, &positive]);

        assert!(concatenated.is_flat());
        let signs: Vec<bool> = concatenated
            .flat_values()
            .unwrap()
            .iter()
            .map(|value| value.is_sign_negative())
            .collect();
        assert_eq!(signs, [true, true, false, false]);

        // The same bytes are the same value, so a repeated `NaN` still collapses.
        let nans = PlPrimitiveArray::new_scalar(f64::NAN, 1_000_000_000);
        let concatenated = concatenate_primitive(&[&nans, &nans]);

        assert_eq!(concatenated.len(), 2_000_000_000);
        assert!(concatenated.is_scalar());
        assert!(concatenated.value(1_999_999_999).is_nan());
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
