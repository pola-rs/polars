//! The string view of a [`PlBinaryViewArray`].
//!
//! `polars-array` is physical storage only: a [`PlBinaryViewArray`] is a sequence of byte strings
//! and nothing in it says those bytes are a string. A [`StringChunked`](crate::prelude::StringChunked),
//! on the other hand, is exactly the promise that they are — so the UTF-8 invariant lives here, at
//! the `ChunkedArray` level, rather than in the array.
//!
//! [`PlUtf8ViewArray`] is what carries that invariant into the typed API. It is
//! `repr(transparent)` over the [`PlBinaryViewArray`] a `StringChunked` chunk actually is, so a
//! chunk is *borrowed* as one — [`PlUtf8ViewArray::from_binview_ref_unchecked`] — rather than
//! stored as one, and reading an element hands out the `&str` those bytes are without validating
//! them again.
//!
//! # The invariant
//!
//! Every element of a [`PlUtf8ViewArray`] — including the ones masked off as null, which a
//! validity mask can be put back over at any time — is valid UTF-8. Constructing one is therefore
//! `unsafe`; [`PlUtf8ViewArray::from_binview`] is the checked constructor that establishes it.
//!
//! Because the wrapper is transparent, it must never *own* the trait object a chunk is stored as:
//! the rest of the world downcasts a `dyn PlArray` of [`PlArrayType::BinaryView`] to a
//! [`PlBinaryViewArray`], and a `Box<PlUtf8ViewArray>` would carry the wrong vtable for that. Every
//! method here that hands out a `Box<dyn PlArray>` therefore boxes the inner array.

use std::any::Any;

use arrow::array::View;
use arrow::bitmap::Bitmap;
use arrow::trusted_len::TrustedLen;
use polars_array::binview::{PlBinaryViewIter, PlBinaryViewValuesIter};
use polars_array::builder::ShareStrategy;
use polars_array::{
    ArrayFromIter, Flat, PlArray, PlArrayType, PlBinaryViewArray, PlBinaryViewArrayBuilder,
    PlBitmapRef, StaticArray, StaticArrayBuilder, ZeroableArrayFromIter,
};
use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_utils::IdxSize;

/// A [`PlBinaryViewArray`] whose every element is valid UTF-8.
///
/// See the [module docs](self) for the invariant and for why this is a borrowed view of a chunk
/// rather than the type a chunk is stored as.
#[derive(Clone)]
#[repr(transparent)]
pub struct PlUtf8ViewArray(PlBinaryViewArray);

impl PlUtf8ViewArray {
    /// Wraps `array`, checking that every one of its elements is valid UTF-8.
    ///
    /// This is `O(total_bytes_len)`: the elements masked off as null are checked too, because a
    /// validity mask can be replaced without the bytes under it changing.
    pub fn from_binview(array: PlBinaryViewArray) -> PolarsResult<Self> {
        validate_utf8(&array)?;
        // SAFETY: just validated.
        Ok(unsafe { Self::from_binview_unchecked(array) })
    }

    /// Wraps `array` without checking that its elements are valid UTF-8.
    ///
    /// # Safety
    /// Every element of `array`, including the ones masked off as null, must be valid UTF-8.
    #[inline(always)]
    pub const unsafe fn from_binview_unchecked(array: PlBinaryViewArray) -> Self {
        Self(array)
    }

    /// Borrows `array` as an array of strings without checking that its elements are valid UTF-8.
    ///
    /// # Safety
    /// Every element of `array`, including the ones masked off as null, must be valid UTF-8.
    #[inline(always)]
    pub const unsafe fn from_binview_ref_unchecked(array: &PlBinaryViewArray) -> &Self {
        // SAFETY: `Self` is `repr(transparent)` over the array it wraps.
        unsafe { &*(std::ptr::from_ref(array).cast::<Self>()) }
    }

    /// The bytes of these strings, giving up the promise that they are one.
    #[inline(always)]
    pub const fn as_binview(&self) -> &PlBinaryViewArray {
        &self.0
    }

    /// The bytes of these strings, giving up the promise that they are one.
    #[inline(always)]
    pub fn into_binview(self) -> PlBinaryViewArray {
        self.0
    }

    /// An empty array.
    #[inline]
    pub fn new_empty() -> Self {
        Self(PlBinaryViewArray::new_empty())
    }

    /// An array of `length` nulls.
    #[inline]
    pub fn new_full_null(length: usize) -> Self {
        Self(PlBinaryViewArray::new_full_null(length))
    }

    /// An array of `length` copies of `value`, in `O(1)` memory.
    #[inline]
    pub fn new_scalar(value: &str, length: usize) -> Self {
        Self(PlBinaryViewArray::new_scalar(value.as_bytes(), length))
    }

    /// Returns the element at `i`, whether or not it is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value(&self, i: usize) -> &str {
        // SAFETY: the elements of this array are valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(self.0.value(i)) }
    }

    /// Returns the element at `i`, whether or not it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &str {
        // SAFETY: the caller keeps `i` in bounds, and the elements are valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(self.0.value_unchecked(i)) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn get(&self, i: usize) -> Option<&str> {
        // SAFETY: the elements of this array are valid UTF-8.
        self.0
            .get(i)
            .map(|v| unsafe { std::str::from_utf8_unchecked(v) })
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> Option<&str> {
        // SAFETY: the caller keeps `i` in bounds, and the elements are valid UTF-8.
        unsafe {
            self.0
                .get_unchecked(i)
                .map(|v| std::str::from_utf8_unchecked(v))
        }
    }

    /// The single value every element of a scalar array is, or `None` if this array is not scalar.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<&str>> {
        // SAFETY: the elements of this array are valid UTF-8.
        self.0
            .scalar_value()
            .map(|v| v.map(|v| unsafe { std::str::from_utf8_unchecked(v) }))
    }

    /// Iterates the elements, ignoring validity.
    #[inline]
    pub fn values_iter(&self) -> PlUtf8ViewValuesIter<'_> {
        PlUtf8ViewValuesIter(self.0.values_iter())
    }

    /// Iterates the elements, `None` for the null ones.
    #[inline]
    pub fn iter(&self) -> PlUtf8ViewIter<'_> {
        PlUtf8ViewIter(self.0.iter())
    }

    /// Iterates `length` elements, repeating the single element of a scalar array.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlUtf8ViewValuesIter<'_> {
        PlUtf8ViewValuesIter(self.0.broadcast_values_iter(length))
    }

    /// Iterates `length` elements, repeating the single element of a scalar array.
    #[inline]
    pub fn broadcast_iter(&self, length: usize) -> PlUtf8ViewIter<'_> {
        PlUtf8ViewIter(self.0.broadcast_iter(length))
    }

    /// Returns this array with its validity mask replaced.
    #[inline]
    #[must_use]
    pub fn with_validity(self, validity: Option<Bitmap>) -> Self {
        Self(self.0.with_validity(validity))
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    #[inline]
    #[must_use]
    pub fn sliced(self, offset: usize, length: usize) -> Self {
        Self(self.0.sliced(offset, length))
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    #[inline]
    #[must_use]
    pub unsafe fn sliced_unchecked(self, offset: usize, length: usize) -> Self {
        // SAFETY: the caller keeps the slice in bounds.
        Self(unsafe { self.0.sliced_unchecked(offset, length) })
    }

    /// Returns an array of `length` copies of the element at `index`.
    #[inline]
    #[must_use]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        Self(self.0.new_from_index(index, length))
    }

    /// The total number of bytes the elements of this array are.
    #[inline]
    pub fn total_bytes_len(&self) -> usize {
        self.0.total_bytes_len()
    }

    /// The views of this array, which index its data buffers.
    #[inline]
    pub fn flat_views(&self) -> Option<&Buffer<View>> {
        self.0.flat_views()
    }

    /// The data buffers the views of this array point into.
    #[inline]
    pub const fn data_buffers(&self) -> &Buffer<Buffer<u8>> {
        self.0.data_buffers()
    }

    /// Whether every backing buffer of this array holds one slot per element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.0.is_flat()
    }

    /// Returns this array in the flat representation.
    #[inline]
    pub fn to_flat(&self) -> Flat<Self> {
        // SAFETY: the inner array is written out flat, and the wrapper is transparent over it.
        unsafe { Flat::from_array_unchecked(Self(self.0.to_flat().into_array())) }
    }

    /// Borrows this array as a flat one, or `None` if any backing buffer is scalar.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: the inner array is flat, and the wrapper is transparent over it.
        self.0
            .as_flat()
            .map(|_| unsafe { Flat::from_ref_unchecked(self) })
    }
}

/// Checks that every element of `array`, including the ones masked off as null, is valid UTF-8.
fn validate_utf8(array: &PlBinaryViewArray) -> PolarsResult<()> {
    // The validity mask is dropped rather than honoured: a null element still holds bytes, and
    // replacing the mask must not be able to expose bytes that were never checked.
    for value in array.to_flat().into_array().without_validity().values_iter() {
        std::str::from_utf8(value).map_err(|e| {
            polars_error::polars_err!(ComputeError: "invalid utf8: {}", e)
        })?;
    }
    Ok(())
}

impl Default for PlUtf8ViewArray {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

impl PartialEq for PlUtf8ViewArray {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for PlUtf8ViewArray {}

impl std::fmt::Debug for PlUtf8ViewArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PlUtf8ViewArray")?;
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a> IntoIterator for &'a PlUtf8ViewArray {
    type Item = Option<&'a str>;
    type IntoIter = PlUtf8ViewIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl PlArray for PlUtf8ViewArray {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.0
    }

    #[inline]
    fn array_type(&self) -> PlArrayType {
        PlArrayType::BinaryView
    }

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn validity(&self) -> Option<PlBitmapRef<'_>> {
        self.0.validity()
    }

    #[inline]
    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    #[inline]
    fn slice(&mut self, offset: usize, length: usize) {
        self.0.slice(offset, length);
    }

    #[inline]
    unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        // SAFETY: the caller keeps the slice in bounds.
        unsafe { self.0.slice_unchecked(offset, length) };
    }

    #[inline]
    fn set_validity(&mut self, validity: Option<Bitmap>) {
        self.0.set_validity(validity);
    }

    // The boxed array is the *inner* one: the world downcasts a `dyn PlArray` of
    // `PlArrayType::BinaryView` to a `PlBinaryViewArray`, so this wrapper must not own a vtable.

    #[inline]
    unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Box<dyn PlArray> {
        // SAFETY: the caller keeps `index` in bounds.
        unsafe { PlArray::new_from_index_unchecked(&self.0, index, length) }
    }

    #[inline]
    fn to_boxed(&self) -> Box<dyn PlArray> {
        Box::new(self.0.clone())
    }

    #[inline]
    fn eq_dyn(&self, other: &dyn PlArray) -> bool {
        self.0.eq_dyn(other)
    }
}

impl StaticArray for PlUtf8ViewArray {
    type ValueT<'a> = &'a str;
    type ZeroableValueT<'a> = Option<&'a str>;
    type ValueIterT<'a> = PlUtf8ViewValuesIter<'a>;
    type IterT<'a> = PlUtf8ViewIter<'a>;
    type Builder = PlUtf8ViewArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> &str {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<&str> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn broadcast_iter(&self, length: usize) -> Self::IterT<'_> {
        self.broadcast_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }

    #[inline]
    fn into_boxed(self) -> Box<dyn PlArray> {
        // The trait object is the array this wrapper is transparent over — see the module docs.
        Box::new(self.0)
    }
}

/// Iterator over the elements of a [`PlUtf8ViewArray`], ignoring validity.
#[derive(Clone)]
pub struct PlUtf8ViewValuesIter<'a>(PlBinaryViewValuesIter<'a>);

impl<'a> Iterator for PlUtf8ViewValuesIter<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0
            .next()
            .map(|v| unsafe { std::str::from_utf8_unchecked(v) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl DoubleEndedIterator for PlUtf8ViewValuesIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0
            .next_back()
            .map(|v| unsafe { std::str::from_utf8_unchecked(v) })
    }
}

impl ExactSizeIterator for PlUtf8ViewValuesIter<'_> {}

// SAFETY: the iterator it wraps is trusted, and mapping the bytes to the string they are does not
// change how many there are.
unsafe impl TrustedLen for PlUtf8ViewValuesIter<'_> {}

/// Iterator over the elements of a [`PlUtf8ViewArray`], `None` for the null ones.
#[derive(Clone)]
pub struct PlUtf8ViewIter<'a>(PlBinaryViewIter<'a>);

impl<'a> Iterator for PlUtf8ViewIter<'a> {
    type Item = Option<&'a str>;

    #[inline]
    fn next(&mut self) -> Option<Option<&'a str>> {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0
            .next()
            .map(|v| v.map(|v| unsafe { std::str::from_utf8_unchecked(v) }))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl DoubleEndedIterator for PlUtf8ViewIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8.
        self.0
            .next_back()
            .map(|v| v.map(|v| unsafe { std::str::from_utf8_unchecked(v) }))
    }
}

impl ExactSizeIterator for PlUtf8ViewIter<'_> {}

// SAFETY: the iterator it wraps is trusted, and mapping the bytes to the string they are does not
// change how many there are.
unsafe impl TrustedLen for PlUtf8ViewIter<'_> {}

/// The builder of a [`PlUtf8ViewArray`].
///
/// Every value appended is a `&str`, which is what keeps the built array's invariant: the bytes
/// that reach the inner [`PlBinaryViewArrayBuilder`] were a string to begin with.
#[derive(Default)]
pub struct PlUtf8ViewArrayBuilder(PlBinaryViewArrayBuilder);

impl PlUtf8ViewArrayBuilder {
    /// A builder with no capacity reserved.
    #[inline]
    pub fn new() -> Self {
        Self(PlBinaryViewArrayBuilder::new())
    }

    /// A builder with room for `capacity` elements.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(PlBinaryViewArrayBuilder::with_capacity(capacity))
    }

    /// The builder of the bytes under these strings.
    ///
    /// # Safety
    /// Every value appended through it must be valid UTF-8.
    #[inline]
    pub unsafe fn inner_mut(&mut self) -> &mut PlBinaryViewArrayBuilder {
        &mut self.0
    }
}

/// Borrows an array of strings as the bytes under them.
#[inline(always)]
fn as_binview(array: &PlUtf8ViewArray) -> &PlBinaryViewArray {
    &array.0
}

impl StaticArrayBuilder for PlUtf8ViewArrayBuilder {
    type Array = PlUtf8ViewArray;

    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn freeze(self) -> PlUtf8ViewArray {
        // SAFETY: every value appended was a `&str`.
        unsafe { PlUtf8ViewArray::from_binview_unchecked(self.0.freeze()) }
    }

    #[inline]
    fn freeze_reset(&mut self) -> PlUtf8ViewArray {
        // SAFETY: every value appended was a `&str`.
        unsafe { PlUtf8ViewArray::from_binview_unchecked(self.0.freeze_reset()) }
    }

    #[inline]
    fn extend_nulls(&mut self, length: usize) {
        self.0.extend_nulls(length);
    }

    #[inline]
    fn subslice_extend(
        &mut self,
        other: &PlUtf8ViewArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        self.0.subslice_extend(as_binview(other), start, length, share);
    }

    #[inline]
    fn subslice_extend_repeated(
        &mut self,
        other: &PlUtf8ViewArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        self.0
            .subslice_extend_repeated(as_binview(other), start, length, repeats, share);
    }

    #[inline]
    fn subslice_extend_each_repeated(
        &mut self,
        other: &PlUtf8ViewArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        self.0
            .subslice_extend_each_repeated(as_binview(other), start, length, repeats, share);
    }

    #[inline]
    unsafe fn gather_extend(
        &mut self,
        other: &PlUtf8ViewArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        // SAFETY: the caller keeps every index in bounds.
        unsafe { self.0.gather_extend(as_binview(other), idxs, share) };
    }

    #[inline]
    fn opt_gather_extend(
        &mut self,
        other: &PlUtf8ViewArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        self.0.opt_gather_extend(as_binview(other), idxs, share);
    }
}

/// A string that a [`PlUtf8ViewArray`] can be collected from.
///
/// This is the marker that keeps the array's invariant across a collect: the bytes reach the inner
/// [`PlBinaryViewArray`] through the same conversion any byte string does, and it is membership of
/// this trait — a `&str`, a `String`, a `Cow<str>`, and nothing else — that says they were a
/// string to begin with.
pub trait IntoUtf8Bytes: Sized {}

impl IntoUtf8Bytes for &str {}
impl IntoUtf8Bytes for String {}
impl IntoUtf8Bytes for std::borrow::Cow<'_, str> {}

impl<V: IntoUtf8Bytes> ArrayFromIter<V> for PlUtf8ViewArray
where
    PlBinaryViewArray: ArrayFromIter<V>,
{
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = V>>(iter: I) -> Self {
        // SAFETY: `IntoUtf8Bytes` says every value collected was a string.
        unsafe { Self::from_binview_unchecked(PlBinaryViewArray::arr_from_iter(iter)) }
    }

    #[inline]
    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<V, E>>>(iter: I) -> Result<Self, E> {
        let bytes = PlBinaryViewArray::try_arr_from_iter(iter)?;
        // SAFETY: as above.
        Ok(unsafe { Self::from_binview_unchecked(bytes) })
    }
}

impl<V: IntoUtf8Bytes> ArrayFromIter<Option<V>> for PlUtf8ViewArray
where
    PlBinaryViewArray: ArrayFromIter<Option<V>>,
{
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = Option<V>>>(iter: I) -> Self {
        // SAFETY: `IntoUtf8Bytes` says every value collected was a string.
        unsafe { Self::from_binview_unchecked(PlBinaryViewArray::arr_from_iter(iter)) }
    }

    #[inline]
    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<V>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        let bytes = PlBinaryViewArray::try_arr_from_iter(iter)?;
        // SAFETY: as above.
        Ok(unsafe { Self::from_binview_unchecked(bytes) })
    }
}

// The zeroable stand-in for a `&str` is `Option<&str>`, which is what the collect above takes.
impl ZeroableArrayFromIter for PlUtf8ViewArray {}

impl<'a> FromIterator<Option<&'a str>> for PlUtf8ViewArray {
    #[inline]
    fn from_iter<I: IntoIterator<Item = Option<&'a str>>>(iter: I) -> Self {
        Self::arr_from_iter(iter)
    }
}
