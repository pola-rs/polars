//! The string view of a [`PlBinaryViewArray`].

use std::any::Any;
use std::borrow::Cow;

use arrow::array::View;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_err};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::binview::PlBinaryViewArray;
use crate::bitmap::{PlBitmap, PlBitmapRef};
use crate::broadcast::ArrayRepr;
use crate::flat::Flat;

mod builder;
mod iterator;

pub use builder::PlUtf8ViewArrayBuilder;
pub use iterator::{PlUtf8ViewIter, PlUtf8ViewValuesIter};

/// A [`PlBinaryViewArray`] whose every element is valid UTF-8.
#[derive(Clone)]
#[repr(transparent)]
pub struct PlUtf8ViewArray(PlBinaryViewArray);

impl PlUtf8ViewArray {
    /// Wraps `array`, checking that every one of its elements is valid UTF-8.
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

    /// Whether this array is entirely stored in the scalar representation — see
    /// [`PlBinaryViewArray::is_scalar`].
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.0.is_scalar()
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
        // SAFETY: the elements of this array are valid UTF-8.
        unsafe { PlUtf8ViewValuesIter::new(self.0.values_iter()) }
    }

    /// Iterates the elements, `None` for the null ones.
    #[inline]
    pub fn iter(&self) -> PlUtf8ViewIter<'_> {
        // SAFETY: the elements of this array are valid UTF-8.
        unsafe { PlUtf8ViewIter::new(self.0.iter()) }
    }

    /// Iterates `length` elements, repeating the single element of a scalar array.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlUtf8ViewValuesIter<'_> {
        // SAFETY: the elements of this array are valid UTF-8.
        unsafe { PlUtf8ViewValuesIter::new(self.0.broadcast_values_iter(length)) }
    }

    /// Returns this array with its validity mask replaced, keeping the representation it is in.
    ///
    /// # Panics
    /// Panics unless `validity` covers exactly [`len`](Self::len) elements.
    #[inline]
    #[must_use]
    pub fn with_validity(self, validity: Option<PlBitmap>) -> Self {
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

    /// Which representation the backing views buffer is in — see
    /// [`PlBinaryViewArray::views_repr`].
    #[inline]
    pub fn views_repr(&self) -> ArrayRepr<&Buffer<View>, View> {
        self.0.views_repr()
    }

    /// The views of this array, which index its data buffers.
    #[inline]
    pub fn flat_views(&self) -> Option<&Buffer<View>> {
        self.0.flat_views()
    }

    /// The view every element of this array reads, if the views buffer holds a single slot.
    #[inline]
    pub fn scalar_views(&self) -> Option<View> {
        self.0.scalar_views()
    }

    /// Whether the views buffer holds a single view shared by every element.
    #[inline]
    pub fn views_are_scalar(&self) -> bool {
        self.0.views_are_scalar()
    }

    /// Whether the views buffer holds one slot per element.
    #[inline]
    pub fn views_are_flat(&self) -> bool {
        self.0.views_are_flat()
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

    /// Returns this array in the flat representation, borrowing this array itself if it is already
    /// laid out flat.
    #[inline]
    pub fn to_flat(&self) -> Cow<'_, Flat<Self>> {
        if let Some(flat) = self.as_flat() {
            return Cow::Borrowed(flat);
        }

        // SAFETY: the inner array is written out flat, and the wrapper is transparent over it.
        Cow::Owned(unsafe { Flat::new(Self(self.0.to_flat().into_owned().into_array())) })
    }

    /// Returns this array with every view replaced by what `update_view` makes of it.
    ///
    /// # Safety
    /// The views the closure hands back must uphold every invariant of a view: each must read bytes
    /// this array's data buffers hold, and those bytes must be valid UTF-8.
    pub unsafe fn apply_views<F: FnMut(View, &str) -> View>(&self, mut update_view: F) -> Self {
        // A scalar views buffer holds the one view every element reads, so the closure maps that
        // view alone and what it hands back stands for every element in turn. The mask is put
        // back as it was, which is what keeps a null element null.
        let length = self.0.len();
        if self.0.views_are_scalar() && length > 1 {
            let validity = self.0.validity().map(PlBitmap::from);
            // The mask is dropped first so that the one element is read as a value whatever the
            // mask says of it, which is what the written-out path does as well.
            // SAFETY: the elements were valid UTF-8, and dropping the mask and slicing leaves
            // every one of them as it was; the caller's contract carries over to the closure.
            let single = unsafe {
                Self::from_binview_unchecked(self.0.clone().without_validity()).sliced(0, 1)
            };

            // SAFETY: as above.
            let mapped = unsafe { single.apply_views(update_view) };

            return mapped.new_from_index(0, length).with_validity(validity);
        }

        let flat = self.0.to_flat();
        let (views, buffers, validity) = flat.into_owned().into_inner();
        let validity = validity.map(PlBitmap::from_bitmap);

        let views: Vec<View> = views
            .as_slice()
            .iter()
            .enumerate()
            // SAFETY: `i` is in bounds of the array the views came from.
            .map(|(i, &view)| update_view(view, unsafe { self.value_unchecked(i) }))
            .collect();

        // SAFETY: the caller keeps every view reading bytes the buffers hold, and valid UTF-8.
        unsafe {
            Self::from_binview_unchecked(PlBinaryViewArray::new_unchecked(
                views.into(),
                buffers,
                length,
                validity,
            ))
        }
    }

    /// Borrows this array as a flat one, or `None` if any backing buffer is scalar.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: the inner array is flat, and the wrapper is transparent over it.
        self.0.as_flat().map(|_| unsafe { Flat::new_ref(self) })
    }
}

/// Checks that every element of `array`, including the ones masked off as null, is valid UTF-8.
fn validate_utf8(array: &PlBinaryViewArray) -> PolarsResult<()> {
    // The validity mask is dropped rather than honoured: a null element still holds bytes, and
    // replacing the mask must not be able to expose bytes that were never checked.
    for value in array
        .to_flat()
        .into_owned()
        .without_validity()
        .values_iter()
    {
        std::str::from_utf8(value).map_err(|e| polars_err!(ComputeError: "invalid utf8: {}", e))?;
    }
    Ok(())
}

impl Default for PlUtf8ViewArray {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

impl<'a> FromIterator<Option<&'a str>> for PlUtf8ViewArray {
    #[inline]
    fn from_iter<I: IntoIterator<Item = Option<&'a str>>>(iter: I) -> Self {
        // SAFETY: every value collected was a `&str`.
        unsafe { Self::from_binview_unchecked(iter.into_iter().collect()) }
    }
}

/// Compares two arrays element-wise, exactly like comparing the bytes under them.
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
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    #[inline]
    fn array_type(&self) -> PlArrayType {
        PlArrayType::Utf8View
    }

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        self.0.is_scalar()
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
    fn set_validity(&mut self, validity: Option<PlBitmap>) {
        self.0.set_validity(validity);
    }

    #[inline]
    unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Box<dyn PlArray> {
        // SAFETY: the caller keeps `index` in bounds. Repeating one element of a valid string
        // array keeps every element valid UTF-8.
        let array = unsafe { self.0.new_from_index_unchecked(index, length) };
        Box::new(unsafe { Self::from_binview_unchecked(array) })
    }

    #[inline]
    fn to_boxed(&self) -> Box<dyn PlArray> {
        Box::new(self.clone())
    }

    #[inline]
    fn eq_dyn(&self, other: &dyn PlArray) -> bool {
        // A string array equals another string array with the same elements; a byte array of the
        // same bytes is a different array type, which `PlBinaryViewArray::eq_dyn` also rejects.
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self == other)
    }
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use super::*;
    /// A value of more than [`View::MAX_INLINE_SIZE`] bytes, which no view inlines.
    const LONG: &str = "a value that is too long to inline";

    #[test]
    fn elements_are_the_strings_the_bytes_are() {
        let arr: PlUtf8ViewArray = [Some("foo"), None, Some(LONG)].into_iter().collect();

        assert_eq!(arr.len(), 3);
        assert!(arr.is_flat());
        assert_eq!(arr.null_count(), 1);
        assert_eq!(arr.value(0), "foo");
        assert_eq!(arr.get(0), Some("foo"));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.total_bytes_len(), 3 + LONG.len());
        assert_eq!(
            arr.iter().collect::<Vec<_>>(),
            [Some("foo"), None, Some(LONG)],
        );
        assert_eq!(arr.values_iter().rev().collect::<Vec<_>>().len(), 3);
        assert_eq!(unsafe { arr.value_unchecked(2) }, LONG);
        assert_eq!(unsafe { arr.get_unchecked(1) }, None);

        // A scalar array of a billion elements holds one view, as it does for the bytes.
        let scalar = PlUtf8ViewArray::new_scalar(LONG, 1_000_000_000);

        assert!(scalar.is_scalar());
        assert_eq!(scalar.scalar_value(), Some(Some(LONG)));
        assert_eq!(scalar.value(999_999_999), LONG);

        // A single element broadcasts to as many as the caller asks for.
        let one = PlUtf8ViewArray::new_scalar(LONG, 1);
        assert_eq!(one.broadcast_values_iter(3).collect::<Vec<_>>(), [LONG; 3]);
    }

    #[test]
    fn wrapping_bytes_as_strings_borrows_them() {
        let bytes = PlBinaryViewArray::from_values_iter([b"foo".as_slice(), b"bar"]);
        let arr = PlUtf8ViewArray::from_binview(bytes.clone()).expect("the bytes are UTF-8");

        assert_eq!(arr.value(1), "bar");
        assert_eq!(arr.as_binview(), &bytes);
        assert!(
            arr.data_buffers().is_same_buffer(bytes.data_buffers()),
            "the wrapper must share the buffers, not copy them",
        );
    }

    /// Truncates every element to its first `keep` bytes, which is what a `str.head` does.
    ///
    /// `keep` stays above [`View::MAX_INLINE_SIZE`] so that a view of the buffer stays one: the
    /// prefix a longer view already carries is the prefix of what is kept.
    fn head(arr: &PlUtf8ViewArray, keep: u32, calls: &mut usize) -> PlUtf8ViewArray {
        assert!(keep > View::MAX_INLINE_SIZE);
        // SAFETY: a prefix of a view reads bytes the same buffer already holds, and the values
        // here are ASCII, so a prefix of one is valid UTF-8 too.
        unsafe {
            arr.apply_views(|mut view, _| {
                *calls += 1;
                view.length = view.length.min(keep);
                view
            })
        }
    }

    #[test]
    fn a_scalar_array_maps_its_one_view() {
        let mut calls = 0;
        let out = head(&PlUtf8ViewArray::new_scalar(LONG, 1_000), 20, &mut calls);

        assert_eq!(calls, 1, "the one view every element reads is mapped once");
        assert_eq!(out.len(), 1_000);
        assert!(out.is_scalar(), "{out:?}");
        assert_eq!(out.scalar_value(), Some(Some(&LONG[..20])));

        // A mask over the scalar views is kept, element for element.
        let arr = PlUtf8ViewArray::new_scalar(LONG, 4).with_validity(Some(PlBitmap::from_bitmap(
            Bitmap::from_iter([true, false, true, false]),
        )));
        let mut calls = 0;
        let out = head(&arr, 20, &mut calls);

        assert_eq!(calls, 1);
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            [Some(&LONG[..20]), None, Some(&LONG[..20]), None],
        );

        // A flat array is mapped view by view, as before.
        let arr: PlUtf8ViewArray = [Some(LONG), None, Some(&LONG[10..])].into_iter().collect();
        let mut calls = 0;
        let out = head(&arr, 20, &mut calls);

        assert_eq!(calls, 3);
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            [Some(&LONG[..20]), None, Some(&LONG[10..30])],
        );
    }

    #[test]
    fn a_scalar_array_is_written_out_flat() {
        let scalar = PlUtf8ViewArray::new_scalar(LONG, 3);

        assert!(scalar.as_flat().is_none());
        let flat = scalar.to_flat();
        assert!(flat.is_flat());
        assert_eq!(*flat, scalar);
        assert_eq!(flat.value(2), LONG);

        // An array that is already flat is borrowed rather than written out again.
        let arr: PlUtf8ViewArray = [Some("foo")].into_iter().collect();
        let flat = arr.as_flat().expect("the array is flat");
        assert_eq!(flat.value(0), "foo");
    }
}
