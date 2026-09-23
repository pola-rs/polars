//! The string view of a [`PlBinaryViewArray`].

use std::any::Any;
use std::borrow::Cow;

use polars_arrow::array::View;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_err};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::binview::PlBinaryViewArray;
use crate::bitmap::{PlBitmap, PlBitmapRef};
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

    /// [`Self::new_scalar`], taking over the allocation `value` already holds its bytes in.
    #[inline]
    pub fn new_scalar_owned(value: String, length: usize) -> Self {
        // SAFETY: the bytes of a `String` are valid UTF-8, and they are every element's value.
        unsafe {
            Self::from_binview_unchecked(PlBinaryViewArray::new_scalar_owned(
                value.into_bytes(),
                length,
            ))
        }
    }

    /// Returns the element at `i`, whether or not it is null.
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

    /// Whether this array is scalar throughout — see [`PlBinaryViewArray::is_scalar`].
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
    #[inline]
    #[must_use]
    pub fn with_validity(self, validity: Option<PlBitmap>) -> Self {
        Self(self.0.with_validity(validity))
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    #[inline]
    #[must_use]
    pub fn sliced(&self, offset: usize, length: usize) -> Self {
        Self(self.0.sliced(offset, length))
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    #[inline]
    #[must_use]
    pub unsafe fn sliced_unchecked(&self, offset: usize, length: usize) -> Self {
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

    /// The view every element of this array reads, if the views buffer holds a single slot.
    #[inline]
    pub fn scalar_views(&self) -> Option<View> {
        self.0.scalar_views()
    }

    /// The string every element of this array reads, if the views buffer holds a single slot.
    #[inline]
    pub fn scalar_value_ignore_validity(&self) -> Option<&str> {
        // SAFETY: the elements of this array are valid UTF-8.
        self.0
            .scalar_value_ignore_validity()
            .map(|v| unsafe { std::str::from_utf8_unchecked(v) })
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

    /// Returns this array in the flat representation, borrowing it if it is already flat.
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
    /// Every view handed back must read bytes this array's data buffers hold, valid as UTF-8.
    pub unsafe fn apply_views<F: FnMut(View, &str) -> View>(&self, mut update_view: F) -> Self {
        let length = self.0.len();
        if self.0.views_are_scalar() && length > 1 {
            let validity = self.0.validity().map(PlBitmap::from);
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
            .map(|&view| {
                // SAFETY: the view is one of this array's, so it reads bytes the buffers hold,
                // and every one of them is valid UTF-8.
                let value = unsafe {
                    std::str::from_utf8_unchecked(view.get_slice_unchecked(buffers.as_slice()))
                };
                update_view(view, value)
            })
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
    fn check(value: &[u8]) -> PolarsResult<()> {
        std::str::from_utf8(value)
            .map(|_| ())
            .map_err(|e| polars_err!(ComputeError: "invalid utf8: {}", e))
    }

    if let Some(value) = array.scalar_value_ignore_validity() {
        return check(value);
    }

    for value in array.values_iter() {
        check(value)?;
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

crate::impl_into_iterator!(PlUtf8ViewArray, PlUtf8ViewIter<'a>);

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

    fn new_full_null_like_self(&self, length: usize) -> Box<dyn PlArray> {
        Box::new(Self::new_full_null(length))
    }

    #[inline]
    fn eq_dyn(&self, other: &dyn PlArray) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self == other)
    }
}
