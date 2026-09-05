use std::borrow::Cow;

use arrow::array::View;
use arrow::bitmap::{Bitmap, BitmapBuilder};
use buffers::{copy_only_value, copy_value};
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmap, PlBitmapRef};
use crate::broadcast::{
    ArrayRepr, assert_broadcastable, broadcast_index, is_flat_buffer_len, is_scalar_buffer_len,
    normalize_buffer, scalar_buffer_len, slice_buffer, slice_validity, try_validity_covering,
    validity_covering, validity_covering_unchecked,
};
use crate::flat::Flat;

mod buffers;
mod builder;
mod flat;
mod iterator;

pub use builder::PlBinaryViewArrayBuilder;
pub use iterator::{PlBinaryViewIter, PlBinaryViewValuesIter};

/// An immutable, cheaply cloneable sequence of `length` optional byte slices.
#[derive(Clone)]
pub struct PlBinaryViewArray {
    /// Scalar: views.len() == 1
    views: Buffer<View>,
    /// A side table indexed by the views, so neither flat nor scalar for `length`.
    buffers: Buffer<Buffer<u8>>,
    length: usize,
    /// Scalar: validity.len() == 1
    validity: Option<Bitmap>,
}

impl PlBinaryViewArray {
    /// Creates a flat [`PlBinaryViewArray`] out of its internal components.
    ///
    /// # Errors
    /// This function errors if `views` does not hold exactly `length` slots, if `validity` does not
    /// cover exactly `length` elements, or if a
    /// view does not read bytes that `buffers` holds.
    pub fn try_new(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> PolarsResult<Self> {
        let validity = try_validity_covering(validity, length)?;
        polars_ensure!(
            is_flat_buffer_len(views.len(), length),
            ComputeError:
            "views buffer of length {} is not flat for an array of length {}",
            views.len(), length,
        );

        validate_views(&views, &buffers)?;

        Ok(Self {
            views,
            buffers,
            length,
            validity,
        })
    }

    /// Creates a flat [`PlBinaryViewArray`] out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        Self::try_new(views, buffers, length, validity).unwrap()
    }

    /// Creates a flat [`PlBinaryViewArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// `views` must hold exactly `length` slots, `validity` must cover exactly `length` elements in
    /// either representation, and every view must read bytes
    /// that `buffers` holds.
    #[inline]
    pub unsafe fn new_unchecked(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        let validity = validity_covering_unchecked(validity, length);
        if cfg!(debug_assertions) {
            assert!(is_flat_buffer_len(views.len(), length));
            validate_views(&views, &buffers).unwrap();
        }

        Self {
            views,
            buffers,
            length,
            validity,
        }
    }

    /// Creates a scalar [`PlBinaryViewArray`] of `length` elements out of its internal components.
    ///
    /// # Errors
    /// This function errors if `views` is not scalar for `length`, per
    /// [`is_scalar_buffer_len`], or if `validity` does not cover exactly `length` elements, per
    /// [`is_scalar_buffer_len`], or if the view does not read bytes that `buffers` holds.
    pub fn try_new_broadcast(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> PolarsResult<Self> {
        let validity = try_validity_covering(validity, length)?;
        polars_ensure!(
            is_scalar_buffer_len(views.len(), length),
            ComputeError:
            "views buffer of length {} is not the single view the {} elements of a broadcast \
             array share",
            views.len(), length,
        );

        validate_views(&views, &buffers)?;

        Ok(Self {
            views: normalize_buffer(views, length),
            buffers,
            length,
            validity,
        })
    }

    /// Creates a scalar [`PlBinaryViewArray`] of `length` elements out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new_broadcast`] errors.
    #[inline]
    pub fn new_broadcast(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        Self::try_new_broadcast(views, buffers, length, validity).unwrap()
    }

    /// Creates a scalar [`PlBinaryViewArray`] of `length` elements out of its internal components
    /// without validating them.
    ///
    /// # Safety
    /// `views` must be scalar for `length`, per [`is_scalar_buffer_len`], `validity` must cover
    /// exactly `length` elements in either representation, and
    /// every view must read bytes that `buffers` holds.
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        let validity = validity_covering_unchecked(validity, length);
        if cfg!(debug_assertions) {
            assert!(is_scalar_buffer_len(views.len(), length));
            validate_views(&views, &buffers).unwrap();
        }

        Self {
            views: normalize_buffer(views, length),
            buffers,
            length,
            validity,
        }
    }

    /// Creates an empty [`PlBinaryViewArray`].
    #[inline]
    pub fn new_empty() -> Self {
        Self {
            views: Buffer::new(),
            buffers: Buffer::new(),
            length: 0,
            validity: None,
        }
    }

    /// Creates a flat, fully valid [`PlBinaryViewArray`] from `views` over `buffers`.
    ///
    /// # Panics
    /// Panics if a view does not read bytes that `buffers` holds.
    #[inline]
    pub fn from_views(views: Buffer<View>, buffers: Buffer<Buffer<u8>>) -> Self {
        let length = views.len();
        Self::new(views, buffers, length, None)
    }

    /// Creates a flat, fully valid [`PlBinaryViewArray`] holding `values`, in order.
    ///
    /// # Panics
    /// Panics if a value is longer than
    /// [`BINVIEW_MAX_ROW_BYTE_LEN`](arrow::array::BINVIEW_MAX_ROW_BYTE_LEN) bytes.
    pub fn from_values_iter<V: AsRef<[u8]>, I: IntoIterator<Item = V>>(values: I) -> Self {
        let values = values.into_iter();
        let (lower, _) = values.size_hint();

        let mut views = Vec::with_capacity(lower);
        let mut buffers = Vec::new();
        for value in values {
            views.push(copy_value(&mut buffers, 0, value.as_ref()));
        }

        let length = views.len();
        // SAFETY: there is one view per element, and every one of them was just written over the
        // buffers it reads.
        unsafe { Self::new_unchecked(Buffer::from(views), collect_buffers(buffers), length, None) }
    }

    /// Creates a [`PlBinaryViewArray`] of `length` copies of `value`, in `O(value.len())` memory.
    ///
    /// # Panics
    /// Panics if `value` is longer than
    /// [`BINVIEW_MAX_ROW_BYTE_LEN`](arrow::array::BINVIEW_MAX_ROW_BYTE_LEN) bytes.
    pub fn new_scalar(value: &[u8], length: usize) -> Self {
        // There is no element for the value to be shared by when there are no elements at all,
        // which is why an empty array is the one that keeps nothing of the value it repeats.
        if length == 0 {
            return Self::new_empty();
        }

        // The one value is all the array ever holds, so its bytes are copied into a buffer that
        // fits them exactly: a scalar array costs what the value costs, and no block more.
        let (view, buffers) = copy_only_value(value);

        Self {
            views: Buffer::from_owner([view]),
            buffers: collect_buffers(buffers),
            length,
            validity: None,
        }
    }

    /// Creates a [`PlBinaryViewArray`] of `length` nulls, in `O(1)` memory.
    #[inline]
    pub fn new_full_null(length: usize) -> Self {
        Self {
            // A zeroed view holds no bytes at all, which is a view like any other: the value of a
            // null element is undetermined, so it need not be written out.
            views: Buffer::zeroed(scalar_buffer_len(length)),
            buffers: Buffer::new(),
            length,
            validity: Some(Bitmap::new_zeroed(scalar_buffer_len(length))),
        }
    }

    /// The number of elements in this array.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Whether this array holds no elements.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Which representation the backing views buffer is in, along with what it holds.
    #[inline]
    pub fn views_repr(&self) -> ArrayRepr<&Buffer<View>, View> {
        if self.views_are_scalar() {
            ArrayRepr::Scalar(self.views[0])
        } else {
            ArrayRepr::Flat(&self.views)
        }
    }

    /// The backing views buffer, if it holds one slot per element.
    #[inline]
    pub fn flat_views(&self) -> Option<&Buffer<View>> {
        self.views_repr().flat()
    }

    /// The backing views buffer, if it holds one slot per element.
    ///
    /// # Safety
    /// Every view left in the buffer must still read bytes that [`data_buffers`](Self::data_buffers)
    /// holds, and the buffer must be left as long as it was found: a view is an index into the
    /// buffers, so nothing else checks it again once the array is built.
    #[inline]
    pub unsafe fn flat_views_mut(&mut self) -> Option<&mut Buffer<View>> {
        if self.views_are_scalar() {
            // A single view stands for every element; there is nothing laid out per element to
            // write into.
            None
        } else {
            Some(&mut self.views)
        }
    }

    /// The view every element of this array reads, if the views buffer holds a single slot.
    #[inline]
    pub fn scalar_views(&self) -> Option<View> {
        self.views_repr().scalar()
    }

    /// The buffers the views that do not inline their bytes point into.
    #[inline(always)]
    pub const fn data_buffers(&self) -> &Buffer<Buffer<u8>> {
        &self.buffers
    }

    /// The buffers the views that do not inline their bytes point into.
    ///
    /// # Safety
    /// Every view of this array must still read bytes the buffers hold once they are written. A
    /// buffer may be appended without reading the views, since that leaves every existing index
    /// pointing where it did; removing or reordering one does not.
    #[inline]
    pub unsafe fn data_buffers_mut(&mut self) -> &mut Buffer<Buffer<u8>> {
        &mut self.buffers
    }

    /// The validity mask, if any element may be null.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_broadcast_unchecked(validity, self.length) })
    }

    /// Whether the views buffer holds a single view shared by every element.
    ///
    /// An array of no elements holds no such view: it keeps the empty buffer in place of the one
    /// slot a scalar buffer would, and is flat.
    #[inline]
    pub fn views_are_scalar(&self) -> bool {
        self.views.len() == 1 && self.length > 0
    }

    /// Whether the views buffer holds one slot per element.
    #[inline]
    pub fn views_are_flat(&self) -> bool {
        self.views.len() == self.length
    }

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether every backing buffer has one slot per element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.views_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array is entirely stored in the scalar representation, and therefore is a
    /// single logical value repeated [`Self::len`] times in `O(1)` memory.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.views_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element of this array equals, if every backing buffer holds one
    /// slot.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<&[u8]>> {
        let is_shared = self.views.len() == 1
            && self
                .validity
                .as_ref()
                .is_none_or(|validity| validity.len() == 1);

        // SAFETY: the array is not empty, so element 0 is in bounds.
        (is_shared && self.length > 0).then(|| unsafe { self.get_unchecked(0) })
    }

    /// Returns the view of the element at `i`.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn view(&self, i: usize) -> View {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.view_unchecked(i) }
    }

    /// Returns the view of the element at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn view_unchecked(&self, i: usize) -> View {
        debug_assert!(i < self.length);
        unsafe {
            *self
                .views
                .get_unchecked(broadcast_index(i, self.views.len()))
        }
    }

    /// Returns the value at `i`.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value(&self, i: usize) -> &[u8] {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the value at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        debug_assert!(i < self.length);
        // SAFETY: every view reads bytes the data buffers hold, upheld by every constructor.
        unsafe {
            self.views
                .get_unchecked(broadcast_index(i, self.views.len()))
                .get_slice_unchecked(self.buffers.as_slice())
        }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.is_valid_unchecked(i) }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        debug_assert!(i < self.length);
        // SAFETY: `i` is in bounds of the array, and therefore of its validity mask.
        self.validity()
            .is_none_or(|validity| unsafe { validity.get_unchecked(i) })
    }

    /// Returns whether the element at `i` is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        !self.is_valid(i)
    }

    /// Returns whether the element at `i` is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        unsafe { !self.is_valid_unchecked(i) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> Option<&[u8]> {
        unsafe { self.is_valid_unchecked(i).then(|| self.value_unchecked(i)) }
    }

    /// The number of null elements.
    ///
    /// Inlined so that an array with no mask to count is answered without a call at all: one left
    /// standing is an opaque write as far as the compiler is concerned, and sinks behind it every
    /// fact the caller had established about the array — the representation of its buffers
    /// included — which is exactly what a caller asking [`Self::has_nulls`] ahead of a walk is
    /// trying to hand the walk.
    #[inline]
    pub fn null_count(&self) -> usize {
        self.validity().map_or(0, |validity| validity.unset_bits())
    }

    /// Whether this array has at least one null element.
    #[inline]
    pub fn has_nulls(&self) -> bool {
        self.null_count() > 0
    }

    /// The number of bytes it would take to lay the values of the valid elements end to end.
    ///
    /// # Panics
    /// Panics if the total overflows a `usize`, which the scalar representation makes possible
    /// without the memory to back it.
    pub fn total_bytes_len(&self) -> usize {
        if self.views_are_scalar() {
            let valid = self
                .validity()
                .map_or(self.length, |validity| validity.set_bits());

            return (self.views[0].length as usize)
                .checked_mul(valid)
                .expect("the total length of the values overflows a `usize`");
        }

        match self.validity() {
            None => self.views.iter().map(|view| view.length as usize).sum(),
            Some(validity) => self
                .views
                .iter()
                .zip(validity)
                .filter(|(_, is_valid)| *is_valid)
                .map(|(view, _)| view.length as usize)
                .sum(),
        }
    }

    /// The number of bytes the data buffers hold.
    pub fn total_buffer_len(&self) -> usize {
        self.buffers.iter().map(|buffer| buffer.len()).sum()
    }

    /// Returns an iterator over the values, ignoring validity.
    #[inline]
    pub fn values_iter(&self) -> PlBinaryViewValuesIter<'_> {
        PlBinaryViewValuesIter::new(&self.views, &self.buffers, self.length)
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlBinaryViewIter<'_> {
        PlBinaryViewIter::new(&self.views, &self.buffers, self.validity(), self.length)
    }

    /// Returns an iterator over `length` values, repeating the single value of this array if that
    /// is all it holds, and ignoring validity.
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlBinaryViewValuesIter<'_> {
        assert_broadcastable(self.length, length);
        // SAFETY: a single view is scalar for any length, and otherwise the views are already
        // valid for `length`; either way every view reads bytes the buffers hold.
        PlBinaryViewValuesIter::new(&self.views, &self.buffers, length)
    }

    /// Returns this array with its validity mask replaced.
    ///
    /// The mask keeps whichever representation it is in: one that stands for a single bit shared
    /// by every element is not written out one bit per element to be set.
    ///
    /// # Panics
    /// Panics unless `validity` covers exactly [`len`](Self::len) elements.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<PlBitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Replaces the validity mask, which keeps the representation it is in.
    ///
    /// # Panics
    /// Panics unless `validity` covers exactly [`len`](Self::len) elements.
    pub fn set_validity(&mut self, validity: Option<PlBitmap>) {
        let length = self.len();
        self.validity = validity_covering(validity, length);
    }

    /// Drops the validity mask, making every element valid.
    #[must_use]
    pub fn without_validity(mut self) -> Self {
        self.validity = None;
        self
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Panics
    /// Panics if `offset + length > self.len()`.
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.length,
            "the offset of the new slice must be smaller than the length of the array",
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        debug_assert!(offset + length <= self.length);

        unsafe {
            slice_buffer(&mut self.views, self.length, offset, length);
            slice_validity(&mut self.validity, self.length, offset, length);
        }

        self.length = length;
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// # Panics
    /// Panics if `offset + length > self.len()`.
    #[must_use]
    pub fn sliced(mut self, offset: usize, length: usize) -> Self {
        self.slice(offset, length);
        self
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    #[must_use]
    pub unsafe fn sliced_unchecked(mut self, offset: usize, length: usize) -> Self {
        unsafe { self.slice_unchecked(offset, length) };
        self
    }

    /// Creates a [`PlBinaryViewArray`] of `length` copies of the element at `index`.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        assert!(index < self.length, "index out of bounds");
        unsafe { self.new_from_index_unchecked(index, length) }
    }

    /// Creates a [`PlBinaryViewArray`] of `length` copies of the element at `index`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        if unsafe { self.is_null_unchecked(index) } {
            return Self::new_full_null(length);
        }

        if length == 0 {
            return Self::new_empty();
        }

        let view = unsafe { self.view_unchecked(index) };

        // The one buffer the view reads is all the result needs, which is what it is rebased onto;
        // a view that inlines its bytes needs no buffer at all.
        let (view, buffers) = if view.is_inline() {
            (view, Buffer::new())
        } else {
            let buffer = self.buffers[view.buffer_idx as usize].clone();
            let view = View {
                buffer_idx: 0,
                ..view
            };
            (view, Buffer::from_owner([buffer]))
        };

        Self {
            views: Buffer::from_owner([view]),
            buffers,
            length,
            validity: None,
        }
    }

    /// Returns an equivalent array whose views and mask both hold one slot per element, borrowing
    /// this array itself if they already do.
    pub fn to_flat(&self) -> Cow<'_, Flat<Self>> {
        if let Some(flat) = self.as_flat() {
            return Cow::Borrowed(flat);
        }

        let views = if self.views_are_flat() {
            self.views.clone()
        } else if self.length == 0 {
            Buffer::new()
        } else if self.scalar_value() == Some(None) {
            // Every element is null, and the value of a null element is undetermined, so the
            // repeated view need not be written out: a zeroed one, which holds no bytes at all,
            // stands in for it.
            Buffer::zeroed(self.length)
        } else {
            Buffer::from(vec![self.views[0]; self.length])
        };

        let validity = self
            .validity()
            .map(|validity| validity.to_flat().into_owned());

        // SAFETY: the views hold one slot per element, written out above, and the mask is the
        // flat counterpart of this array's own.
        Cow::Owned(unsafe {
            Flat::new(Self {
                views,
                buffers: self.buffers.clone(),
                length: self.length,
                validity,
            })
        })
    }

    /// Borrows this array as a [`Flat`] one, if its views and mask already hold one slot per
    /// element.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: the views and the mask of a flat array hold one slot per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }
}

/// Validates that every view of `views` reads bytes that `buffers` holds.
fn validate_views(views: &[View], buffers: &[Buffer<u8>]) -> PolarsResult<()> {
    for view in views {
        if let Some(inlined) = view.get_inlined_slice() {
            // The bytes past the inlined ones are padding rather than value, so they have to be
            // zeroed for two views over the same bytes to be the same 16 bytes.
            if view.length < View::MAX_INLINE_SIZE && view.as_u128() >> (32 + view.length * 8) != 0
            {
                polars_bail!(
                    ComputeError:
                    "view of {} inlined bytes holds non-zero padding past them", inlined.len(),
                );
            }
            continue;
        }

        let buffer = buffers.get(view.buffer_idx as usize).ok_or_else(|| {
            polars_err!(
                OutOfBounds:
                "view points at data buffer {} of {} buffers", view.buffer_idx, buffers.len(),
            )
        })?;

        let start = view.offset as usize;
        let end = start + view.length as usize;
        let bytes = buffer.as_slice().get(start..end).ok_or_else(|| {
            polars_err!(
                OutOfBounds:
                "view covers bytes {}..{} of a data buffer of {} bytes",
                start, end, buffer.len(),
            )
        })?;

        polars_ensure!(
            bytes.starts_with(&view.prefix.to_le_bytes()),
            ComputeError: "view holds a prefix that the bytes it points at do not start with",
        );
    }

    Ok(())
}

/// The data buffers a view builder filled, as the buffer of buffers an array holds them in.
fn collect_buffers(buffers: Vec<Vec<u8>>) -> Buffer<Buffer<u8>> {
    buffers.into_iter().map(Buffer::from).collect()
}

impl Default for PlBinaryViewArray {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

impl<V: AsRef<[u8]>> FromIterator<Option<V>> for PlBinaryViewArray {
    fn from_iter<I: IntoIterator<Item = Option<V>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut views = Vec::with_capacity(lower);
        let mut buffers = Vec::new();
        let mut validity = BitmapBuilder::with_capacity(lower);

        for value in iter {
            match value {
                Some(value) => {
                    views.push(copy_value(&mut buffers, 0, value.as_ref()));
                    validity.push(true);
                },
                // The value of a null element is undetermined, so nothing is written out for it.
                None => {
                    views.push(View::default());
                    validity.push(false);
                },
            }
        }

        let length = views.len();
        // SAFETY: there is one view per element and one bit per element, and every view was just
        // written over the buffers it reads.
        unsafe {
            Self::new_unchecked(
                Buffer::from(views),
                collect_buffers(buffers),
                length,
                validity.into_opt_validity().map(PlBitmap::from_bitmap),
            )
        }
    }
}

impl<'a> IntoIterator for &'a PlBinaryViewArray {
    type Item = Option<&'a [u8]>;
    type IntoIter = PlBinaryViewIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares two arrays element-wise, disregarding the representation and how the bytes are
/// reached.
impl PartialEq for PlBinaryViewArray {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }

        // Never walk two scalar arrays element by element: their length is unbounded by their
        // memory use.
        if let (Some(lhs), Some(rhs)) = (self.scalar_value(), other.scalar_value()) {
            return lhs == rhs;
        }

        self.iter().eq(other.iter())
    }
}

impl Eq for PlBinaryViewArray {}

impl std::fmt::Debug for PlBinaryViewArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// Renders nulls as `null` instead of `None`, and the bytes of a value as they are.
        struct Element<'a>(Option<&'a [u8]>);

        impl std::fmt::Debug for Element<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    Some(value) => value.fmt(f),
                    None => f.write_str("null"),
                }
            }
        }

        f.write_str("PlBinaryViewArray")?;

        // Never materialize a scalar array: its length is unbounded by its memory use.
        if self.length > 1 {
            if let Some(element) = self.scalar_value() {
                return write!(f, "[{:?}; {}]", Element(element), self.length);
            }
        }

        f.debug_list().entries(self.iter().map(Element)).finish()
    }
}

impl PlArray for PlBinaryViewArray {
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn array_type(&self) -> PlArrayType {
        PlArrayType::BinaryView
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        self.is_scalar()
    }

    #[inline]
    fn validity(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    #[inline]
    fn slice(&mut self, offset: usize, length: usize) {
        self.slice(offset, length)
    }

    #[inline]
    unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        unsafe { self.slice_unchecked(offset, length) }
    }

    #[inline]
    fn set_validity(&mut self, validity: Option<PlBitmap>) {
        self.set_validity(validity)
    }

    #[inline]
    unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Box<dyn PlArray> {
        Box::new(unsafe { self.new_from_index_unchecked(index, length) })
    }

    #[inline]
    fn to_boxed(&self) -> Box<dyn PlArray> {
        Box::new(self.clone())
    }

    fn eq_dyn(&self, other: &dyn PlArray) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self == other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A value of more than [`View::MAX_INLINE_SIZE`] bytes, which no view inlines.
    const LONG: &[u8] = b"a value that is too long to inline";

    #[test]
    fn flat() {
        let arr = PlBinaryViewArray::from_values_iter([b"foo".as_slice(), b"bar", LONG]);

        assert_eq!(arr.len(), 3);
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert_eq!(arr.null_count(), 0);
        assert_eq!(arr.value(1), b"bar");
        assert_eq!(arr.get(2), Some(LONG));
        assert_eq!(
            arr.iter().collect::<Vec<_>>(),
            [Some(b"foo".as_slice()), Some(b"bar"), Some(LONG)],
        );
        assert_eq!(
            arr.values_iter().collect::<Vec<_>>(),
            [b"foo".as_slice(), b"bar", LONG],
        );
    }

    #[test]
    fn scalar_shares_one_view() {
        let arr = PlBinaryViewArray::new_scalar(b"foo", 4);

        assert_eq!(arr.len(), 4);
        assert!(arr.flat_views().is_none());
        assert!(arr.scalar_views().is_some());
        assert!(arr.is_scalar());
        assert!(!arr.is_flat());
        assert_eq!(arr.scalar_value(), Some(Some(b"foo".as_slice())));
        assert_eq!(arr.null_count(), 0);

        for i in 0..arr.len() {
            assert_eq!(arr.get(i), Some(b"foo".as_slice()));
        }
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(b"foo".as_slice()); 4]);
        assert_eq!(
            arr.values_iter().rev().collect::<Vec<_>>(),
            [b"foo".as_slice(); 4],
        );

        // The bytes are held once however many elements stand for them.
        let arr = PlBinaryViewArray::new_scalar(LONG, 1_000_000_000);

        assert_eq!(arr.total_buffer_len(), LONG.len());
        assert_eq!(arr.total_bytes_len(), LONG.len() * 1_000_000_000);
        assert_eq!(arr.value(999_999_999), LONG);
    }

    #[test]
    fn slicing_a_flat_array_slices_its_views_but_not_its_buffers() {
        let arr: PlBinaryViewArray = [Some(b"foo".as_slice()), None, Some(LONG), Some(b"baz")]
            .into_iter()
            .collect();
        let sliced = arr.clone().sliced(1, 2);

        assert_eq!(sliced.len(), 2);
        assert_eq!(sliced.flat_views().unwrap().len(), 2);
        assert_eq!(sliced.validity().unwrap().len(), 2);
        assert_eq!(sliced.iter().collect::<Vec<_>>(), [None, Some(LONG)]);

        // The bytes of the elements that were sliced away are still held onto.
        let sliced = arr.sliced(0, 1);
        assert_eq!(sliced.total_buffer_len(), LONG.len());
        assert_eq!(sliced.total_bytes_len(), 3);
    }

    #[test]
    fn an_array_of_no_elements_keeps_no_view() {
        // A single slot is scalar for no elements too, but there is no element left to read it, so
        // it is not kept: the array is flat, like every empty array, rather than scalar.
        let arr = PlBinaryViewArray::new_broadcast(
            Buffer::zeroed(1),
            Buffer::new(),
            0,
            Some(PlBitmap::new_scalar(false, 0)),
        );

        assert!(arr.is_empty());
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert!(arr.flat_views().unwrap().is_empty());
        assert!(arr.validity().unwrap().is_empty());
    }
}
