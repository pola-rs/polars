use arrow::array::View;
use arrow::bitmap::{Bitmap, BitmapBuilder};
use buffers::{copy_only_value, copy_value};
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::PlBitmapRef;
use crate::broadcast::{
    assert_broadcastable, broadcast_index, is_flat_buffer_len, is_scalar_buffer_len,
    is_valid_buffer_len, scalar_buffer_len,
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
    /// Every backing buffer has to hold one slot per element. [`Self::try_new_broadcast`] is what
    /// builds the scalar representation; this function never infers it from a buffer that happens
    /// to hold a single slot. Unlike the `try_new` of the other arrays, this is `O(views)` rather
    /// than `O(1)`: every view is checked against the buffers it reads.
    ///
    /// # Errors
    /// This function errors if `views` or `validity` does not hold exactly `length` slots, or if a
    /// view does not read bytes that `buffers` holds.
    pub fn try_new(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_flat_buffer_len(views.len(), length),
            ComputeError:
            "views buffer of length {} is not flat for an array of length {}",
            views.len(), length,
        );

        if let Some(validity) = validity.as_ref() {
            polars_ensure!(
                is_flat_buffer_len(validity.len(), length),
                ComputeError:
                "validity mask of length {} is not flat for an array of length {}",
                validity.len(), length,
            );
        }

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
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new(views, buffers, length, validity).unwrap()
    }

    /// Creates a flat [`PlBinaryViewArray`] out of its internal components without validating
    /// them.
    ///
    /// This is the `O(1)` counterpart of [`Self::try_new`], which walks every view.
    ///
    /// # Safety
    /// `views` and `validity` must each hold exactly `length` slots, and every view must read
    /// bytes that `buffers` holds.
    #[inline]
    pub unsafe fn new_unchecked(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        if cfg!(debug_assertions) {
            assert!(is_flat_buffer_len(views.len(), length));
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_flat_buffer_len(v.len(), length))
            );
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
    /// Every backing buffer has to hold the single value every element shares, which makes this
    /// `O(1)` in `length`. [`Self::try_new`] is what builds the flat representation.
    ///
    /// # Errors
    /// This function errors if `views` or `validity` does not hold exactly one slot, or if the
    /// view does not read bytes that `buffers` holds. An array of no elements reads no slot at
    /// all, so it additionally admits an empty buffer.
    pub fn try_new_broadcast(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_scalar_buffer_len(views.len(), length),
            ComputeError:
            "views buffer of length {} is not the single view the {} elements of a broadcast \
             array share",
            views.len(), length,
        );

        if let Some(validity) = validity.as_ref() {
            polars_ensure!(
                is_scalar_buffer_len(validity.len(), length),
                ComputeError:
                "validity mask of length {} is not the single bit the {} elements of a broadcast \
                 array share",
                validity.len(), length,
            );
        }

        validate_views(&views, &buffers)?;

        Ok(Self {
            views,
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
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new_broadcast(views, buffers, length, validity).unwrap()
    }

    /// Creates a scalar [`PlBinaryViewArray`] of `length` elements out of its internal components
    /// without validating them.
    ///
    /// # Safety
    /// `views` and `validity` must each hold exactly one slot, or none at all if `length` is zero,
    /// and every view must read bytes that `buffers` holds.
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        if cfg!(debug_assertions) {
            assert!(is_scalar_buffer_len(views.len(), length));
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_scalar_buffer_len(v.len(), length))
            );
            validate_views(&views, &buffers).unwrap();
        }

        Self {
            views,
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
    /// The length is the length of the views, which is what makes the result flat.
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
    /// The values that are too long to be inlined into a view are copied into the data buffers of
    /// the result, so this is `O(total bytes)`.
    ///
    /// # Panics
    /// Panics if a value is longer than
    /// [`BINVIEW_MAX_ROW_BYTE_LEN`](arrow::array::BINVIEW_MAX_ROW_BYTE_LEN) bytes, which no view
    /// can point at.
    ///
    /// # Example
    /// ```
    /// use polars_array::PlBinaryViewArray;
    ///
    /// let arr = PlBinaryViewArray::from_values_iter([b"foo".as_slice(), b"bar"]);
    /// assert_eq!(arr.len(), 2);
    /// assert_eq!(arr.null_count(), 0);
    /// assert_eq!(arr.value(0), b"foo");
    /// ```
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
    /// [`BINVIEW_MAX_ROW_BYTE_LEN`](arrow::array::BINVIEW_MAX_ROW_BYTE_LEN) bytes, which no view
    /// can point at.
    ///
    /// # Example
    /// ```
    /// use polars_array::PlBinaryViewArray;
    ///
    /// // The bytes are inlined into the one view all billion elements share.
    /// let arr = PlBinaryViewArray::new_scalar(b"foo", 1_000_000_000);
    /// assert!(arr.is_scalar());
    /// assert!(arr.data_buffers().is_empty());
    ///
    /// // Bytes that do not fit in a view are held by a single data buffer.
    /// let arr = PlBinaryViewArray::new_scalar(b"a value too long to inline", 1_000_000_000);
    /// assert_eq!(arr.data_buffers().len(), 1);
    /// assert_eq!(arr.value(999_999_999), b"a value too long to inline");
    /// ```
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

    /// The backing views buffer, if it holds one slot per element.
    ///
    /// Slot `i` is then the view of element `i`, with no [`broadcast_index`] in the way. This is
    /// the `O(1)` counterpart of [`Self::to_flat`]: it materializes nothing, and returns `None`
    /// rather than writing out a scalar buffer. Reach for the view a scalar buffer shares with
    /// [`Self::scalar_views`] instead — between them the two cover every array that has elements
    /// at all, so a `None` from both is an empty array. The views of null elements are
    /// undetermined (they can be anything that reads bytes this array holds).
    #[inline]
    pub fn flat_views(&self) -> Option<&Buffer<View>> {
        self.views_are_flat().then_some(&self.views)
    }

    /// The view every element of this array reads, if the views buffer holds a single slot.
    ///
    /// This is the views half of [`Self::scalar_value`], which additionally asks that the validity
    /// mask be scalar and reports the null the mask makes of this view. Returns `None` for a views
    /// buffer that is flat over more than one element, and for an empty array, which has no
    /// element to share a view. The view of a null element is undetermined (it can be anything
    /// that reads bytes this array holds).
    #[inline]
    pub fn scalar_views(&self) -> Option<View> {
        (self.views_are_scalar() && self.length > 0).then(|| self.views[0])
    }

    /// The buffers the views that do not inline their bytes point into.
    ///
    /// These are indexed by the views rather than by an element index, so — unlike the views
    /// themselves — they are neither flat nor scalar for this array's length, and being
    /// [`flat`](Self::is_flat) says nothing about them.
    #[inline(always)]
    pub const fn data_buffers(&self) -> &Buffer<Buffer<u8>> {
        &self.buffers
    }

    /// The validity mask, if any element may be null.
    ///
    /// The returned [`PlBitmapRef`] has [`Self::len`] bits regardless of whether the backing
    /// bitmap is flat or scalar, so reading validity through it needs no knowledge of which
    /// representation this array is in.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_broadcast_unchecked(validity, self.length) })
    }

    /// Whether the views buffer holds a single view shared by every element.
    ///
    /// An array of one element is both scalar and [`flat`](Self::is_flat): the two representations
    /// coincide, and this reports them both.
    #[inline]
    pub fn views_are_scalar(&self) -> bool {
        self.views.len() == 1
    }

    /// Whether the views buffer holds one slot per element.
    ///
    /// An array of one element is both flat and [`scalar`](Self::views_are_scalar).
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
    ///
    /// The data buffers are not one of them: they are indexed by the views, so how many of them
    /// there are has nothing to do with this array's length.
    ///
    /// An array of one element is both flat and [`scalar`](Self::is_scalar).
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
    ///
    /// The inner [`Option`] is that element, so an array of nothing but nulls yields
    /// `Some(None)`. Returns `None` for an empty array, and whenever a backing buffer is flat over
    /// more than one element — its elements need not be equal, even if the other buffer is scalar.
    ///
    /// This is what lets equality and formatting avoid walking a scalar array of unbounded length.
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
    /// The view of a null element is undetermined (it can be anything that reads bytes this array
    /// holds).
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
    /// The view of a null element is undetermined (it can be anything that reads bytes this array
    /// holds).
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
    /// The value of a null element is undetermined (it can be anything).
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
    /// The value of a null element is undetermined (it can be anything).
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
    /// This is `O(1)` for a scalar validity mask and `O(len)` for a flat one, amortized over
    /// repeated calls on the same [`Bitmap`].
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
    /// This counts the bytes of every element, whether its view inlines them or points at a data
    /// buffer, and counts a repeated element once per element it stands for — so it is *not* the
    /// size of this array in memory. That makes it `O(1)` for scalar views and `O(len)` for flat
    /// ones. See [`Self::total_buffer_len`] for what the data buffers hold.
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
    ///
    /// This is what the views that do not inline their bytes point into, which is neither what the
    /// elements hold — the bytes of an element that no view points at are still counted, and the
    /// inlined ones are not — nor bounded by this array's length. See [`Self::total_bytes_len`]
    /// for the elements themselves.
    pub fn total_buffer_len(&self) -> usize {
        self.buffers.iter().map(|buffer| buffer.len()).sum()
    }

    /// Returns an iterator over the values, ignoring validity.
    ///
    /// The values of null elements are undetermined (they can be anything).
    #[inline]
    pub fn values_iter(&self) -> PlBinaryViewValuesIter<'_> {
        PlBinaryViewValuesIter::new(&self.views, &self.buffers, self.length)
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlBinaryViewIter<'_> {
        PlBinaryViewIter::new(&self.views, &self.buffers, self.validity(), self.length)
    }

    /// Returns an iterator over `length` values, repeating the single value of this array if
    /// that is all it holds, and ignoring validity.
    ///
    /// This array either has `length` elements — in which case this is [`Self::values_iter`] — or
    /// a single element, which the `length` values this yields are then all read from.
    /// Broadcasting is `O(1)`, and allocates nothing: the value is repeated as it is read, rather
    /// than materialized into an array to iterate the way [`Self::new_from_index`] would have to.
    /// The values of null elements are undetermined (they can be anything).
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

    /// Returns this array with its validity mask replaced by a flat one.
    ///
    /// # Panics
    /// Panics if `validity` does not hold one bit per element.
    /// [`Self::with_validity_broadcast`] is what installs the single bit every element shares;
    /// this function never infers that from a mask that happens to hold one bit.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Replaces the validity mask with a flat one.
    ///
    /// # Panics
    /// Panics if `validity` does not hold one bit per element.
    /// [`Self::set_validity_broadcast`] is what installs the single bit every element shares;
    /// this function never infers that from a mask that happens to hold one bit.
    pub fn set_validity(&mut self, validity: Option<Bitmap>) {
        if let Some(validity) = validity.as_ref() {
            assert!(
                is_flat_buffer_len(validity.len(), self.length),
                "validity mask of length {} is not flat for an array of length {}",
                validity.len(),
                self.length,
            );
        }
        self.validity = validity;
    }

    /// Returns this array with its validity mask replaced by one that broadcasts over it.
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor scalar for this array's length.
    #[must_use]
    pub fn with_validity_broadcast(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity_broadcast(validity);
        self
    }

    /// Replaces the validity mask with one that broadcasts over this array.
    ///
    /// This is [`Self::set_validity`] widened to the scalar representation: the mask is either
    /// flat — one bit per element — or the single bit every element shares. See
    /// [`crate::broadcast`].
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor scalar for this array's length.
    pub fn set_validity_broadcast(&mut self, validity: Option<Bitmap>) {
        if let Some(validity) = validity.as_ref() {
            assert!(
                is_valid_buffer_len(validity.len(), self.length),
                "validity mask of length {} is neither flat nor scalar for an array of length {}",
                validity.len(),
                self.length,
            );
        }
        self.validity = validity;
    }

    /// Drops the validity mask, making every element valid.
    #[must_use]
    pub fn without_validity(mut self) -> Self {
        self.validity = None;
        self
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// This function is `O(1)`. The data buffers are left as they are, so the bytes of the
    /// elements this drops are still held onto — [`Self::total_buffer_len`] does not shrink.
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
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        debug_assert!(offset + length <= self.length);

        // Scalar buffers are unaffected by slicing — every element reads the same slot — with the
        // one exception of an empty slice, which keeps no element to read it.
        if self.views_are_flat() {
            unsafe { self.views.slice_in_place_unchecked(offset..offset + length) };
        } else if length == 0 {
            unsafe { self.views.slice_in_place_unchecked(0..0) };
        }
        if let Some(validity) = self.validity.as_mut() {
            if validity.len() == self.length {
                unsafe { validity.slice_unchecked(offset, length) };
            } else if length == 0 {
                unsafe { validity.slice_unchecked(0, 0) };
            }
        }

        self.length = length;
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// This function is `O(1)`.
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
    /// This function is `O(1)`.
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
    /// This function is `O(1)`: the result is scalar, so it holds a single view no matter how long
    /// it is, and it keeps at most the one data buffer that view reads rather than the bytes
    /// themselves. A null element repeats as `length` nulls.
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
    /// This function is `O(1)`.
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

    /// Returns an equivalent array whose views and mask both hold one slot per element.
    ///
    /// This materializes any scalar buffer and is therefore `O(len)`; it is a no-op clone when
    /// this array [`is_flat`](Self::is_flat). Only the views are written out — the bytes they read
    /// stay in the data buffers they are already in, so repeating an element costs the 16 bytes of
    /// its view rather than the bytes of its value. The result carries its representation in its
    /// type: see [`Flat`] for what a flat array can do that this one cannot.
    ///
    /// # Example
    /// ```
    /// use polars_array::PlBinaryViewArray;
    ///
    /// let scalar = PlBinaryViewArray::new_scalar(b"foo", 3);
    /// assert!(scalar.flat_views().is_none());
    ///
    /// let flat = scalar.to_flat();
    /// assert_eq!(flat.views().len(), 3);
    /// assert_eq!(flat, scalar);
    /// ```
    pub fn to_flat(&self) -> Flat<Self> {
        if self.is_flat() {
            return Flat(self.clone());
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

        let validity = self.validity().map(|validity| validity.to_flat());

        Flat(Self {
            views,
            buffers: self.buffers.clone(),
            length: self.length,
            validity,
        })
    }

    /// Borrows this array as a [`Flat`] one, if its views and mask already hold one slot per
    /// element.
    ///
    /// This is the `O(1)` counterpart of [`Self::to_flat`]: it materializes nothing, and returns
    /// `None` rather than writing out a scalar buffer when this array is not
    /// [`flat`](Self::is_flat).
    ///
    /// # Example
    /// ```
    /// use polars_array::PlBinaryViewArray;
    ///
    /// let arr = PlBinaryViewArray::from_values_iter([b"foo".as_slice(), b"bar"]);
    /// assert_eq!(arr.as_flat().unwrap().views().len(), 2);
    ///
    /// // A billion copies of one value share a single view, so it has to be written out.
    /// assert!(PlBinaryViewArray::new_scalar(b"foo", 1_000_000_000).as_flat().is_none());
    /// ```
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: the views and the mask of a flat array hold one slot per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }
}

/// Validates that every view of `views` reads bytes that `buffers` holds.
///
/// This is the binary counterpart of
/// [`validate_utf8_views`](arrow::array::validate_utf8_views): the bytes themselves are anything
/// at all, so what is left to check is that they are there to be read, and that a view agrees with
/// the copy of their first four bytes it carries — two views over the same bytes have to compare
/// equal as the 16 bytes they are.
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
                validity.into_opt_validity(),
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

/// Compares two arrays element-wise; neither the representation (flat or scalar) nor how the bytes
/// of an element are reached is part of a value, so an array compares equal to any other one
/// holding the same values.
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
    fn set_validity(&mut self, validity: Option<Bitmap>) {
        self.set_validity(validity)
    }

    #[inline]
    fn set_validity_broadcast(&mut self, validity: Option<Bitmap>) {
        self.set_validity_broadcast(validity)
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
}
