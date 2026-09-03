use std::ops::Range;

use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmapRef, validity_eq};
use crate::broadcast::{
    assert_broadcastable, is_flat_buffer_len, is_flat_fixed_size_values_len, is_scalar_buffer_len,
    is_scalar_fixed_size_values_len, is_valid_buffer_len,
};
use crate::flat::Flat;

mod builder;
mod flat;
mod iterator;

pub use builder::PlFixedSizeBinaryArrayBuilder;
pub use iterator::{PlFixedSizeBinaryIter, PlFixedSizeBinaryValuesIter};

/// An immutable, cheaply cloneable sequence of `length` optional byte strings of `width` bytes each, over one values buffer.
#[derive(Clone)]
pub struct PlFixedSizeBinaryArray {
    /// Scalar: values.len() == width
    values: Buffer<u8>,
    width: usize,
    length: usize,
    /// Scalar: validity.len() == 1
    validity: Option<Bitmap>,
}

impl PlFixedSizeBinaryArray {
    /// Creates a flat [`PlFixedSizeBinaryArray`] out of its internal components.
    ///
    /// The values have to hold the bytes of every element, laid end to end, and the validity mask
    /// one bit per element. [`Self::try_new_broadcast`] is what builds the scalar representation;
    /// this function never infers it from a buffer that happens to be one element wide. This
    /// function is `O(1)`: there is only the length of `values` to check against the width.
    ///
    /// # Errors
    /// This function errors if `values` does not hold exactly `length * width` bytes, or if
    /// `validity` does not hold exactly `length` bits.
    pub fn try_new(
        values: Buffer<u8>,
        width: usize,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_flat_fixed_size_values_len(values.len(), width, length),
            ComputeError:
            "values buffer of length {} is not flat for a fixed size binary array of length {} \
             and width {}: it needs the width of every element laid end to end",
            values.len(), length, width,
        );

        if let Some(validity) = validity.as_ref() {
            polars_ensure!(
                is_flat_buffer_len(validity.len(), length),
                ComputeError:
                "validity mask of length {} is not flat for an array of length {}",
                validity.len(), length,
            );
        }

        Ok(Self {
            values,
            width,
            length,
            validity,
        })
    }

    /// Creates a flat [`PlFixedSizeBinaryArray`] out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(values: Buffer<u8>, width: usize, length: usize, validity: Option<Bitmap>) -> Self {
        Self::try_new(values, width, length, validity).unwrap()
    }

    /// Creates a flat [`PlFixedSizeBinaryArray`] out of its internal components without validating
    /// them.
    ///
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `values` must hold exactly `length * width` bytes, and `validity` exactly `length` bits.
    #[inline]
    pub unsafe fn new_unchecked(
        values: Buffer<u8>,
        width: usize,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        if cfg!(debug_assertions) {
            assert!(is_flat_fixed_size_values_len(values.len(), width, length));
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_flat_buffer_len(v.len(), length))
            );
        }

        Self {
            values,
            width,
            length,
            validity,
        }
    }

    /// Creates a scalar [`PlFixedSizeBinaryArray`] of `length` elements out of its internal
    /// components.
    ///
    /// The values have to hold the `width` bytes of the one element every element covers, and the
    /// validity mask the single bit they share, which makes this `O(1)` in `length` as well as in
    /// time. [`Self::try_new`] is what builds the flat representation.
    ///
    /// # Errors
    /// This function errors if `values` does not hold exactly `width` bytes, or if `validity` does
    /// not hold exactly one bit. An empty array has no element for the values to stand for, so it
    /// admits only empty values, and an empty mask alongside the single bit.
    pub fn try_new_broadcast(
        values: Buffer<u8>,
        width: usize,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_scalar_fixed_size_values_len(values.len(), width, length),
            ComputeError:
            "values buffer of length {} is not the one element the {} elements of a broadcast \
             fixed size binary array of width {} cover",
            values.len(), length, width,
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

        Ok(Self {
            values,
            width,
            length,
            validity,
        })
    }

    /// Creates a scalar [`PlFixedSizeBinaryArray`] of `length` elements out of its internal
    /// components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new_broadcast`] errors.
    #[inline]
    pub fn new_broadcast(
        values: Buffer<u8>,
        width: usize,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new_broadcast(values, width, length, validity).unwrap()
    }

    /// Creates a scalar [`PlFixedSizeBinaryArray`] of `length` elements out of its internal
    /// components without validating them.
    ///
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `values` must hold exactly `width` bytes — or none at all, if `length` is zero — and
    /// `validity` exactly one bit, or none at all if `length` is zero.
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        values: Buffer<u8>,
        width: usize,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        if cfg!(debug_assertions) {
            assert!(is_scalar_fixed_size_values_len(values.len(), width, length));
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_scalar_buffer_len(v.len(), length))
            );
        }

        Self {
            values,
            width,
            length,
            validity,
        }
    }

    /// Creates an empty [`PlFixedSizeBinaryArray`] of elements `width` bytes wide.
    #[inline]
    pub fn new_empty(width: usize) -> Self {
        Self {
            values: Buffer::new(),
            width,
            length: 0,
            validity: None,
        }
    }

    /// Creates a fully valid, flat [`PlFixedSizeBinaryArray`] by cutting `values` into elements of
    /// `width` bytes, taking its length from how many of them there are.
    ///
    /// The values are read as flat — `length * width` bytes — so this never builds the scalar
    /// representation: [`Self::new_scalar`] is what does. This function is `O(1)`.
    ///
    /// # Panics
    /// Panics if `width` is zero, which leaves no number of elements to cut the values into, or if
    /// the length of `values` is not a multiple of `width`.
    pub fn from_values(values: Buffer<u8>, width: usize) -> Self {
        assert!(
            width > 0,
            "the length of a fixed size binary array of width zero cannot be taken from its \
             values",
        );
        assert!(
            values.len().is_multiple_of(width),
            "the values of length {} do not divide into elements of width {}",
            values.len(),
            width,
        );

        let length = values.len() / width;
        Self {
            values,
            width,
            length,
            validity: None,
        }
    }

    /// Creates a fully valid, flat [`PlFixedSizeBinaryArray`] by cutting a [`Vec`] into elements
    /// of `width` bytes.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::from_values`] panics.
    #[inline]
    pub fn from_vec(values: Vec<u8>, width: usize) -> Self {
        Self::from_values(Buffer::from(values), width)
    }

    /// Creates a [`PlFixedSizeBinaryArray`] of `length` copies of `value`, in the memory of that
    /// one value.
    ///
    /// The length of `value` is the width of the result: every element covers all of it. This
    /// function is `O(value.len())`, and so is the result's memory use. Repeating an element of an
    /// array at hand is [`Self::new_from_index`], which copies nothing at all.
    #[inline]
    pub fn new_scalar(value: &[u8], length: usize) -> Self {
        let width = value.len();

        // There is no element for the values to be shared by when there are no elements at all,
        // which is why an empty array is the one that keeps nothing of the value it repeats.
        let values = if length == 0 {
            Buffer::new()
        } else {
            Buffer::from(value.to_vec())
        };

        Self {
            values,
            width,
            length,
            validity: None,
        }
    }

    /// Creates a [`PlFixedSizeBinaryArray`] of `length` nulls whose elements are `width` bytes
    /// wide.
    ///
    /// Every element is null, so its bytes are undetermined; a zeroed buffer stands in for the one
    /// element they all cover, which is what keeps both the validity mask and the values a single
    /// shared slot. This function is `O(1)`.
    #[inline]
    pub fn new_full_null(width: usize, length: usize) -> Self {
        Self {
            values: if length == 0 {
                Buffer::new()
            } else {
                Buffer::zeroed(width)
            },
            width,
            length,
            validity: Some(Bitmap::new_zeroed(1)),
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

    /// The number of bytes in every element.
    ///
    /// This is as much a part of the array as its length: a null element is as wide as any other,
    /// since it is the mask and not the width that makes it null.
    #[inline(always)]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// The backing values buffer, if it holds the bytes of every element, laid end to end.
    ///
    /// Element `i` is then the [`width`](Self::width) bytes at `i * width`, with no
    /// [`broadcast_index`](crate::broadcast::broadcast_index) in the way. This is the `O(1)`
    /// counterpart of [`Self::to_flat`]: it materializes nothing, and returns `None` rather than
    /// repeating a scalar buffer. Reach for the bytes a scalar buffer shares with
    /// [`Self::scalar_values`] instead — between them the two cover every array that has elements
    /// at all, so a `None` from both is an empty array. The values of null elements are
    /// undetermined (they can be any byte string of the width).
    #[inline]
    pub fn flat_values(&self) -> Option<&Buffer<u8>> {
        self.values_are_flat().then_some(&self.values)
    }

    /// The bytes every element of this array reads, if the values hold a single element.
    ///
    /// This is the values half of [`Self::scalar_value`], which additionally asks that the
    /// validity mask be scalar and reports the null the mask makes of these bytes. Returns `None`
    /// for values that are flat over more than one element, and for an empty array, which has no
    /// element to share a value. The value of a null element is undetermined (it can be any byte
    /// string of the width).
    #[inline]
    pub fn scalar_values(&self) -> Option<&[u8]> {
        self.values_are_scalar().then(|| self.values.as_slice())
    }

    /// Consumes this array into its internal components.
    ///
    /// The values are *not* guaranteed to hold [`Self::len`] `*` [`Self::width`] bytes: they are
    /// either flat or scalar, which is why the width and the length come with them. See
    /// [`crate::broadcast`] for how to read them.
    #[inline]
    pub fn into_inner(self) -> (Buffer<u8>, usize, usize, Option<Bitmap>) {
        (self.values, self.width, self.length, self.validity)
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

    /// Whether the values hold the single element every element of this array covers, so that
    /// every element is the same value.
    ///
    /// An array of one element is both scalar and [`flat`](Self::values_are_flat): the two
    /// representations coincide, and this reports them both. So is an array of width zero, whose
    /// elements hold no bytes to lay out either way.
    #[inline]
    pub fn values_are_scalar(&self) -> bool {
        self.values.len() == self.width && self.length >= 1
    }

    /// Whether the values hold the bytes of every element, laid end to end.
    ///
    /// An array of one element is both flat and [`scalar`](Self::values_are_scalar).
    #[inline]
    pub fn values_are_flat(&self) -> bool {
        // A length times a width that overflows a `usize` is longer than any buffer can be, so
        // such an array is never flat.
        self.length.checked_mul(self.width) == Some(self.values.len())
    }

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether this array's values hold the bytes of every element and its mask one bit per
    /// element.
    ///
    /// An array of one element is both flat and [`scalar`](Self::is_scalar).
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.values_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array is entirely stored in the scalar representation, and therefore stands
    /// for a single value repeated [`Self::len`] times in the memory of that value alone.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.values_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element of this array equals, if both backing buffers hold one
    /// slot.
    ///
    /// The inner [`Option`] is that element, so an array of nothing but nulls yields `Some(None)`.
    /// Returns `None` for an empty array, and whenever a buffer is flat over more than one element
    /// — its elements need not be equal, even if the other buffer is scalar.
    ///
    /// This is what lets equality and formatting avoid walking a scalar array of unbounded length.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<&[u8]>> {
        let is_shared = self.values.len() == self.width
            && self
                .validity
                .as_ref()
                .is_none_or(|validity| validity.len() == 1);

        // SAFETY: the array is not empty, so element 0 is in bounds.
        (is_shared && self.length > 0).then(|| unsafe { self.get_unchecked(0) })
    }

    /// The range of the backing values buffer the element at `i` covers, which is always
    /// [`Self::width`] bytes wide.
    ///
    /// This is an index into the buffer [`Self::flat_values`] hands out, or into the single
    /// element [`Self::scalar_values`] does — for scalar values every element covers the same
    /// range, which is all of them.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value_range(&self, i: usize) -> Range<usize> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.value_range_unchecked(i) }
    }

    /// The range of the backing values buffer the element at `i` covers, which is always
    /// [`Self::width`] bytes wide.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_range_unchecked(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.length);

        // Scalar values hold the one element every element covers, so they are read from the
        // start; flat ones lay the elements end to end, one width apart.
        let start = if self.values_are_scalar() {
            0
        } else {
            i * self.width
        };
        start..start + self.width
    }

    /// Returns the bytes of the element at `i`.
    ///
    /// This function is `O(1)`. The value of a null element is undetermined (it can be any byte
    /// string of the width).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value(&self, i: usize) -> &[u8] {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the bytes of the element at `i`.
    ///
    /// This function is `O(1)`. The value of a null element is undetermined (it can be any byte
    /// string of the width).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        let range = unsafe { self.value_range_unchecked(i) };
        // SAFETY: the values hold the width of every element, or the one they all cover.
        unsafe { self.values.get_unchecked(range) }
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

    /// Returns the bytes of the element at `i`, or `None` if it is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the bytes of the element at `i`, or `None` if it is null.
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

    /// Returns an iterator over the elements, ignoring validity.
    ///
    /// The values of null elements are undetermined (they can be any byte string of the width).
    #[inline]
    pub fn values_iter(&self) -> PlFixedSizeBinaryValuesIter<'_> {
        PlFixedSizeBinaryValuesIter::new(self.values.as_slice(), self.width, self.length)
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlFixedSizeBinaryIter<'_> {
        PlFixedSizeBinaryIter::new(
            self.values.as_slice(),
            self.width,
            self.validity(),
            self.length,
        )
    }

    /// Returns an iterator over `length` elements, repeating the single element of this array if
    /// that is all it holds, and ignoring validity.
    ///
    /// This array either has `length` elements — in which case this is [`Self::values_iter`] — or
    /// a single element, which the `length` values this yields are then all read from.
    /// Broadcasting is `O(1)`, and allocates nothing: the value is repeated as it is read, rather
    /// than materialized into an array to iterate the way [`Self::new_from_index`] would have to.
    /// The values of null elements are undetermined (they can be any byte string of the width).
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlFixedSizeBinaryValuesIter<'_> {
        assert_broadcastable(self.length, length);
        // SAFETY: an array of one element holds the width of that one element, which is scalar
        // for any length; otherwise `length` is the length the values are already valid for.
        PlFixedSizeBinaryValuesIter::new(self.values.as_slice(), self.width, length)
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
    /// This function is `O(1)`.
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

        if self.values_are_flat() {
            // There is nothing to leave the bytes outside the slice behind, so they are sliced
            // along with it, a width at a time.
            unsafe {
                self.values
                    .slice_in_place_unchecked(offset * self.width..(offset + length) * self.width)
            };
        } else if length == 0 {
            // Scalar values are unaffected by slicing — every element covers the same bytes — with
            // the one exception of an empty slice, which keeps no element to share them.
            unsafe { self.values.slice_in_place_unchecked(0..0) };
        }

        // A scalar mask is unaffected by slicing: every element reads the same bit.
        if let Some(validity) = self.validity.as_mut() {
            if validity.len() == self.length {
                unsafe { validity.slice_unchecked(offset, length) };
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

    /// Creates a [`PlFixedSizeBinaryArray`] of `length` copies of the element at `index`.
    ///
    /// This function is `O(1)`: the values of the result are the element being repeated, sliced
    /// out of the very same buffer, and every one of its elements covers all of them. A null
    /// element repeats as `length` nulls.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        assert!(index < self.length, "index out of bounds");
        unsafe { self.new_from_index_unchecked(index, length) }
    }

    /// Creates a [`PlFixedSizeBinaryArray`] of `length` copies of the element at `index`.
    ///
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        // The bytes of a null element are undetermined, so they are not carried over: it is the
        // mask that makes every element of the result null, over a zeroed element of the width.
        if unsafe { self.is_null_unchecked(index) } {
            return Self::new_full_null(self.width, length);
        }

        // The element is sliced out of the values it is already in, which every element of the
        // result covers: nothing is copied.
        let range = unsafe { self.value_range_unchecked(index) };
        let values = if length == 0 {
            Buffer::new()
        } else {
            self.values.clone().sliced(range)
        };

        Self {
            values,
            width: self.width,
            length,
            validity: None,
        }
    }

    /// Returns an equivalent array whose values hold the bytes of every element and whose mask
    /// holds one bit per element.
    ///
    /// Materializing scalar values is what costs here: the one element every element covers has to
    /// be written out once per element, which is `O(len * width)`. Nothing is written out when
    /// every element is null, since the bytes of a null element are undetermined and a zeroed
    /// buffer stands in for them. The result carries its representation in its type: see [`Flat`]
    /// for what a flat array is a proof of.
    ///
    /// # Example
    /// ```
    /// use polars_array::PlFixedSizeBinaryArray;
    ///
    /// // Three copies of `ab`, over the bytes of that one value.
    /// let scalar = PlFixedSizeBinaryArray::new_scalar(b"ab", 3);
    /// assert!(scalar.flat_values().is_none());
    ///
    /// // Its flat counterpart holds the three elements one after the other.
    /// let flat = scalar.to_flat();
    /// assert_eq!(flat.as_slice(), b"ababab");
    /// assert_eq!(flat, scalar);
    /// ```
    pub fn to_flat(&self) -> Flat<Self> {
        if self.is_flat() {
            return Flat(self.clone());
        }

        let validity = self.validity().map(|validity| validity.to_flat());

        let values = if self.values_are_flat() {
            self.values.clone()
        } else if self.null_count() == self.length {
            // Every element is null, so every value is undetermined: a zeroed buffer of the right
            // length stands in for them, which is not written out one element at a time.
            Buffer::zeroed(self.flat_values_len())
        } else {
            // The one element every element covers, written out once per element.
            let flat_len = self.flat_values_len();
            let element = self.values.as_slice();

            let mut values = Vec::with_capacity(flat_len);
            for _ in 0..self.length {
                values.extend_from_slice(element);
            }
            Buffer::from(values)
        };

        // SAFETY: the values are the element every element covers, repeated once per element, and
        // the mask is the flat counterpart of one valid for this array's length.
        Flat(unsafe { Self::new_unchecked(values, self.width, self.length, validity) })
    }

    /// Borrows this array as a [`Flat`] one, if its values already hold the bytes of every element
    /// and its mask one bit per element.
    ///
    /// This is the `O(1)` counterpart of [`Self::to_flat`]: it materializes nothing, and returns
    /// `None` rather than writing out a scalar buffer when this array is not
    /// [`flat`](Self::is_flat).
    ///
    /// # Example
    /// ```
    /// use polars_array::PlFixedSizeBinaryArray;
    ///
    /// let arr = PlFixedSizeBinaryArray::from_vec(vec![1u8, 2, 3, 4], 2);
    /// assert_eq!(arr.as_flat().unwrap().as_slice(), [1, 2, 3, 4]);
    ///
    /// // A billion copies of one value share its bytes, so they have to be written out.
    /// let scalar = PlFixedSizeBinaryArray::new_scalar(b"ab", 1_000_000_000);
    /// assert!(scalar.as_flat().is_none());
    /// ```
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: the values of a flat array hold the width of every element, and its mask one bit
        // per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }

    /// The number of bytes a flat counterpart of this array holds.
    ///
    /// # Panics
    /// Panics if that overflows a `usize`, which no buffer has the memory to back.
    #[inline]
    fn flat_values_len(&self) -> usize {
        self.length.checked_mul(self.width).expect(
            "the values of the flat counterpart of the fixed size binary array overflow a `usize`",
        )
    }
}

impl<'a> IntoIterator for &'a PlFixedSizeBinaryArray {
    type Item = Option<&'a [u8]>;
    type IntoIter = PlFixedSizeBinaryIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares two arrays element-wise; neither the representation (flat or scalar) nor the values of
/// null elements are part of a value, so an array compares equal to any other one holding the same
/// byte strings. The width is part of one: byte strings of different widths are different values,
/// and an array holds nothing else.
impl PartialEq for PlFixedSizeBinaryArray {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length || self.width != other.width {
            return false;
        }

        if !validity_eq(self.validity(), other.validity(), self.length) {
            return false;
        }

        // Every element is null on both sides, so every value is undetermined and there is nothing
        // left to compare. This is also what keeps comparing two fully null scalar arrays `O(1)`.
        if self.length > 0 && self.null_count() == self.length {
            return true;
        }

        // Never walk two scalar arrays element by element: their length is unbounded by their
        // memory use. Comparing the one element they each stand for costs that element.
        if let (Some(lhs), Some(rhs)) = (self.scalar_value(), other.scalar_value()) {
            return lhs == rhs;
        }

        self.iter().eq(other.iter())
    }
}

impl Eq for PlFixedSizeBinaryArray {}

/// Compares an array of unknown representation against a flat one; see
/// [`PartialEq<PlFixedSizeBinaryArray> for Flat<PlFixedSizeBinaryArray>`](Flat).
impl PartialEq<Flat<PlFixedSizeBinaryArray>> for PlFixedSizeBinaryArray {
    #[inline]
    fn eq(&self, other: &Flat<PlFixedSizeBinaryArray>) -> bool {
        *self == other.0
    }
}

impl std::fmt::Debug for PlFixedSizeBinaryArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The buffers are listed as they are backed, which is one element's worth for a scalar
        // array: this never materializes a length that is unbounded by the memory use.
        let mut s = f.debug_struct("PlFixedSizeBinaryArray");
        s.field("length", &self.length);
        s.field("width", &self.width);
        if let Some(validity) = self.validity() {
            s.field("validity", &validity);
        }
        s.field("values", &self.values).finish()
    }
}

impl PlArray for PlFixedSizeBinaryArray {
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
        PlArrayType::FixedSizeBinary
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

    /// The bytes every test cuts into elements.
    fn values() -> Buffer<u8> {
        Buffer::from(vec![1u8, 2, 3, 4, 5, 6])
    }

    /// The three elements `[1, 2]`, `[3, 4]` and `[5, 6]` over [`values`].
    fn arr() -> PlFixedSizeBinaryArray {
        PlFixedSizeBinaryArray::from_values(values(), 2)
    }

    #[test]
    fn flat() {
        let arr = arr();

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.width(), 2);
        assert!(!arr.is_empty());
        assert_eq!(arr.null_count(), 0);
        assert!(!arr.has_nulls());
        assert!(arr.is_valid(1));
        assert!(!arr.is_null(1));
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert!(arr.values_are_flat());
        assert!(!arr.values_are_scalar());
        assert_eq!(arr.flat_values().unwrap().len(), 6);

        assert_eq!(arr.value_range(0), 0..2);
        assert_eq!(arr.value_range(1), 2..4);
        assert_eq!(arr.value_range(2), 4..6);

        assert_eq!(arr.value(0), [1, 2]);
        assert_eq!(arr.value(2), [5, 6]);
        assert_eq!(arr.get(1), Some([3, 4].as_slice()));
        assert_eq!(arr.iter().collect::<Vec<_>>().len(), 3);
    }

    #[test]
    fn null_scalar() {
        // Every element is null, over the two bytes they all share: the mask is a single bit, and
        // the values are one element.
        let arr = PlFixedSizeBinaryArray::new_full_null(2, 1_000_000);

        assert!(arr.is_scalar());
        assert!(arr.validity_is_scalar());
        assert!(arr.values_are_scalar());
        assert!(!arr.values_are_flat());
        assert_eq!(arr.width(), 2);
        assert_eq!(arr.validity().unwrap().len(), 1_000_000);
        assert_eq!(arr.scalar_values().unwrap().len(), 2);
        assert_eq!(arr.null_count(), 1_000_000);
        assert!(arr.has_nulls());
        assert!(arr.is_null(999_999));
        assert_eq!(arr.get(999_999), None);
        assert_eq!(arr.value_range(999_999), 0..2);
        assert_eq!(arr.scalar_value(), Some(None));

        let valid = arr.without_validity();
        assert_eq!(valid.null_count(), 0);
        assert!(valid.validity().is_none());
    }

    #[test]
    fn slicing_slices_the_values() {
        let arr = PlFixedSizeBinaryArray::new(
            values(),
            2,
            3,
            Some(Bitmap::from_iter([true, false, true])),
        )
        .sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.width(), 2);
        assert_eq!(arr.null_count(), 1);
        assert!(arr.is_flat());

        // There is nothing to leave the bytes outside the slice behind.
        assert_eq!(arr.flat_values().unwrap().as_slice(), [3, 4, 5, 6]);
        assert_eq!(arr.value(1), [5, 6]);

        // Slicing away every element leaves no bytes behind either.
        let arr = arr.sliced(2, 0);
        assert!(arr.is_empty());
        assert_eq!(arr.flat_values().unwrap().len(), 0);
        assert_eq!(arr.width(), 2);
    }
}
