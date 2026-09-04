use std::borrow::Cow;
use std::ops::Range;

use arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmapRef, validity_eq};
use crate::broadcast::{
    ArrayRepr, assert_broadcastable, broadcast_index, is_flat_buffer_len, is_flat_offsets_len,
    is_scalar_buffer_len, is_scalar_offsets_len, is_valid_buffer_len, normalize_offsets,
    normalize_validity, scalar_buffer_len, scalar_offsets_len, slice_offsets, slice_validity,
};
use crate::flat::Flat;

mod builder;
mod flat;
mod iterator;

pub use builder::PlBinaryArrayBuilder;
pub use iterator::{PlBinaryIter, PlBinaryValues, PlBinaryValuesIter};

/// An immutable, cheaply cloneable sequence of `length` optional byte strings over one values
/// buffer.
#[derive(Clone)]
pub struct PlBinaryArray {
    values: Buffer<u8>,
    /// Scalar: offsets.len() == 2
    offsets: Buffer<u64>,
    length: usize,
    /// Scalar: validity.len() == 1
    validity: Option<Bitmap>,
}

impl PlBinaryArray {
    /// Creates a flat [`PlBinaryArray`] out of its internal components.
    ///
    /// # Errors
    /// This function errors unless `offsets` holds `length + 1` non-decreasing offsets ending
    /// within `values`, and `validity` holds `length` bits.
    pub fn try_new(
        values: Buffer<u8>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_flat_offsets_len(offsets.len(), length),
            ComputeError:
            "offsets buffer of length {} is not flat for a binary array of length {}: it needs one \
             offset per element plus the end of the last",
            offsets.len(), length,
        );

        validate_offsets(&values, &offsets)?;

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
            offsets,
            length,
            validity,
        })
    }

    /// Creates a flat [`PlBinaryArray`] out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(
        values: Buffer<u8>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new(values, offsets, length, validity).unwrap()
    }

    /// Creates a flat [`PlBinaryArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// `offsets` must be non-decreasing, hold `length + 1` offsets and end within `values`;
    /// `validity` must hold `length` bits.
    #[inline]
    pub unsafe fn new_unchecked(
        values: Buffer<u8>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        if cfg!(debug_assertions) {
            assert!(is_flat_offsets_len(offsets.len(), length));
            assert!(offsets.windows(2).all(|window| window[0] <= window[1]));
            assert!(offsets[offsets.len() - 1] <= values.len() as u64);
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_flat_buffer_len(v.len(), length))
            );
        }

        Self {
            values,
            offsets,
            length,
            validity,
        }
    }

    /// Creates a scalar [`PlBinaryArray`] of `length` elements out of its internal components.
    ///
    /// # Errors
    /// This function errors unless `offsets` is scalar for `length`, per [`is_scalar_offsets_len`],
    /// non-decreasing and ending within `values`, and `validity` is scalar for `length`, per
    /// [`is_scalar_buffer_len`].
    pub fn try_new_broadcast(
        values: Buffer<u8>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_scalar_offsets_len(offsets.len(), length),
            ComputeError:
            "offsets buffer of length {} is not the single range the {} elements of a broadcast \
             binary array share: it needs the two offsets standing for that range",
            offsets.len(), length,
        );

        validate_offsets(&values, &offsets)?;

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
            offsets: normalize_offsets(offsets, length),
            length,
            validity: normalize_validity(validity, length),
        })
    }

    /// Creates a scalar [`PlBinaryArray`] of `length` elements out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new_broadcast`] errors.
    #[inline]
    pub fn new_broadcast(
        values: Buffer<u8>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new_broadcast(values, offsets, length, validity).unwrap()
    }

    /// Creates a scalar [`PlBinaryArray`] of `length` elements out of its internal components
    /// without validating them.
    ///
    /// # Safety
    /// `offsets` must be non-decreasing, scalar for `length` per [`is_scalar_offsets_len`], and end
    /// within `values`; `validity` must be scalar for `length`, per [`is_scalar_buffer_len`].
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        values: Buffer<u8>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        if cfg!(debug_assertions) {
            assert!(is_scalar_offsets_len(offsets.len(), length));
            assert!(offsets.windows(2).all(|window| window[0] <= window[1]));
            assert!(offsets[offsets.len() - 1] <= values.len() as u64);
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_scalar_buffer_len(v.len(), length))
            );
        }

        Self {
            values,
            offsets: normalize_offsets(offsets, length),
            length,
            validity: normalize_validity(validity, length),
        }
    }

    /// Creates an empty [`PlBinaryArray`].
    #[inline]
    pub fn new_empty() -> Self {
        Self {
            values: Buffer::new(),
            // The end of the last element of an empty array, which is always needed.
            offsets: Buffer::zeroed(1),
            length: 0,
            validity: None,
        }
    }

    /// Creates a fully valid, flat [`PlBinaryArray`] from `values` and `offsets`, taking its length
    /// from the offsets.
    ///
    /// # Panics
    /// Panics if `offsets` is empty — the end of the last element is always needed, so even an
    /// empty array has one offset — or under the conditions [`Self::try_new`] errors.
    pub fn from_offsets(values: Buffer<u8>, offsets: Buffer<u64>) -> Self {
        let length = offsets
            .len()
            .checked_sub(1)
            .expect("a binary array needs at least one offset");
        Self::new(values, offsets, length, None)
    }

    /// Creates a fully valid, flat [`PlBinaryArray`] holding `values`, in order.
    pub fn from_values_iter<V: AsRef<[u8]>, I: IntoIterator<Item = V>>(values: I) -> Self {
        let values = values.into_iter();
        let (lower, _) = values.size_hint();

        let mut bytes = Vec::new();
        let mut offsets = Vec::with_capacity(lower + 1);
        offsets.push(0);

        for value in values {
            bytes.extend_from_slice(value.as_ref());
            offsets.push(bytes.len() as u64);
        }

        let length = offsets.len() - 1;
        // SAFETY: the offsets are the ends of the values appended so far: ordered, one per element
        // plus the end of the last, ending at the length of the bytes they were built over.
        unsafe { Self::new_unchecked(Buffer::from(bytes), Buffer::from(offsets), length, None) }
    }

    /// Creates a [`PlBinaryArray`] of `length` copies of `value`, in the memory of that one value.
    #[inline]
    pub fn new_scalar(value: &[u8], length: usize) -> Self {
        // There is no element for the bytes to be shared by when there are no elements at all,
        // which is why an empty array is the one that keeps nothing of the value it repeats: no
        // bytes, and no range over them either.
        if length == 0 {
            return Self::new_empty();
        }

        Self {
            values: Buffer::from(value.to_vec()),
            offsets: Buffer::from_owner([0, value.len() as u64]),
            length,
            validity: None,
        }
    }

    /// Creates a [`PlBinaryArray`] of `length` nulls.
    #[inline]
    pub fn new_full_null(length: usize) -> Self {
        Self {
            values: Buffer::new(),
            offsets: Buffer::zeroed(scalar_offsets_len(length)),
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

    /// The backing values buffer, holding the bytes the offsets cut the elements out of.
    #[inline(always)]
    pub const fn values(&self) -> &Buffer<u8> {
        &self.values
    }

    /// Which representation the backing offsets buffer is in, along with what it holds.
    ///
    /// The two arms are read differently: [`Flat`](ArrayRepr::Flat) is the raw offsets, which hold the
    /// start of every element plus the end of the last and are resolved per element with
    /// [`Self::value_range_unchecked`], while [`Scalar`](ArrayRepr::Scalar) is the one range every
    /// element covers, already resolved out of the two offsets a scalar buffer holds.
    #[inline]
    pub fn offsets_repr(&self) -> ArrayRepr<&Buffer<u64>, Range<u64>> {
        if self.offsets_are_scalar() {
            // SAFETY: a scalar offsets buffer holds two slots, so both are in bounds.
            let (start, end) = unsafe {
                (
                    *self.offsets.get_unchecked(0),
                    *self.offsets.get_unchecked(1),
                )
            };
            ArrayRepr::Scalar(start..end)
        } else {
            ArrayRepr::Flat(&self.offsets)
        }
    }

    /// The backing offsets buffer, if it holds the range of every element, laid end to end.
    #[inline]
    pub fn flat_offsets(&self) -> Option<&Buffer<u64>> {
        self.offsets_repr().flat()
    }

    /// The range of [`Self::values`] every element of this array covers, if the offsets hold a
    /// single range.
    #[inline]
    pub fn scalar_offsets(&self) -> Option<Range<usize>> {
        // Every offset of an array that upholds its invariants fits in a `usize`.
        self.offsets_repr()
            .scalar()
            .map(|range| range.start as usize..range.end as usize)
    }

    /// The bytes every element of this array reads, if the offsets hold a single range.
    #[inline]
    pub fn scalar_values(&self) -> Option<&[u8]> {
        // SAFETY: the range comes from the offsets, so it is in bounds of the values.
        self.scalar_offsets()
            .map(|range| unsafe { self.values.get_unchecked(range) })
    }

    /// Consumes this array into its internal components.
    #[inline]
    pub fn into_inner(self) -> (Buffer<u8>, Buffer<u64>, usize, Option<Bitmap>) {
        (self.values, self.offsets, self.length, self.validity)
    }

    /// The validity mask, if any element may be null.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_broadcast_unchecked(validity, self.length) })
    }

    /// Whether the offsets hold the single range every element covers, so that every element is the
    /// same byte string.
    #[inline]
    pub fn offsets_are_scalar(&self) -> bool {
        // The offsets hold one slot more than the starts that are flat or scalar for this array's
        // length, so the two of a scalar array are a single start and the end of it. An array of
        // no elements holds the one offset it starts at and no range at all, and is flat.
        self.offsets.len() == 2 && self.length > 0
    }

    /// Whether the offsets hold the range of every element, laid end to end.
    #[inline]
    pub fn offsets_are_flat(&self) -> bool {
        // The offsets are never empty, and hold the start of every element plus the end of the
        // last. This is spelled as the predicate the iterators resolve their own representation
        // with, rather than as the subtraction it comes down to for an array that upholds its
        // invariants: a caller that asserts this ahead of a walk is then asserting the very
        // condition the walk branches on, which folds the branch — and the tag it reads, and the
        // step it computes — out of the loop.
        is_flat_offsets_len(self.offsets.len(), self.length)
    }

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether this array's offsets hold the range of every element and its mask one bit per
    /// element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.offsets_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array is entirely stored in the scalar representation, and therefore stands for
    /// a single value repeated [`Self::len`] times in the memory of that value alone.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.offsets_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element of this array equals, if both of its own backing buffers
    /// hold one slot.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<&[u8]>> {
        let is_shared = self.offsets.len() == 2
            && self
                .validity
                .as_ref()
                .is_none_or(|validity| validity.len() == 1);

        // SAFETY: the array is not empty, so element 0 is in bounds.
        (is_shared && self.length > 0).then(|| unsafe { self.get_unchecked(0) })
    }

    /// The range of [`Self::values`] the element at `i` covers.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value_range(&self, i: usize) -> Range<usize> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.value_range_unchecked(i) }
    }

    /// The range of [`Self::values`] the element at `i` covers.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_range_unchecked(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.length);

        // Scalar offsets hold the one range every element covers, so they are read at slot zero.
        let i = broadcast_index(i, self.offsets.len() - 1);

        // SAFETY: the offsets hold one slot more than the starts `broadcast_index` maps onto, so
        // `i + 1` is in bounds, and every offset fits in a `usize`.
        unsafe {
            let start = *self.offsets.get_unchecked(i) as usize;
            let end = *self.offsets.get_unchecked(i + 1) as usize;
            start..end
        }
    }

    /// The number of bytes in the element at `i`.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value_length(&self, i: usize) -> usize {
        self.value_range(i).len()
    }

    /// The number of bytes in the element at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_length_unchecked(&self, i: usize) -> usize {
        unsafe { self.value_range_unchecked(i) }.len()
    }

    /// Returns the bytes of the element at `i`.
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
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        let range = unsafe { self.value_range_unchecked(i) };
        // SAFETY: the offsets are ordered and bounded by the length of the values buffer.
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

    /// Returns an iterator over the elements, ignoring validity.
    #[inline]
    pub fn values_iter(&self) -> PlBinaryValuesIter<'_> {
        PlBinaryValuesIter::new(&self.values, &self.offsets, self.length)
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlBinaryIter<'_> {
        PlBinaryIter::new(&self.values, &self.offsets, self.validity(), self.length)
    }

    /// Returns an iterator over `length` elements, repeating the single element of this array if
    /// that is all it holds, and ignoring validity.
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlBinaryValuesIter<'_> {
        assert_broadcastable(self.length, length);
        // SAFETY: an array of one element holds the one range that element covers, which is scalar
        // for any length; otherwise `length` is the length the offsets are already valid for.
        PlBinaryValuesIter::new(&self.values, &self.offsets, length)
    }

    /// Returns this array with its validity mask replaced by a flat one.
    ///
    /// # Panics
    /// Panics if `validity` does not hold one bit per element.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Replaces the validity mask with a flat one.
    ///
    /// # Panics
    /// Panics if `validity` does not hold one bit per element.
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
        self.validity = normalize_validity(validity, self.length);
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

        // The bytes the offsets point into are left as they are; see `slice_offsets`.
        unsafe {
            slice_offsets(&mut self.offsets, self.length, offset, length);
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

    /// Creates a [`PlBinaryArray`] of `length` copies of the element at `index`.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        assert!(index < self.length, "index out of bounds");
        unsafe { self.new_from_index_unchecked(index, length) }
    }

    /// Creates a [`PlBinaryArray`] of `length` copies of the element at `index`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        // The bytes of a null element are undetermined, so they are not carried over: it is the
        // mask that makes every element of the result null, over no bytes at all.
        if unsafe { self.is_null_unchecked(index) } {
            return Self::new_full_null(length);
        }

        if length == 0 {
            return Self::new_empty();
        }

        // Nothing is copied: the values are cloned as they are, and the two offsets every element
        // of the result shares are the ones of the element being repeated.
        let range = unsafe { self.value_range_unchecked(index) };

        Self {
            values: self.values.clone(),
            offsets: Buffer::from_owner([range.start as u64, range.end as u64]),
            length,
            validity: None,
        }
    }

    /// Returns an equivalent array whose offsets hold the range of every element and whose mask
    /// holds one bit per element, borrowing this array itself if they already do.
    pub fn to_flat(&self) -> Cow<'_, Flat<Self>> {
        if let Some(flat) = self.as_flat() {
            return Cow::Borrowed(flat);
        }

        let validity = self
            .validity()
            .map(|validity| validity.to_flat().into_owned());

        let (values, offsets) = if self.offsets_are_flat() {
            (self.values.clone(), self.offsets.clone())
        } else if self.length == 0
            || self.offsets[0] == self.offsets[1]
            || self.null_count() == self.length
        {
            // Every element is the empty byte string, or is null and therefore holds an
            // undetermined one: no value is written out, and the offsets all point at the same
            // place. That place is the start of the values rather than the range every element
            // covered, which is the same empty byte string.
            (self.values.clone(), Buffer::zeroed(self.length + 1))
        } else {
            // The one value every element covers, written out once per element.
            let range = unsafe { self.value_range_unchecked(0) };
            // SAFETY: the range comes from the offsets, so it is in bounds of the values.
            let element = unsafe { self.values.get_unchecked(range.clone()) };

            let flat_len = self.length.checked_mul(range.len()).expect(
                "the values of the flat counterpart of the binary array overflow a `usize`",
            );
            let mut values = Vec::with_capacity(flat_len);
            for _ in 0..self.length {
                values.extend_from_slice(element);
            }

            let offsets = (0..=self.length as u64)
                .map(|i| i * range.len() as u64)
                .collect::<Vec<_>>();

            (Buffer::from(values), Buffer::from(offsets))
        };

        // SAFETY: the offsets are ordered, one per element plus the end of the last, and within the
        // values; the mask is the flat counterpart of one valid for this array's length.
        Cow::Owned(Flat(unsafe {
            Self::new_unchecked(values, offsets, self.length, validity)
        }))
    }

    /// Borrows this array as a [`Flat`] one, if its offsets already hold the range of every element
    /// and its mask one bit per element.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: the offsets of a flat array hold the range of every element, and its mask one
        // bit per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }
}

impl Default for PlBinaryArray {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

impl<V: AsRef<[u8]>> FromIterator<Option<V>> for PlBinaryArray {
    fn from_iter<I: IntoIterator<Item = Option<V>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut bytes = Vec::new();
        let mut offsets = Vec::with_capacity(lower + 1);
        offsets.push(0);
        let mut validity = BitmapBuilder::with_capacity(lower);

        for value in iter {
            // The value of a null element is undetermined, so nothing is written out for it: it
            // covers the empty byte string that ends the element before it.
            if let Some(value) = value.as_ref() {
                bytes.extend_from_slice(value.as_ref());
            }
            offsets.push(bytes.len() as u64);
            validity.push(value.is_some());
        }

        let length = offsets.len() - 1;
        // SAFETY: the offsets are the ends of the values appended so far, ending at the length of
        // the bytes they were built over, and the mask holds one bit per element.
        unsafe {
            Self::new_unchecked(
                Buffer::from(bytes),
                Buffer::from(offsets),
                length,
                validity.into_opt_validity(),
            )
        }
    }
}

impl<'a> IntoIterator for &'a PlBinaryArray {
    type Item = Option<&'a [u8]>;
    type IntoIter = PlBinaryIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares two arrays element-wise, disregarding the offsets and the bytes of null elements.
impl PartialEq for PlBinaryArray {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
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

impl Eq for PlBinaryArray {}

/// Compares an array of unknown representation against a flat one; see [`PartialEq<PlBinaryArray>
/// for Flat<PlBinaryArray>`](Flat).
impl PartialEq<Flat<PlBinaryArray>> for PlBinaryArray {
    #[inline]
    fn eq(&self, other: &Flat<PlBinaryArray>) -> bool {
        *self == other.0
    }
}

impl std::fmt::Debug for PlBinaryArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The buffers are listed as they are backed, which is two offsets and one value's worth of
        // bytes for a scalar array: this never materializes a length that is unbounded by the
        // memory use.
        let mut s = f.debug_struct("PlBinaryArray");
        s.field("length", &self.length);
        if let Some(validity) = self.validity() {
            s.field("validity", &validity);
        }
        s.field("offsets", &self.offsets.as_slice());
        s.field("values", &self.values.as_slice()).finish()
    }
}

impl PlArray for PlBinaryArray {
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
        PlArrayType::Binary
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

/// Checks that `offsets` are monotonically non-decreasing and stay within `values`.
fn validate_offsets(values: &Buffer<u8>, offsets: &Buffer<u64>) -> PolarsResult<()> {
    // The offsets are ordered, so checking the last one against the values covers them all —
    // including that every one of them fits in a `usize`.
    for (i, window) in offsets.windows(2).enumerate() {
        polars_ensure!(
            window[0] <= window[1],
            ComputeError:
            "offset {} of the binary array is {}, which is smaller than the offset {} before it",
            i + 1, window[1], window[0],
        );
    }

    let last = offsets[offsets.len() - 1];
    polars_ensure!(
        last <= values.len() as u64,
        ComputeError:
        "the last offset of the binary array is {}, which exceeds the length {} of its values",
        last, values.len(),
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three byte strings `foo`, `` and `bar`, over the six bytes of the two that hold any.
    fn arr() -> PlBinaryArray {
        PlBinaryArray::from_offsets(
            Buffer::from(b"foobar".to_vec()),
            Buffer::from(vec![0u64, 3, 3, 6]),
        )
    }

    #[test]
    fn flat() {
        let arr = arr();

        assert_eq!(arr.len(), 3);
        assert!(!arr.is_empty());
        assert_eq!(arr.null_count(), 0);
        assert!(!arr.has_nulls());
        assert!(arr.is_valid(1));
        assert!(!arr.is_null(1));
        assert_eq!(arr.flat_offsets().unwrap().as_slice(), [0, 3, 3, 6]);
        assert_eq!(arr.values().as_slice(), b"foobar");

        assert_eq!(arr.value_range(0), 0..3);
        assert_eq!(arr.value_range(1), 3..3);
        assert_eq!(arr.value_range(2), 3..6);
        assert_eq!(arr.value_length(0), 3);
        assert_eq!(arr.value_length(1), 0);
        assert_eq!(unsafe { arr.value_length_unchecked(2) }, 3);

        assert_eq!(arr.value(0), b"foo");
        assert_eq!(arr.value(1), b"");
        assert_eq!(arr.value(2), b"bar");
        assert_eq!(arr.get(0), Some(b"foo".as_slice()));
        assert_eq!(unsafe { arr.get_unchecked(2) }, Some(b"bar".as_slice()));
    }

    #[test]
    fn null_scalar() {
        // Every element is null, and every value is empty: the mask is a single bit, the offsets
        // are the empty range every element shares, and there are no bytes at all.
        let arr = PlBinaryArray::new_full_null(1_000_000);

        assert!(arr.is_scalar());
        assert!(arr.validity_is_scalar());
        assert!(arr.offsets_are_scalar());
        assert!(arr.values().is_empty());
        assert_eq!(arr.validity().unwrap().len(), 1_000_000);
        assert_eq!(arr.scalar_offsets(), Some(0..0));
        assert_eq!(arr.null_count(), 1_000_000);
        assert!(arr.has_nulls());
        assert!(arr.is_null(999_999));
        assert_eq!(arr.get(999_999), None);
        assert_eq!(arr.value_length(999_999), 0);
        assert_eq!(arr.scalar_value(), Some(None));

        let valid = arr.without_validity();
        assert_eq!(valid.null_count(), 0);
        assert!(valid.validity().is_none());
        assert_eq!(valid.value(999_999), b"");
    }

    #[test]
    fn to_flat_lays_the_values_end_to_end() {
        // Three copies of `bar`, which share the one range they cover.
        let arr = PlBinaryArray::new_broadcast(
            Buffer::from(b"foobar".to_vec()),
            Buffer::from(vec![3u64, 6]),
            3,
            None,
        );
        let flat = arr.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.len(), 3);
        assert_eq!(flat.offsets().as_slice(), [0, 3, 6, 9]);
        assert_eq!(flat.as_slice(), b"barbarbar");
        assert_eq!(flat.null_count(), 0);
        assert_eq!(flat.value(2), b"bar");

        // The representation is not part of a value, in either direction.
        assert_eq!(*flat, arr);
        assert_eq!(arr, *flat);

        // A flat validity mask is carried over as it is.
        let masked_arr = arr
            .clone()
            .with_validity(Some(Bitmap::from_iter([true, false, true])));
        let masked = masked_arr.to_flat();
        assert!(masked.is_flat());
        assert_eq!(masked.null_count(), 1);
        assert_eq!(masked.offsets().as_slice(), [0, 3, 6, 9]);
        assert_eq!(masked.get(1), None);
    }

    #[test]
    fn an_array_of_no_elements_keeps_no_range() {
        // A single slot is scalar for no elements too, but there is no element left to read it, so
        // it is not kept: the array is flat, like every empty array, rather than scalar.
        let arr = PlBinaryArray::new_broadcast(
            Buffer::from(b"hello".to_vec()),
            Buffer::from(vec![1u64, 4]),
            0,
            Some(Bitmap::new_zeroed(1)),
        );

        assert!(arr.is_empty());
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        // The one offset that holds no starts is what is left of the range, as slicing leaves it.
        assert_eq!(arr.flat_offsets().unwrap().as_slice(), [1]);
        assert!(arr.validity().unwrap().is_empty());
    }
}
