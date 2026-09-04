use std::borrow::Cow;
use std::ops::Range;

use arrow::bitmap::Bitmap;
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
use crate::concatenate::concatenate_repeated;
use crate::flat::Flat;

mod builder;
mod flat;
mod iterator;

pub use builder::PlListArrayBuilder;
pub use iterator::{PlListIter, PlListValuesIter};

/// An immutable, cheaply cloneable sequence of `length` optional lists over one values array.
#[derive(Clone)]
pub struct PlListArray {
    values: Box<dyn PlArray>,
    /// Scalar: offsets.len() == 2
    offsets: Buffer<u64>,
    length: usize,
    /// Scalar: validity.len() == 1
    validity: Option<Bitmap>,
}

impl PlListArray {
    /// Creates a flat [`PlListArray`] out of its internal components.
    ///
    /// # Errors
    /// This function errors unless `offsets` holds `length + 1` non-decreasing offsets ending
    /// within `values`, and `validity` holds `length` bits.
    pub fn try_new(
        values: Box<dyn PlArray>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_flat_offsets_len(offsets.len(), length),
            ComputeError:
            "offsets buffer of length {} is not flat for a list array of length {}: it needs one \
             offset per element plus the end of the last",
            offsets.len(), length,
        );

        validate_offsets(&*values, &offsets)?;

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

    /// Creates a flat [`PlListArray`] out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(
        values: Box<dyn PlArray>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new(values, offsets, length, validity).unwrap()
    }

    /// Creates a flat [`PlListArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// `offsets` must be non-decreasing, hold `length + 1` offsets and end within `values`;
    /// `validity` must hold `length` bits.
    #[inline]
    pub unsafe fn new_unchecked(
        values: Box<dyn PlArray>,
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

    /// Creates a scalar [`PlListArray`] of `length` elements out of its internal components.
    ///
    /// # Errors
    /// This function errors unless `offsets` is scalar for `length`, per [`is_scalar_offsets_len`],
    /// non-decreasing and ending within `values`, and `validity` is scalar for `length`, per
    /// [`is_scalar_buffer_len`].
    pub fn try_new_broadcast(
        values: Box<dyn PlArray>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_scalar_offsets_len(offsets.len(), length),
            ComputeError:
            "offsets buffer of length {} is not the single range the {} elements of a broadcast \
             list array share: it needs the two offsets standing for that range",
            offsets.len(), length,
        );

        validate_offsets(&*values, &offsets)?;

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

    /// Creates a scalar [`PlListArray`] of `length` elements out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new_broadcast`] errors.
    #[inline]
    pub fn new_broadcast(
        values: Box<dyn PlArray>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new_broadcast(values, offsets, length, validity).unwrap()
    }

    /// Creates a scalar [`PlListArray`] of `length` elements out of its internal components without
    /// validating them.
    ///
    /// # Safety
    /// `offsets` must be non-decreasing, scalar for `length` per [`is_scalar_offsets_len`], and end
    /// within `values`; `validity` must be scalar for `length`, per [`is_scalar_buffer_len`].
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        values: Box<dyn PlArray>,
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

    /// Creates an empty [`PlListArray`] over `values`.
    #[inline]
    pub fn new_empty(values: Box<dyn PlArray>) -> Self {
        Self {
            values,
            offsets: Buffer::zeroed(1),
            length: 0,
            validity: None,
        }
    }

    /// Creates a fully valid, flat [`PlListArray`] from `values` and `offsets`, taking its length
    /// from the offsets.
    ///
    /// # Panics
    /// Panics if `offsets` is empty — the end of the last list is always needed, so even an empty
    /// array has one offset — or under the conditions [`Self::try_new`] errors.
    pub fn from_offsets(values: Box<dyn PlArray>, offsets: Buffer<u64>) -> Self {
        let length = offsets
            .len()
            .checked_sub(1)
            .expect("a list array needs at least one offset");
        Self::new(values, offsets, length, None)
    }

    /// Creates a [`PlListArray`] of `length` copies of the list `element`, in the memory of that
    /// one list.
    #[inline]
    pub fn new_scalar(element: Box<dyn PlArray>, length: usize) -> Self {
        // There is no element for the list to be shared by when there are no elements at all,
        // which is why an empty array is the one that covers no range of the values it repeats.
        if length == 0 {
            return Self::new_empty(element);
        }

        Self {
            offsets: Buffer::from_owner([0, element.len() as u64]),
            values: element,
            length,
            validity: None,
        }
    }

    /// Creates a [`PlListArray`] of `length` nulls over `values`.
    #[inline]
    pub fn new_full_null(values: Box<dyn PlArray>, length: usize) -> Self {
        Self {
            values,
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

    /// The values array the lists are taken over.
    #[inline]
    pub fn values(&self) -> &dyn PlArray {
        &*self.values
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

    /// Consumes this array into its internal components.
    #[inline]
    pub fn into_inner(self) -> (Box<dyn PlArray>, Buffer<u64>, usize, Option<Bitmap>) {
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
    /// same list.
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

    /// Whether both of this array's own backing buffers hold one slot per element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.offsets_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array's own backing buffers stand for a single list repeated [`Self::len`]
    /// times.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.offsets_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element of this array equals, if both of its own backing buffers
    /// hold one slot.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<Box<dyn PlArray>>> {
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

    /// The number of values in the element at `i`.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value_length(&self, i: usize) -> usize {
        self.value_range(i).len()
    }

    /// The number of values in the element at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_length_unchecked(&self, i: usize) -> usize {
        unsafe { self.value_range_unchecked(i) }.len()
    }

    /// Returns the element at `i`: the values array sliced to the range the element covers.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value(&self, i: usize) -> Box<dyn PlArray> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the element at `i`: the values array sliced to the range the element covers.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> Box<dyn PlArray> {
        let range = unsafe { self.value_range_unchecked(i) };
        // SAFETY: the offsets are ordered and bounded by the length of the values array.
        unsafe { self.values.sliced_unchecked(range.start, range.len()) }
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
    pub fn get(&self, i: usize) -> Option<Box<dyn PlArray>> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> Option<Box<dyn PlArray>> {
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
    pub fn values_iter(&self) -> PlListValuesIter<'_> {
        // SAFETY: the offsets are flat or scalar for this array's length, are ordered and are
        // bounded by the length of the values, all upheld by every constructor.
        PlListValuesIter::new(&*self.values, &self.offsets, self.length)
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlListIter<'_> {
        // SAFETY: the offsets are flat or scalar for this array's length, are ordered and are
        // bounded by the length of the values, all upheld by every constructor.
        PlListIter::new(&*self.values, &self.offsets, self.validity(), self.length)
    }

    /// Returns an iterator over `length` elements, repeating the single element of this array if
    /// that is all it holds, and ignoring validity.
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlListValuesIter<'_> {
        assert_broadcastable(self.length, length);
        // SAFETY: this array broadcasts to `length`, which is what was just asserted, so its
        // offsets are flat or scalar for it.
        PlListValuesIter::new(&*self.values, &self.offsets, length)
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

        // The values array the offsets point into is left as it is; see `slice_offsets`.
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

    /// Creates a [`PlListArray`] of `length` copies of the element at `index`.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        assert!(index < self.length, "index out of bounds");
        unsafe { self.new_from_index_unchecked(index, length) }
    }

    /// Creates a [`PlListArray`] of `length` copies of the element at `index`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        if unsafe { self.is_null_unchecked(index) } {
            return Self::new_full_null(self.values.clone(), length);
        }

        if length == 0 {
            return Self::new_empty(self.values.clone());
        }

        // Nothing is repeated: the values array is cloned as it is, and the two offsets every
        // element of the result shares are the ones of the element being repeated.
        let range = unsafe { self.value_range_unchecked(index) };

        Self {
            values: self.values.clone(),
            offsets: Buffer::from_owner([range.start as u64, range.end as u64]),
            length,
            validity: None,
        }
    }

    /// Returns an equivalent array whose own backing buffers both hold one slot per element,
    /// borrowing this array itself if they already do.
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
            // Every element is the empty list, or is null and therefore holds an undetermined one:
            // no value is written out, and the offsets all point at the same place. That place is
            // the start of the values array rather than the range every element covered, which is
            // the same empty list and is a buffer that need not be written out either.
            (self.values.clone(), Buffer::zeroed(self.length + 1))
        } else {
            // The one list every element covers, written out once per element. Concatenating it
            // with copies of itself is what repeats it, and that keeps the values of the result
            // scalar when the list is itself a single repeated value.
            let range = unsafe { self.value_range_unchecked(0) };
            let element = self.values.sliced(range.start, range.len());
            let values = concatenate_repeated(&*element, self.length)
                .expect("copies of one array always concatenate");

            let offsets = (0..=self.length as u64)
                .map(|i| i * range.len() as u64)
                .collect::<Vec<_>>();

            (values, Buffer::from(offsets))
        };

        // SAFETY: the offsets are ordered, one per element plus the end of the last, and within the
        // values; the mask is the flat counterpart of one valid for this array's length.
        Cow::Owned(Flat(unsafe {
            Self::new_unchecked(values, offsets, self.length, validity)
        }))
    }

    /// Borrows this array as a [`Flat`] one, if both of its own backing buffers already hold one
    /// slot per element.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: both own backing buffers of a flat array hold one slot per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }
}

impl<'a> IntoIterator for &'a PlListArray {
    type Item = Option<Box<dyn PlArray>>;
    type IntoIter = PlListIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares two arrays element-wise; neither the offsets nor the values of null elements are part
/// of a value, so an array compares equal to any other one holding the same lists.
impl PartialEq for PlListArray {
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

        (0..self.length).all(|i| unsafe {
            self.is_null_unchecked(i) || self.value_unchecked(i).eq_dyn(&*other.value_unchecked(i))
        })
    }
}

impl Eq for PlListArray {}

/// Compares an array of unknown representation against a flat one; see [`PartialEq<PlListArray> for
/// Flat<PlListArray>`](Flat).
impl PartialEq<Flat<PlListArray>> for PlListArray {
    #[inline]
    fn eq(&self, other: &Flat<PlListArray>) -> bool {
        *self == other.0
    }
}

impl std::fmt::Debug for PlListArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The values array formats its own scalar representation, so this never materializes one,
        // and neither are the offsets: they are listed as they are backed, which is two of them
        // for a scalar array.
        let mut s = f.debug_struct("PlListArray");
        s.field("length", &self.length);
        if let Some(validity) = self.validity() {
            s.field("validity", &validity);
        }
        s.field("offsets", &self.offsets.as_slice());
        s.field("values", &self.values).finish()
    }
}

impl PlArray for PlListArray {
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
        PlArrayType::List
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
fn validate_offsets(values: &dyn PlArray, offsets: &Buffer<u64>) -> PolarsResult<()> {
    // The offsets are ordered, so checking the last one against the values array covers them all —
    // including that every one of them fits in a `usize`.
    for (i, window) in offsets.windows(2).enumerate() {
        polars_ensure!(
            window[0] <= window[1],
            ComputeError:
            "offset {} of the list array is {}, which is smaller than the offset {} before it",
            i + 1, window[1], window[0],
        );
    }

    let last = offsets[offsets.len() - 1];
    polars_ensure!(
        last <= values.len() as u64,
        ComputeError:
        "the last offset of the list array is {}, which exceeds the length {} of its values",
        last, values.len(),
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlPrimitiveArray;

    /// The values array every test builds its lists over.
    fn values() -> Box<dyn PlArray> {
        Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5]))
    }

    /// The three lists `[1, 2]`, `[]` and `[3, 4, 5]` over [`values`].
    fn arr() -> PlListArray {
        PlListArray::from_offsets(values(), Buffer::from(vec![0u64, 2, 2, 5]))
    }

    /// The elements of a list, which every test takes over a `PlPrimitiveArray<i32>`.
    fn elements(list: &dyn PlArray) -> Vec<Option<i32>> {
        list.as_any()
            .downcast_ref::<PlPrimitiveArray<i32>>()
            .unwrap()
            .iter()
            .collect()
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
        assert_eq!(arr.flat_offsets().unwrap().as_slice(), [0, 2, 2, 5]);
        assert_eq!(arr.values().len(), 5);

        assert_eq!(arr.value_range(0), 0..2);
        assert_eq!(arr.value_range(1), 2..2);
        assert_eq!(arr.value_range(2), 2..5);
        assert_eq!(arr.value_length(0), 2);
        assert_eq!(arr.value_length(1), 0);
        assert_eq!(arr.value_length(2), 3);

        assert_eq!(elements(&*arr.value(0)), [Some(1), Some(2)]);
        assert!(elements(&*arr.value(1)).is_empty());
        assert_eq!(elements(&*arr.value(2)), [Some(3), Some(4), Some(5)]);
        assert_eq!(elements(&*arr.get(0).unwrap()), [Some(1), Some(2)]);
    }

    #[test]
    fn null_scalar() {
        // Every element is null, and every list is empty: the mask is a single bit, and the offsets
        // are the empty range every element shares.
        let arr = PlListArray::new_full_null(values(), 1_000_000);

        assert!(arr.is_scalar());
        assert!(arr.validity_is_scalar());
        assert!(arr.offsets_are_scalar());
        assert_eq!(arr.validity().unwrap().len(), 1_000_000);
        assert_eq!(arr.scalar_offsets(), Some(0..0));
        assert_eq!(arr.null_count(), 1_000_000);
        assert!(arr.has_nulls());
        assert!(arr.is_null(999_999));
        assert_eq!(arr.get(999_999), None);
        assert_eq!(arr.value_length(999_999), 0);

        // The values are untouched: it is the list array that masks the elements out.
        assert_eq!(arr.values().null_count(), 0);

        let valid = arr.without_validity();
        assert_eq!(valid.null_count(), 0);
        assert!(valid.validity().is_none());
    }

    #[test]
    fn to_flat_lays_the_lists_end_to_end() {
        // Three copies of `[3, 4, 5]`, which share the one range they cover.
        let arr = PlListArray::new_broadcast(values(), Buffer::from(vec![2u64, 5]), 3, None);
        let flat = arr.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.len(), 3);
        assert_eq!(flat.offsets().as_slice(), [0, 3, 6, 9]);
        assert_eq!(flat.values().len(), 9);
        assert_eq!(flat.null_count(), 0);
        for i in 0..flat.len() {
            assert_eq!(elements(&*flat.value(i)), [Some(3), Some(4), Some(5)]);
        }

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
        let arr = PlListArray::new_broadcast(
            values(),
            Buffer::from(vec![2u64, 5]),
            0,
            Some(Bitmap::new_zeroed(1)),
        );

        assert!(arr.is_empty());
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        // The one offset that holds no starts is what is left of the range, as slicing leaves it.
        assert_eq!(arr.flat_offsets().unwrap().as_slice(), [2]);
        assert!(arr.validity().unwrap().is_empty());
    }
}
