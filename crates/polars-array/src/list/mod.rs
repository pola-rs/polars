use std::ops::Range;

use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmapRef, validity_eq};
use crate::broadcast::{
    assert_broadcastable, broadcast_index, is_flat_buffer_len, is_flat_offsets_len,
    is_scalar_buffer_len, is_scalar_offsets_len, is_valid_buffer_len,
};
use crate::concatenate::concatenate_repeated;
use crate::flat::Flat;

mod builder;
mod flat;
mod iterator;

pub use builder::PlListArrayBuilder;
pub use iterator::{PlListIter, PlListValuesIter};

/// An immutable, cheaply cloneable sequence of `length` optional lists over one values array.
///
/// This is the variable-length nested array of this crate: it holds no values of its own, only a
/// validity mask and the offsets that cut a single values array into `length` consecutive slices.
/// Element `i` is `values[offsets[i]..offsets[i + 1]]`. It carries no logical type — the values
/// array is a [`PlArray`], and what a caller thinks of as the list's inner type lives at a higher
/// level.
///
/// The offsets are always `u64`, and what the separate `length` field buys is what it buys
/// everywhere else in this crate: each backing buffer is either flat or scalar, so a list array
/// whose length is unbounded by its memory use is representable. The offsets hold the start of
/// every element plus the end of the last, so it is their *starts* that are flat or scalar:
///
/// * *flat*: `length + 1` offsets, one range per element laid end to end — the end of one list is
///   the start of the next, and the last list needs an end of its own.
/// * *scalar*: two offsets, the one range every element covers, which is what lets a single list
///   repeated `length` times cost the memory of that one list.
///
/// Element `i` is read through
/// [`broadcast_index(i, offsets.len() - 1)`](crate::broadcast::broadcast_index), and the validity
/// mask through [`broadcast_index(i, bitmap.len())`](crate::broadcast::broadcast_index), so it is
/// either flat (one bit per element) or scalar (a single bit shared by every element), which is
/// what lets a fully null list array carry a one-bit mask. See [`crate::broadcast`] for the full
/// rules.
///
/// The values array is never trimmed to what the offsets reach: it may hold elements before the
/// first offset and after the last, and after slicing it usually does.
///
/// # Example
/// ```
/// use polars_array::{PlArray, PlListArray, PlPrimitiveArray};
/// use polars_buffer::Buffer;
///
/// // Three lists over five values: `[1, 2]`, `[]` and `[3, 4, 5]`.
/// let arr = PlListArray::from_offsets(
///     Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5])),
///     Buffer::from(vec![0u64, 2, 2, 5]),
/// );
/// assert_eq!(arr.len(), 3);
/// assert_eq!(arr.null_count(), 0);
/// assert_eq!(arr.value_length(0), 2);
/// assert_eq!(arr.value_range(2), 2..5);
///
/// // Reading an element slices the values array, which is `O(1)`.
/// let element = arr.value(2);
/// assert_eq!(element.len(), 3);
/// assert_eq!(
///     element
///         .as_any()
///         .downcast_ref::<PlPrimitiveArray<i32>>()
///         .unwrap()
///         .flat_values()
///         .unwrap()
///         .as_slice(),
///     [3, 4, 5],
/// );
///
/// // A billion copies of one list cost that list: the offsets are the range they all share.
/// let scalar = PlListArray::new_scalar(
///     Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
///     1_000_000_000,
/// );
/// assert_eq!(scalar.len(), 1_000_000_000);
/// assert_eq!(scalar.scalar_offsets(), Some(0..2));
/// assert!(scalar.flat_offsets().is_none());
/// assert_eq!(scalar.value_range(999_999_999), 0..2);
/// ```
#[derive(Clone)]
pub struct PlListArray {
    values: Box<dyn PlArray>,
    offsets: Buffer<u64>,
    length: usize,
    validity: Option<Bitmap>,
}

impl PlListArray {
    /// Creates a flat [`PlListArray`] out of its internal components.
    ///
    /// The offsets have to hold the range of every element — one per element, plus the end of the
    /// last — and the validity mask one bit per element. [`Self::try_new_broadcast`] is what builds
    /// the scalar representation; this function never infers it from offsets that happen to hold a
    /// single range. This function walks the offsets to check that they are ordered, so it is
    /// `O(len)`.
    ///
    /// # Errors
    /// This function errors if `offsets` does not hold exactly `length + 1` offsets, if the offsets
    /// are not monotonically non-decreasing, if the last offset exceeds the length of `values`, or
    /// if `validity` does not hold exactly `length` bits.
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
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `offsets` must be monotonically non-decreasing, hold exactly `length + 1` offsets, and end
    /// at an offset that does not exceed the length of `values`; `validity` must hold exactly
    /// `length` bits.
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
    /// The offsets have to hold the single range every element covers, and the validity mask the
    /// single bit they share, which makes this `O(1)` in `length`. [`Self::try_new`] is what builds
    /// the flat representation.
    ///
    /// # Errors
    /// This function errors if `offsets` does not hold exactly two offsets, if they are not
    /// monotonically non-decreasing, if the last of them exceeds the length of `values`, or if
    /// `validity` does not hold exactly one bit. An array of no elements covers no range, so it
    /// additionally admits the single offset that begins no element and an empty mask.
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
            offsets,
            length,
            validity,
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

    /// Creates a scalar [`PlListArray`] of `length` elements out of its internal components
    /// without validating them.
    ///
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `offsets` must be monotonically non-decreasing, hold exactly two offsets — or the single one
    /// that begins no element, if `length` is zero — and end at an offset that does not exceed the
    /// length of `values`; `validity` must hold exactly one bit, or none at all if `length` is
    /// zero.
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
            offsets,
            length,
            validity,
        }
    }

    /// Creates an empty [`PlListArray`] over `values`.
    ///
    /// The values array is kept as it is — it is what determines the type of the lists, of which
    /// there are none — but no element of it is reachable.
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
    /// The offsets are read as flat — `length + 1` of them — so this never builds the scalar
    /// representation: [`Self::new_scalar`] is what does. This function is `O(len)`.
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
    ///
    /// The list is given as the array of its values, which becomes the values array of the result:
    /// every element covers all of it, through the two offsets they share. This function is `O(1)`,
    /// and so is the result's memory use on top of `element`. Repeating a list that is already an
    /// element of a list array is [`Self::new_from_index`].
    #[inline]
    pub fn new_scalar(element: Box<dyn PlArray>, length: usize) -> Self {
        let offsets = Buffer::from_owner([0, element.len() as u64]);
        Self {
            values: element,
            offsets,
            length,
            validity: None,
        }
    }

    /// Creates a [`PlListArray`] of `length` nulls over `values`.
    ///
    /// Every element is null, so its value is undetermined; each is given the empty list, which is
    /// what keeps both the validity mask and the offsets a single shared slot, and the values array
    /// untouched. This function is `O(1)`.
    #[inline]
    pub fn new_full_null(values: Box<dyn PlArray>, length: usize) -> Self {
        Self {
            values,
            offsets: Buffer::zeroed(2),
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

    /// The values array the lists are taken over.
    ///
    /// This is *not* trimmed to what the offsets reach: it may hold elements before the first
    /// offset and after the last. Read an element of this array with [`Self::value`] instead of
    /// indexing it directly.
    #[inline]
    pub fn values(&self) -> &dyn PlArray {
        &*self.values
    }

    /// The backing offsets buffer, if it holds the range of every element, laid end to end.
    ///
    /// Element `i` then covers `offsets[i]..offsets[i + 1]` of [`Self::values`], with no
    /// [`broadcast_index`] in the way, and the buffer holds [`Self::len`] `+ 1` offsets — the
    /// start of every element plus the end of the last. This is the `O(1)` counterpart of
    /// [`Self::to_flat`]: it materializes nothing, and returns `None` rather than writing out
    /// scalar offsets. Reach for the range scalar offsets share with
    /// [`Self::scalar_offsets`] instead — between them the two cover every array that has elements
    /// at all, so a `None` from both is an empty array.
    ///
    /// The offsets are not normalized: the first one is whatever slicing left it, not necessarily
    /// zero.
    #[inline]
    pub fn flat_offsets(&self) -> Option<&Buffer<u64>> {
        self.offsets_are_flat().then_some(&self.offsets)
    }

    /// The range of [`Self::values`] every element of this array covers, if the offsets hold a
    /// single range.
    ///
    /// This is the offsets half of [`Self::scalar_value`], which additionally asks that the
    /// validity mask be scalar and reports the null the mask makes of this list. Returns `None`
    /// for offsets that are flat over more than one element, and for an empty array, which has no
    /// element to share a range. The range of a null element is undetermined (it can be any valid
    /// range).
    #[inline]
    pub fn scalar_offsets(&self) -> Option<Range<usize>> {
        // SAFETY: the array is not empty, so element 0 is in bounds.
        (self.offsets_are_scalar() && self.length > 0)
            .then(|| unsafe { self.value_range_unchecked(0) })
    }

    /// Consumes this array into its internal components.
    ///
    /// The offsets are *not* guaranteed to hold [`Self::len`] `+ 1` slots: they are either flat or
    /// scalar, which is why the length comes with them. See [`crate::broadcast`] for how to read
    /// them.
    #[inline]
    pub fn into_inner(self) -> (Box<dyn PlArray>, Buffer<u64>, usize, Option<Bitmap>) {
        (self.values, self.offsets, self.length, self.validity)
    }

    /// The validity mask, if any element may be null.
    ///
    /// The returned [`PlBitmapRef`] has [`Self::len`] bits regardless of whether the backing bitmap
    /// is flat or scalar, so reading validity through it needs no knowledge of which representation
    /// this array is in. This mask says nothing about the values: a valid list may still hold null
    /// values, and so may a list of length zero.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_broadcast_unchecked(validity, self.length) })
    }

    /// Whether the offsets hold the single range every element covers, so that every element is
    /// the same list.
    ///
    /// An array of one element is both scalar and [`flat`](Self::is_flat): the two representations
    /// coincide, and this reports them both.
    #[inline]
    pub fn offsets_are_scalar(&self) -> bool {
        // The offsets hold one slot more than the starts that are flat or scalar for this array's
        // length, so the two of a scalar array are a single start and the end of it.
        self.offsets.len() == 2
    }

    /// Whether the offsets hold the range of every element, laid end to end.
    ///
    /// An array of one element is both flat and [`scalar`](Self::offsets_are_scalar).
    #[inline]
    pub fn offsets_are_flat(&self) -> bool {
        // The offsets are never empty, and hold the start of every element plus the end of the
        // last.
        self.offsets.len() - 1 == self.length
    }

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether both of this array's own backing buffers hold one slot per element.
    ///
    /// The values array carries its own representation, which this says nothing about. An array of
    /// one element is both flat and [`scalar`](Self::is_scalar).
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.offsets_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array's own backing buffers are entirely in the scalar representation, and
    /// therefore stand for a single list repeated [`Self::len`] times in the memory of that list
    /// alone.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.offsets_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element of this array equals, if both of its own backing buffers
    /// hold one slot.
    ///
    /// The inner [`Option`] is that element, so an array of nothing but nulls yields `Some(None)`.
    /// Returns `None` for an empty array, and whenever a buffer is flat over more than one element
    /// — its elements need not be equal, even if the other buffer is scalar.
    ///
    /// This is what lets equality avoid walking a scalar array of unbounded length.
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
    /// The range of a null element is undetermined (it can be any valid range).
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
    /// The range of a null element is undetermined (it can be any valid range).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_range_unchecked(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.length);

        // Scalar offsets hold the one range every element covers, so they are read at slot zero.
        let i = broadcast_index(i, self.offsets.len() - 1);

        // SAFETY: the offsets hold one slot more than the starts `broadcast_index` maps onto, so
        // `i + 1` is in bounds. Every offset is at most the length of the values array, and
        // therefore fits in a `usize`.
        unsafe {
            let start = *self.offsets.get_unchecked(i) as usize;
            let end = *self.offsets.get_unchecked(i + 1) as usize;
            start..end
        }
    }

    /// The number of values in the element at `i`.
    ///
    /// The length of a null element is undetermined (it can be anything).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value_length(&self, i: usize) -> usize {
        self.value_range(i).len()
    }

    /// The number of values in the element at `i`.
    ///
    /// The length of a null element is undetermined (it can be anything).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_length_unchecked(&self, i: usize) -> usize {
        unsafe { self.value_range_unchecked(i) }.len()
    }

    /// Returns the element at `i`: the values array sliced to the range the element covers.
    ///
    /// This function is `O(1)`. The value of a null element is undetermined (it can be any list).
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
    /// This function is `O(1)`. The value of a null element is undetermined (it can be any list).
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
    /// A valid list may still hold null values, and so may a list of length zero.
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
    /// Null values inside the lists do not count: only elements this array itself masks out are
    /// null.
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
    /// The values of null elements are undetermined (they can be any list).
    #[inline]
    pub fn values_iter(&self) -> PlListValuesIter<'_> {
        PlListValuesIter::new(self)
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlListIter<'_> {
        PlListIter::new(self)
    }

    /// Returns an iterator over `length` elements, repeating the single element of this array if
    /// that is all it holds, and ignoring validity.
    ///
    /// This array either has `length` elements — in which case this is [`Self::values_iter`] — or
    /// a single element, which the `length` values this yields are then all read from.
    /// Broadcasting is `O(1)`, and allocates nothing: the value is repeated as it is read, rather
    /// than materialized into an array to iterate the way [`Self::new_from_index`] would have to.
    /// The values of null elements are undetermined (they can be any list).
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlListValuesIter<'_> {
        assert_broadcastable(self.length, length);
        // SAFETY: this array broadcasts to `length`, which is what was just asserted.
        PlListValuesIter::new_broadcast(self, length)
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
    ///
    /// The values keep their own validity: a list that is valid may still hold null values.
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

        // The values array is left as it is: the offsets are what point into it, and they are not
        // normalized, so the elements that fall outside the slice simply stop being reachable.
        // Scalar offsets are left alone as well, like a scalar mask: every element of the slice
        // covers the same range every element of this array does.
        if self.offsets_are_flat() {
            unsafe {
                self.offsets
                    .slice_in_place_unchecked(offset..offset + length + 1)
            };
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

    /// Creates a [`PlListArray`] of `length` copies of the element at `index`.
    ///
    /// This function is `O(1)`: the offsets of the result are scalar, so every one of its elements
    /// covers the range the element covers, of the very same values array. A null element repeats
    /// as `length` nulls.
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
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        if unsafe { self.is_null_unchecked(index) } {
            return Self::new_full_null(self.values.clone(), length);
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

    /// Returns an equivalent array whose own backing buffers both hold one slot per element.
    ///
    /// The values array is left as it is: it carries its own representation, which being
    /// [`flat`](Self::is_flat) says nothing about. The result carries its representation in its
    /// type: see [`Flat`] for what a flat array is a proof of.
    ///
    /// Materializing scalar offsets is what costs here, and it costs more than it does for the
    /// other arrays: flat offsets lay the ranges of the elements end to end, so the one list every
    /// element of a scalar array covers has to be written out once per element — this is `O(len *
    /// value_length)`, and it is [`concatenate_repeated`] that does it, so the values of the
    /// result keep whatever representation copies of that list concatenate into. It is only the
    /// offsets that are materialized, in `O(len)`, when every element covers an empty range or is
    /// null: the value of a null element is undetermined, so it need not be written out, and the
    /// empty list is the same list wherever the offsets point.
    ///
    /// # Example
    /// ```
    /// use polars_array::{PlArray, PlListArray, PlPrimitiveArray};
    ///
    /// // Three copies of `[1, 2]`, over the values of that one list.
    /// let scalar = PlListArray::new_scalar(
    ///     Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
    ///     3,
    /// );
    /// assert_eq!(scalar.scalar_offsets(), Some(0..2));
    ///
    /// // Its flat counterpart holds the three lists one after the other.
    /// let flat = scalar.to_flat();
    /// assert_eq!(flat.offsets().as_slice(), [0, 2, 4, 6]);
    /// assert_eq!(flat.values().len(), 6);
    /// assert_eq!(flat, scalar);
    /// ```
    pub fn to_flat(&self) -> Flat<Self> {
        if self.is_flat() {
            return Flat(self.clone());
        }

        let validity = self.validity().map(|validity| validity.to_flat());

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

        // SAFETY: the offsets are ordered, there is one per element plus the end of the last, and
        // the last of them is within the values they were built for: the length of the values
        // repeated once per element, or zero, which every values array holds. The mask is the flat
        // counterpart of one that was flat or scalar for this array's length.
        Flat(unsafe { Self::new_unchecked(values, offsets, self.length, validity) })
    }

    /// Borrows this array as a [`Flat`] one, if both of its own backing buffers already hold one
    /// slot per element.
    ///
    /// This is the `O(1)` counterpart of [`Self::to_flat`]: it materializes nothing, and returns
    /// `None` rather than writing out a scalar buffer when this array is not
    /// [`flat`](Self::is_flat).
    ///
    /// # Example
    /// ```
    /// use polars_array::{PlListArray, PlPrimitiveArray};
    /// use polars_buffer::Buffer;
    ///
    /// let arr = PlListArray::from_offsets(
    ///     Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
    ///     Buffer::from(vec![0u64, 2, 3]),
    /// );
    /// assert!(arr.as_flat().is_some());
    ///
    /// // A billion copies of one list share two offsets, so they have to be written out.
    /// let scalar = PlListArray::new_scalar(
    ///     Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
    ///     1_000_000_000,
    /// );
    /// assert!(scalar.as_flat().is_none());
    /// ```
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: both own backing buffers of a flat array hold one slot per element.
        self.is_flat()
            .then(|| unsafe { Flat::new_ref(self) })
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

/// Compares an array of unknown representation against a flat one; see
/// [`PartialEq<PlListArray> for Flat<PlListArray>`](Flat).
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
///
/// This is the half of the validation that both families of constructors share; how many offsets
/// there are is what tells the flat representation from the scalar one, and is checked separately.
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
    use crate::{PlBooleanArray, PlPrimitiveArray, PlStructArray};

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
    fn the_offsets_are_reached_through_their_representation() {
        let arr = arr();

        assert_eq!(
            arr.flat_offsets().map(Buffer::as_slice),
            Some([0, 2, 2, 5].as_slice())
        );
        assert_eq!(arr.scalar_offsets(), None);

        // Scalar offsets hand out no flat buffer; it is the one range they hold that is reached.
        let element = Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2]));
        let arr = PlListArray::new_scalar(element, 1_000_000_000);

        assert_eq!(arr.flat_offsets(), None);
        assert_eq!(arr.scalar_offsets(), Some(0..2));

        // The offsets are read whether or not the mask makes the elements null.
        let arr = arr.sliced(0, 3).with_validity(Some(Bitmap::new_zeroed(3)));

        assert_eq!(arr.scalar_offsets(), Some(0..2));
        assert!(
            arr.scalar_value().is_none(),
            "a flat mask leaves no shared element"
        );

        // An empty array has no range to share, and is flat unless it is backed by a stray one.
        assert_eq!(
            PlListArray::new_empty(values())
                .flat_offsets()
                .map(Buffer::as_slice),
            Some([0].as_slice()),
        );

        let empty = PlListArray::new_scalar(values(), 0);
        assert_eq!(empty.flat_offsets(), None);
        assert_eq!(empty.scalar_offsets(), None);
    }

    #[test]
    fn the_values_outside_the_offsets_are_unreachable() {
        // The first list starts after the first value and the last ends before the last one.
        let arr = PlListArray::from_offsets(values(), Buffer::from(vec![1u64, 3]));

        assert_eq!(arr.len(), 1);
        assert_eq!(arr.values().len(), 5);
        assert_eq!(elements(&*arr.value(0)), [Some(2), Some(3)]);
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
    fn scalar_validity_over_flat_offsets() {
        let arr = arr().with_validity_broadcast(Some(Bitmap::new_zeroed(1)));

        assert!(arr.validity_is_scalar());
        assert_eq!(arr.null_count(), 3);
        assert_eq!(arr.iter().collect::<Vec<_>>().len(), 3);
        assert!(arr.iter().all(|element| element.is_none()));

        // The offsets are untouched, so the values of the null elements are still there.
        assert_eq!(elements(&*arr.value(0)), [Some(1), Some(2)]);
        assert_eq!(
            arr.values_iter().map(|list| list.len()).collect::<Vec<_>>(),
            [2, 0, 3],
        );
    }

    #[test]
    fn flat_validity() {
        let arr = PlListArray::new(
            values(),
            Buffer::from(vec![0u64, 2, 2, 5]),
            3,
            Some(Bitmap::from_iter([true, false, true])),
        );

        assert!(!arr.validity_is_scalar());
        assert_eq!(arr.null_count(), 1);
        assert!(arr.is_null(1));
        assert_eq!(
            arr.iter()
                .map(|element| element.map(|list| list.len()))
                .collect::<Vec<_>>(),
            [Some(2), None, Some(3)],
        );
    }

    #[test]
    fn try_new_rejects_invalid_components() {
        // The offsets must hold one slot per element plus the end of the last.
        assert!(PlListArray::try_new(values(), Buffer::from(vec![0u64, 2, 5]), 3, None).is_err());
        assert!(PlListArray::try_new(values(), Buffer::from(vec![0u64, 2, 2, 5]), 3, None).is_ok());

        // Two offsets are a scalar array rather than a flat one, and are never inferred to be
        // either — see `scalar_offsets_stand_for_the_range_every_element_covers`.
        assert!(PlListArray::try_new(values(), Buffer::from(vec![0u64, 2]), 3, None).is_err());

        // They must be monotonically non-decreasing.
        assert!(
            PlListArray::try_new(values(), Buffer::from(vec![0u64, 5, 2, 5]), 3, None).is_err()
        );

        // The last of them must not reach past the values array.
        assert!(PlListArray::try_new(values(), Buffer::from(vec![0u64, 6]), 1, None).is_err());
        assert!(PlListArray::try_new(values(), Buffer::from(vec![0u64, 5]), 1, None).is_ok());

        // The validity mask has to be flat as well.
        let offsets = Buffer::from(vec![0u64, 2, 2, 5]);
        assert!(
            PlListArray::try_new(values(), offsets.clone(), 3, Some(Bitmap::new_zeroed(2)))
                .is_err()
        );
        assert!(
            PlListArray::try_new(values(), offsets.clone(), 3, Some(Bitmap::new_zeroed(1)))
                .is_err()
        );
        assert!(PlListArray::try_new(values(), offsets, 3, Some(Bitmap::new_zeroed(3))).is_ok());
    }

    #[test]
    fn try_new_broadcast_requires_scalar_offsets() {
        let scalar = || Buffer::from(vec![0u64, 2]);

        // The offsets must hold the one range every element covers.
        assert!(PlListArray::try_new_broadcast(values(), scalar(), 1_000_000, None).is_ok());
        assert!(
            PlListArray::try_new_broadcast(values(), Buffer::from(vec![0u64, 2, 2, 5]), 3, None)
                .is_err()
        );

        // They are checked exactly as the flat ones are.
        assert!(
            PlListArray::try_new_broadcast(values(), Buffer::from(vec![5u64, 2]), 3, None).is_err()
        );
        assert!(
            PlListArray::try_new_broadcast(values(), Buffer::from(vec![0u64, 6]), 3, None).is_err()
        );

        // The validity mask has to be scalar as well.
        assert!(
            PlListArray::try_new_broadcast(values(), scalar(), 3, Some(Bitmap::new_zeroed(3)))
                .is_err()
        );
        assert!(
            PlListArray::try_new_broadcast(values(), scalar(), 3, Some(Bitmap::new_zeroed(1)))
                .is_ok()
        );

        // An array of no elements covers no range, so the single offset that begins no element
        // stands for it as well.
        assert!(
            PlListArray::try_new_broadcast(values(), Buffer::from(vec![0u64]), 0, None).is_ok()
        );
        assert!(PlListArray::try_new_broadcast(values(), scalar(), 0, None).is_ok());
        assert!(PlListArray::try_new_broadcast(values(), Buffer::new(), 0, None).is_err());
    }

    #[test]
    fn slicing_only_slices_the_offsets() {
        let arr = PlListArray::new(
            values(),
            Buffer::from(vec![0u64, 2, 2, 5]),
            3,
            Some(Bitmap::from_iter([true, false, true])),
        )
        .sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.flat_offsets().unwrap().as_slice(), [2, 2, 5]);
        assert_eq!(arr.validity().unwrap().flat_bitmap().unwrap().len(), 2);
        assert_eq!(arr.null_count(), 1);

        // The values array is left alone; the offsets are what point into it.
        assert_eq!(arr.values().len(), 5);
        assert_eq!(elements(&*arr.value(1)), [Some(3), Some(4), Some(5)]);

        // Slicing away every element leaves the end of the last list behind.
        let arr = arr.sliced(2, 0);
        assert!(arr.is_empty());
        assert_eq!(arr.flat_offsets().unwrap().as_slice(), [5]);
    }

    #[test]
    fn slicing_keeps_scalar_validity() {
        let arr = arr()
            .with_validity_broadcast(Some(Bitmap::new_zeroed(1)))
            .sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.validity().unwrap().len(), 2);
        assert!(arr.validity_is_scalar());
        assert_eq!(arr.null_count(), 2);
    }

    #[test]
    #[should_panic(expected = "must be smaller than the length")]
    fn slicing_out_of_bounds_panics() {
        let _ = arr().sliced(2, 2);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn reading_out_of_bounds_panics() {
        let _ = arr().value(3);
    }

    #[test]
    fn equality_ignores_the_offsets_and_the_values_behind_them() {
        // The same three lists, laid out over a different values array with different offsets.
        let other = PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::from_vec(vec![9i32, 1, 2, 3, 4, 5])),
            Buffer::from(vec![1u64, 3, 3, 6]),
        );

        assert_eq!(arr(), other);
        assert_eq!(arr(), arr());
        assert_ne!(arr(), arr().sliced(0, 2));

        // Same values, differently cut into lists.
        assert_ne!(
            arr(),
            PlListArray::from_offsets(values(), Buffer::from(vec![0u64, 2, 3, 5])),
        );

        // An absent mask and an all-set one are the same thing.
        assert_eq!(
            arr(),
            arr().with_validity(Some(Bitmap::new_with_value(true, 3))),
        );
        assert_ne!(
            arr(),
            arr().with_validity(Some(Bitmap::from_iter([true, false, true]))),
        );

        // Same lists, different element type.
        assert_ne!(
            arr(),
            PlListArray::from_offsets(
                Box::new(PlPrimitiveArray::from_vec(vec![1i64, 2, 3, 4, 5])),
                Buffer::from(vec![0u64, 2, 2, 5]),
            ),
        );
    }

    #[test]
    fn equality_ignores_the_values_of_null_elements() {
        // The lists `[1, 2]`, `[3]` and `[4, 5]`, of which the second is null.
        let mask = Bitmap::from_iter([true, false, true]);
        let offsets = Buffer::from(vec![0u64, 2, 3, 5]);
        let lhs = PlListArray::new(values(), offsets.clone(), 3, Some(mask.clone()));

        // The second element holds a different value, but it is null on both sides.
        let rhs = PlListArray::new(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 9, 4, 5])),
            offsets.clone(),
            3,
            Some(mask.clone()),
        );
        assert_eq!(lhs, rhs);

        // A null value inside a valid list still counts.
        let with_null_value = PlListArray::new(
            Box::new(PlPrimitiveArray::from_iter([
                Some(1i32),
                None,
                Some(3),
                Some(4),
                Some(5),
            ])),
            offsets,
            3,
            Some(mask),
        );
        assert_ne!(lhs, with_null_value);
    }

    #[test]
    fn equality_of_fully_null_arrays_ignores_their_lists() {
        let null = PlListArray::new_full_null(values(), 3);

        assert_eq!(null, null.clone());
        assert_eq!(
            null,
            arr().with_validity_broadcast(Some(Bitmap::new_zeroed(1))),
            "every element is null on both sides, so no list is determined",
        );
        assert_ne!(null, arr());
        assert_ne!(null, PlListArray::new_full_null(values(), 4));
    }

    #[test]
    fn empty() {
        let arr = PlListArray::new_empty(values());

        assert!(arr.is_empty());
        assert_eq!(arr.null_count(), 0);
        assert_eq!(arr.flat_offsets().unwrap().as_slice(), [0]);
        assert_eq!(arr.iter().next(), None);
        assert_eq!(
            arr,
            PlListArray::from_offsets(values(), Buffer::from(vec![0u64])),
        );
    }

    #[test]
    #[should_panic(expected = "at least one offset")]
    fn an_array_without_offsets_has_no_length_to_take() {
        let _ = PlListArray::from_offsets(values(), Buffer::new());
    }

    #[test]
    fn lists_over_any_array() {
        // The values array is a `PlArray`, so the lists nest arbitrarily. Their inner array keeps
        // its own scalar representation.
        let inner = PlListArray::from_offsets(
            Box::new(PlBooleanArray::new_scalar(true, 1_000)),
            Buffer::from(vec![0u64, 400, 1_000]),
        );
        let outer = PlListArray::from_offsets(Box::new(inner), Buffer::from(vec![0u64, 1, 2]));

        assert_eq!(outer.len(), 2);
        assert_eq!(outer.values().array_type(), PlArrayType::List);
        let element = outer.value(1);
        let element = element.as_any().downcast_ref::<PlListArray>().unwrap();
        assert_eq!(element.len(), 1);
        assert_eq!(element.value_length(0), 600);

        let structs = PlListArray::from_offsets(
            Box::new(PlStructArray::from_fields(vec![Box::new(
                PlPrimitiveArray::from_vec(vec![1i32, 2, 3]),
            )])),
            Buffer::from(vec![0u64, 1, 3]),
        );
        assert_eq!(structs.value(1).array_type(), PlArrayType::Struct);
        assert_eq!(structs.value_length(1), 2);
    }

    #[test]
    fn iterators_are_exact_sized() {
        let arr = arr();

        assert_eq!(arr.iter().len(), 3);
        assert_eq!(arr.values_iter().len(), 3);
        assert_eq!(arr.iter().size_hint(), (3, Some(3)));
        assert_eq!(
            arr.values_iter()
                .rev()
                .map(|list| list.len())
                .collect::<Vec<_>>(),
            [3, 0, 2],
        );
        assert_eq!(
            (&arr)
                .into_iter()
                .map(|element| element.is_some())
                .collect::<Vec<_>>(),
            [true, true, true],
        );
    }

    #[test]
    fn into_inner_returns_the_components() {
        let (values, offsets, length, validity) = PlListArray::new_full_null(values(), 3)
            .with_validity_broadcast(Some(Bitmap::new_zeroed(1)))
            .into_inner();

        assert_eq!(values.len(), 5);
        assert_eq!(offsets.as_slice(), [0, 0]);
        assert_eq!(length, 3);
        assert_eq!(validity, Some(Bitmap::new_zeroed(1)));
    }

    #[test]
    fn debug_lists_the_offsets_and_the_values() {
        assert_eq!(
            format!("{:?}", arr()),
            "PlListArray { length: 3, offsets: [0, 2, 2, 5], \
             values: PlPrimitiveArray[1, 2, 3, 4, 5] }",
        );

        // Neither a scalar validity mask nor a scalar values array is materialized.
        let arr = PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::new_scalar(7i32, 1_000_000_000)),
            Buffer::from(vec![0u64, 1_000_000_000]),
        )
        .with_validity_broadcast(Some(Bitmap::new_zeroed(1)));
        assert_eq!(
            format!("{arr:?}"),
            "PlListArray { length: 1, validity: PlBitmapRef[false], \
             offsets: [0, 1000000000], values: PlPrimitiveArray[7; 1000000000] }",
        );

        // Neither are scalar offsets: they are listed as they are backed, which is the two of them
        // every element of a billion shares.
        assert_eq!(
            format!(
                "{:?}",
                arr.without_validity().new_from_index(0, 1_000_000_000)
            ),
            "PlListArray { length: 1000000000, offsets: [0, 1000000000], \
             values: PlPrimitiveArray[7; 1000000000] }",
        );
    }

    #[test]
    fn behind_the_trait_object() {
        let arr: Box<dyn PlArray> = Box::new(arr());

        assert_eq!(arr.array_type(), PlArrayType::List);
        assert!(arr.array_type().is_list());
        assert!(!arr.array_type().is_struct());
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 0);
        assert!(arr.validity().is_none());
        assert_eq!(&arr, &arr.clone());

        let nulled = arr.with_validity_broadcast(Some(Bitmap::new_zeroed(1)));
        assert_eq!(nulled.null_count(), 3);
        assert!(nulled.validity().unwrap().is_scalar());
        assert_eq!(arr.null_count(), 0);

        let sliced = arr.sliced(1, 2);
        assert_eq!(sliced.len(), 2);
        assert_eq!(
            sliced
                .as_any()
                .downcast_ref::<PlListArray>()
                .unwrap()
                .value_length(1),
            3,
        );

        // A list array is not the array its lists are taken over.
        assert_ne!(&arr, &values());
    }

    #[test]
    fn new_from_index_repeats_one_list() {
        let arr = arr();

        // Nothing is repeated: every element of the result covers the range the element covers,
        // of the very same values array.
        let repeated = arr.new_from_index(2, 3);
        assert_eq!(repeated.len(), 3);
        assert!(repeated.is_scalar());
        assert_eq!(repeated.scalar_offsets(), Some(2..5));
        assert_eq!(repeated.values().len(), 5);
        assert_eq!(repeated.null_count(), 0);
        for i in 0..repeated.len() {
            assert_eq!(elements(&*repeated.value(i)), [Some(3), Some(4), Some(5)]);
        }

        // An empty element repeats as the empty range it covers.
        let repeated = arr.new_from_index(1, 4);
        assert_eq!(repeated.scalar_offsets(), Some(2..2));
        assert_eq!(repeated.values().len(), 5);
        assert!(elements(&*repeated.value(3)).is_empty());

        // A null element repeats as nulls, under a mask of a single bit, and every list is empty.
        let nulls = arr
            .clone()
            .with_validity(Some(Bitmap::from_iter([false, true, true])))
            .new_from_index(0, 4);
        assert_eq!(nulls.null_count(), 4);
        assert!(nulls.is_scalar());
        assert_eq!(nulls.scalar_offsets(), Some(0..0));
        assert_eq!(nulls.get(3), None);

        assert!(arr.new_from_index(0, 0).is_empty());

        // The representation is not part of a value: the repetition of `[1, 2]` compares equal to
        // the flat array of the two lists it stands for.
        assert_eq!(
            unsafe { arr.new_from_index_unchecked(0, 2) },
            PlListArray::from_offsets(
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 1, 2])),
                Buffer::from(vec![0u64, 2, 4]),
            ),
        );
    }

    #[test]
    fn a_repeated_list_costs_that_one_list() {
        // A single list of a thousand values, repeated a billion times.
        let arr = PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::from_vec((0..1_000i32).collect())),
            Buffer::from(vec![0u64, 1_000]),
        );
        let repeated = arr.new_from_index(0, 1_000_000_000);

        // Neither the values nor the offsets are written out: the offsets are the one range every
        // element covers, of the values array the element was already taken over.
        assert_eq!(repeated.len(), 1_000_000_000);
        assert!(repeated.is_scalar());
        assert!(!repeated.is_flat());
        assert_eq!(repeated.scalar_offsets(), Some(0..1_000));
        assert_eq!(repeated.values().len(), 1_000);
        assert_eq!(repeated.null_count(), 0);
        assert_eq!(repeated.scalar_value(), Some(Some(arr.value(0))));

        for i in [0, 1, 999_999_999] {
            assert_eq!(repeated.value_range(i), 0..1_000);
            assert_eq!(repeated.value_length(i), 1_000);
            assert!(repeated.is_valid(i));
        }

        // Slicing it stays free, and keeps the scalar representation.
        let sliced = repeated.clone().sliced(500_000_000, 2);
        assert_eq!(sliced.len(), 2);
        assert!(sliced.offsets_are_scalar());
        assert_eq!(sliced.scalar_offsets(), Some(0..1_000));
        assert_eq!(sliced, arr.new_from_index(0, 2));

        // The repetition of a repeated element is that same element again.
        assert_eq!(sliced.new_from_index(1, 1_000_000_000), repeated);
    }

    #[test]
    fn new_scalar_repeats_one_list() {
        let arr = PlListArray::new_scalar(
            Box::new(PlPrimitiveArray::new_scalar(7i32, 1_000_000_000)),
            1_000_000_000,
        );

        assert_eq!(arr.len(), 1_000_000_000);
        assert!(arr.is_scalar());
        assert_eq!(arr.scalar_offsets(), Some(0..1_000_000_000));
        assert_eq!(arr.value_length(999_999_999), 1_000_000_000);
        assert_eq!(arr.null_count(), 0);

        // A single copy is one element, which the two representations both stand for.
        let one = PlListArray::new_scalar(values(), 1);
        assert!(one.is_scalar());
        assert!(one.is_flat());
        assert_eq!(one.flat_offsets().unwrap().as_slice(), [0, 5]);
        assert_eq!(
            one,
            PlListArray::from_offsets(values(), Buffer::from(vec![0u64, 5])),
        );

        // No copies at all is an empty array, which reads no offset.
        assert!(PlListArray::new_scalar(values(), 0).is_empty());
    }

    #[test]
    fn scalar_offsets_stand_for_the_range_every_element_covers() {
        // Two offsets are valid for any length, and are the range every element covers.
        let arr = PlListArray::new_broadcast(values(), Buffer::from(vec![2u64, 5]), 3, None);

        assert!(arr.offsets_are_scalar());
        assert_eq!(arr.len(), 3);
        assert_eq!(
            arr.values_iter().map(|list| list.len()).collect::<Vec<_>>(),
            [3, 3, 3],
        );
        assert_eq!(elements(&*arr.value(2)), [Some(3), Some(4), Some(5)]);

        // Anything between the two representations is rejected, and so is an empty buffer.
        assert!(PlListArray::try_new(values(), Buffer::from(vec![0u64, 2, 5]), 3, None).is_err());
        assert!(PlListArray::try_new(values(), Buffer::new(), 0, None).is_err());

        // Scalar offsets still have to be ordered and to fit in the values array.
        assert!(PlListArray::try_new(values(), Buffer::from(vec![5u64, 2]), 3, None).is_err());
        assert!(PlListArray::try_new(values(), Buffer::from(vec![0u64, 6]), 3, None).is_err());

        // A scalar array compares equal to the flat one it stands for.
        assert_eq!(
            arr,
            PlListArray::from_offsets(
                Box::new(PlPrimitiveArray::from_vec(vec![
                    3i32, 4, 5, 3, 4, 5, 3, 4, 5
                ])),
                Buffer::from(vec![0u64, 3, 6, 9]),
            ),
            "the same three lists, written out one after the other",
        );
        assert_ne!(
            arr,
            PlListArray::new_broadcast(values(), Buffer::from(vec![2u64, 4]), 3, None),
        );
        assert_ne!(arr, arr.clone().sliced(0, 2));
    }

    #[test]
    fn a_scalar_array_is_never_walked_element_by_element() {
        let values: Box<dyn PlArray> = Box::new(PlPrimitiveArray::new_scalar(7i32, 1_000));
        let arr = PlListArray::new_scalar(values.clone(), usize::MAX);

        // Equality reads the one element the arrays stand for, and never their length.
        assert_eq!(arr, PlListArray::new_scalar(values.clone(), usize::MAX));
        assert_ne!(arr, PlListArray::new_scalar(values, usize::MAX - 1));
        assert_ne!(
            arr,
            PlListArray::new_scalar(
                Box::new(PlPrimitiveArray::new_scalar(7i32, 999)),
                usize::MAX,
            ),
        );

        // A flat mask over more than one element leaves the elements to be compared one by one, so
        // there is no shared element to read.
        let masked = arr
            .sliced(0, 3)
            .with_validity(Some(Bitmap::from_iter([true, false, true])));
        assert!(masked.offsets_are_scalar());
        assert_eq!(masked.scalar_value(), None);
        assert_eq!(masked.null_count(), 1);
        assert_eq!(masked.get(1), None);
        assert_eq!(masked.value_length(1), 1_000);
    }

    #[test]
    fn repeating_a_list_of_scalar_values_does_not_materialize_them() {
        // A single list of a billion sevens, over values that cost `O(1)`.
        let arr = PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::<i32>::new_scalar(7, 1_000_000_000)),
            Buffer::from(vec![0u64, 1_000_000_000]),
        );

        // Nothing here may walk the values: both elements of the result cover the same range of
        // the same scalar values array.
        let repeated = arr.new_from_index(0, 2);
        assert_eq!(repeated.len(), 2);
        assert_eq!(repeated.values().len(), 1_000_000_000);
        assert_eq!(repeated.scalar_offsets(), Some(0..1_000_000_000));
        assert_eq!(repeated.value_length(1), 1_000_000_000);
        assert!(
            repeated
                .values()
                .as_any()
                .downcast_ref::<PlPrimitiveArray<i32>>()
                .unwrap()
                .is_scalar()
        );
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
        assert_eq!(flat, arr);
        assert_eq!(arr, flat);

        // A flat validity mask is carried over as it is.
        let masked = arr
            .clone()
            .with_validity(Some(Bitmap::from_iter([true, false, true])))
            .to_flat();
        assert!(masked.is_flat());
        assert_eq!(masked.null_count(), 1);
        assert_eq!(masked.offsets().as_slice(), [0, 3, 6, 9]);
        assert_eq!(masked.get(1), None);
    }

    #[test]
    fn to_flat_of_an_already_flat_array_only_clones() {
        let arr = arr();
        let flat = arr.to_flat();

        assert_eq!(flat, arr);
        assert!(
            flat.offsets().is_same_buffer(arr.flat_offsets().unwrap()),
            "the offsets must be shared, not written out again",
        );

        // Only the mask is materialized when it is the only scalar buffer.
        let scalar_mask = arr
            .clone()
            .with_validity_broadcast(Some(Bitmap::new_zeroed(1)));
        assert!(!scalar_mask.is_flat());
        let flat = scalar_mask.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.validity().unwrap().len(), 3);
        assert_eq!(flat.null_count(), 3);
        assert!(
            flat.offsets().is_same_buffer(arr.flat_offsets().unwrap()),
            "the offsets were flat already",
        );
        assert_eq!(flat, scalar_mask);
    }

    #[test]
    fn to_flat_writes_out_no_undetermined_list() {
        // Every element is null, so its list is undetermined: it is only the offsets that are
        // written out, and the values array is left untouched.
        let all_null = PlListArray::new_full_null(values(), 4);
        let flat = all_null.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.offsets().as_slice(), [0, 0, 0, 0, 0]);
        assert_eq!(flat.values().len(), 5);
        assert_eq!(flat.null_count(), 4);
        assert_eq!(flat, all_null);

        // The same holds for a scalar array over a range every element covers, once a flat mask
        // makes every one of them null.
        let all_null = PlListArray::new_broadcast(values(), Buffer::from(vec![2u64, 5]), 3, None)
            .with_validity(Some(Bitmap::new_zeroed(3)));
        let flat = all_null.to_flat();

        assert_eq!(flat.offsets().as_slice(), [0, 0, 0, 0]);
        assert_eq!(flat.values().len(), 5);
        assert_eq!(flat, all_null);

        // And for a scalar array of empty lists, which are the same list wherever they point.
        let empty = PlListArray::new_broadcast(values(), Buffer::from(vec![2u64, 2]), 3, None);
        let flat = empty.to_flat();

        assert_eq!(flat.offsets().as_slice(), [0, 0, 0, 0]);
        assert_eq!(flat.values().len(), 5);
        assert_eq!(flat.null_count(), 0);
        assert_eq!(flat, empty);

        // An empty array reaches no element, and keeps the one offset that ends the last list.
        let flat = PlListArray::new_scalar(values(), 0).to_flat();
        assert!(flat.is_empty());
        assert!(flat.is_flat());
        assert_eq!(flat.offsets().as_slice(), [0]);

        // Its mask, which was a single bit for no element at all, is emptied along with it.
        let flat = PlListArray::new_full_null(values(), 0).to_flat();
        assert!(flat.is_empty());
        assert!(flat.is_flat());
        assert_eq!(flat.offsets().as_slice(), [0]);
        assert_eq!(flat.null_count(), 0);
    }

    #[test]
    fn to_flat_of_a_list_of_scalar_values_does_not_materialize_them() {
        // A single list of a billion sevens, repeated three times.
        let arr = PlListArray::new_scalar(
            Box::new(PlPrimitiveArray::<i32>::new_scalar(7, 1_000_000_000)),
            3,
        );
        let flat = arr.to_flat();

        // The lists are written out one after the other, but the values they are taken over are
        // three billion sevens that cost `O(1)`: concatenating copies of a scalar array is scalar.
        assert!(flat.is_flat());
        assert_eq!(
            flat.offsets().as_slice(),
            [0, 1_000_000_000, 2_000_000_000, 3_000_000_000]
        );
        assert_eq!(flat.values().len(), 3_000_000_000);
        assert!(
            flat.values()
                .as_any()
                .downcast_ref::<PlPrimitiveArray<i32>>()
                .unwrap()
                .is_scalar()
        );
        assert_eq!(flat.value_length(2), 1_000_000_000);
    }

    #[test]
    fn as_flat_borrows_an_already_flat_array() {
        let arr = arr();
        let flat = arr.as_flat().expect("the array is flat");

        assert_eq!(*flat, arr);
        assert!(
            flat.offsets().is_same_buffer(arr.flat_offsets().unwrap()),
            "the offsets must be borrowed, not written out again",
        );

        // Neither scalar offsets nor a scalar validity mask can be borrowed as flat.
        assert!(
            PlListArray::new_broadcast(values(), Buffer::from(vec![2u64, 5]), 3, None)
                .as_flat()
                .is_none()
        );
        assert!(
            arr.with_validity_broadcast(Some(Bitmap::new_zeroed(1)))
                .as_flat()
                .is_none()
        );

        // A scalar array of unbounded length is still `O(1)` to reject.
        assert!(
            PlListArray::new_scalar(values(), 1_000_000_000)
                .as_flat()
                .is_none()
        );
    }

    #[test]
    fn broadcasting_one_list() {
        // The single list `[1, 2]`, over values a billion elements long.
        let single = PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::<i32>::new_scalar(7, 1_000_000_000)),
            Buffer::from(vec![0u64, 2]),
        );

        // A billion copies of that list are iterated without the list ever being materialized:
        // every element is the same `O(1)` slice of the same values array.
        let mut iter = single.broadcast_values_iter(1_000_000_000);
        assert_eq!(iter.len(), 1_000_000_000);
        assert_eq!(elements(&*iter.next().unwrap()), [Some(7); 2]);
        assert_eq!(elements(&*iter.nth(999_999_997).unwrap()), [Some(7); 2]);
        assert_eq!(elements(&*iter.next_back().unwrap()), [Some(7); 2]);
        assert!(iter.next().is_none());
        assert_eq!(
            single
                .broadcast_values_iter(1_000_000_000)
                .nth(999_999_999)
                .unwrap()
                .len(),
            2,
        );

        // An array of the length asked for iterates as it is, whatever it is backed by.
        let arr = arr().with_validity(Some(Bitmap::from_iter([true, false, true])));
        let lengths = |iter: PlListValuesIter<'_>| iter.map(|list| list.len()).collect::<Vec<_>>();
        assert_eq!(
            lengths(arr.broadcast_values_iter(3)),
            lengths(arr.values_iter()),
        );
        assert_eq!(lengths(arr.broadcast_values_iter(3)), [2, 0, 3]);

        // A scalar array broadcasts to the length it has like any other.
        let scalar = PlListArray::new_scalar(values(), 3);
        assert_eq!(lengths(scalar.broadcast_values_iter(3)), [5; 3]);

        // Broadcasting to nothing yields nothing, and an empty array broadcasts to nothing
        // else: it has no element to repeat.
        assert_eq!(single.broadcast_values_iter(0).len(), 0);
        assert_eq!(
            PlListArray::new_empty(values())
                .broadcast_values_iter(0)
                .len(),
            0,
        );
    }

    #[test]
    #[should_panic(expected = "an array of length 3 does not broadcast to length 4")]
    fn broadcasting_more_than_one_list_panics() {
        let _ = arr().broadcast_values_iter(4);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn repeating_a_list_out_of_bounds_panics() {
        let _ = arr().new_from_index(3, 1);
    }
}
