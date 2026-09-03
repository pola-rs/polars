use std::ops::Range;

use arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmapRef, validity_eq};
use crate::broadcast::{
    assert_broadcastable, broadcast_index, is_flat_buffer_len, is_flat_offsets_len,
    is_scalar_buffer_len, is_scalar_offsets_len, is_valid_buffer_len,
};
use crate::flat::Flat;

mod builder;
mod flat;
mod iterator;

pub use builder::PlBinaryArrayBuilder;
pub use iterator::{PlBinaryIter, PlBinaryValuesIter};

/// An immutable, cheaply cloneable sequence of `length` optional byte strings over one values
/// buffer.
///
/// This is the offset-based binary array of this crate: it holds a validity mask, the bytes of its
/// elements, and the offsets that cut those bytes into consecutive slices. Element `i` is
/// `values[offsets[i]..offsets[i + 1]]`, so an element costs an offset rather than the view a
/// [`PlBinaryViewArray`](crate::PlBinaryViewArray) spends on it, and the bytes of the array are one
/// buffer rather than many. It carries no logical type: nothing here says the bytes are a string,
/// which lives at a higher level.
///
/// The offsets are always `u64`, and what the separate `length` field buys is what it buys
/// everywhere else in this crate: each backing buffer is either flat or scalar, so an array whose
/// length is unbounded by its memory use is representable. The offsets hold the start of every
/// element plus the end of the last, so it is their *starts* that are flat or scalar:
///
/// * *flat*: `length + 1` offsets, one range per element laid end to end — the end of one byte
///   string is the start of the next, and the last one needs an end of its own.
/// * *scalar*: two offsets, the one range every element covers, which is what lets a single value
///   repeated `length` times cost the memory of that one value.
///
/// Element `i` is read through
/// [`broadcast_index(i, offsets.len() - 1)`](crate::broadcast::broadcast_index), and the validity
/// mask through [`broadcast_index(i, bitmap.len())`](crate::broadcast::broadcast_index), so it is
/// either flat (one bit per element) or scalar (a single bit shared by every element), which is
/// what lets a fully null array carry a one-bit mask. See [`crate::broadcast`] for the full rules.
///
/// The values are never trimmed to what the offsets reach: they may hold bytes before the first
/// offset and after the last, and after slicing they usually do.
///
/// # Example
/// ```
/// use polars_array::PlBinaryArray;
/// use polars_buffer::Buffer;
///
/// // Three byte strings over six bytes: `foo`, `` and `bar`.
/// let arr = PlBinaryArray::from_offsets(
///     Buffer::from(b"foobar".to_vec()),
///     Buffer::from(vec![0u64, 3, 3, 6]),
/// );
/// assert_eq!(arr.len(), 3);
/// assert_eq!(arr.null_count(), 0);
/// assert_eq!(arr.value_length(0), 3);
/// assert_eq!(arr.value_range(2), 3..6);
///
/// // Reading an element slices the values, which is `O(1)`.
/// assert_eq!(arr.value(0), b"foo");
/// assert_eq!(arr.get(1), Some(b"".as_slice()));
/// assert_eq!(arr.get(2), Some(b"bar".as_slice()));
///
/// // A billion copies of one value cost that value: the offsets are the range they all share.
/// let scalar = PlBinaryArray::new_scalar(b"ab", 1_000_000_000);
/// assert_eq!(scalar.len(), 1_000_000_000);
/// assert_eq!(scalar.scalar_offsets(), Some(0..2));
/// assert!(scalar.flat_offsets().is_none());
/// assert_eq!(scalar.value(999_999_999), b"ab");
/// ```
#[derive(Clone)]
pub struct PlBinaryArray {
    values: Buffer<u8>,
    offsets: Buffer<u64>,
    length: usize,
    validity: Option<Bitmap>,
}

impl PlBinaryArray {
    /// Creates a flat [`PlBinaryArray`] out of its internal components.
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
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `offsets` must be monotonically non-decreasing, hold exactly `length + 1` offsets, and end
    /// at an offset that does not exceed the length of `values`; `validity` must hold exactly
    /// `length` bits.
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
            offsets,
            length,
            validity,
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
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `offsets` must be monotonically non-decreasing, hold exactly two offsets — or the single one
    /// that begins no element, if `length` is zero — and end at an offset that does not exceed the
    /// length of `values`; `validity` must hold exactly one bit, or none at all if `length` is
    /// zero.
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
            offsets,
            length,
            validity,
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

    /// Creates a fully valid, flat [`PlBinaryArray`] from `values` and `offsets`, taking its
    /// length from the offsets.
    ///
    /// The offsets are read as flat — `length + 1` of them — so this never builds the scalar
    /// representation: [`Self::new_scalar`] is what does. This function is `O(len)`.
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
    ///
    /// The bytes of the values are laid end to end into the values buffer of the result, so this
    /// is `O(total bytes)`.
    ///
    /// # Example
    /// ```
    /// use polars_array::PlBinaryArray;
    ///
    /// let arr = PlBinaryArray::from_values_iter([b"foo".as_slice(), b"bar"]);
    /// assert_eq!(arr.len(), 2);
    /// assert_eq!(arr.null_count(), 0);
    /// assert_eq!(arr.value(0), b"foo");
    /// assert_eq!(arr.flat_offsets().unwrap().as_slice(), [0, 3, 6]);
    /// ```
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
        // SAFETY: the offsets are the ends of the values appended so far, so they are ordered,
        // there is one per element plus the end of the last, and the last of them is the length of
        // the bytes they were built over.
        unsafe { Self::new_unchecked(Buffer::from(bytes), Buffer::from(offsets), length, None) }
    }

    /// Creates a [`PlBinaryArray`] of `length` copies of `value`, in the memory of that one value.
    ///
    /// Every element covers all of the bytes of `value`, through the two offsets they share. This
    /// function is `O(value.len())`, and so is the result's memory use. Repeating an element of an
    /// array at hand is [`Self::new_from_index`], which copies nothing at all.
    ///
    /// # Example
    /// ```
    /// use polars_array::PlBinaryArray;
    ///
    /// let arr = PlBinaryArray::new_scalar(b"foo", 1_000_000_000);
    /// assert!(arr.is_scalar());
    /// assert_eq!(arr.values().len(), 3);
    /// assert_eq!(arr.value(999_999_999), b"foo");
    /// ```
    #[inline]
    pub fn new_scalar(value: &[u8], length: usize) -> Self {
        let offsets = Buffer::from_owner([0, value.len() as u64]);
        Self {
            values: Buffer::from(value.to_vec()),
            offsets,
            length,
            validity: None,
        }
    }

    /// Creates a [`PlBinaryArray`] of `length` nulls.
    ///
    /// Every element is null, so its value is undetermined; each is given the empty byte string,
    /// which is what keeps both the validity mask and the offsets a single shared slot and leaves
    /// no bytes to write out at all. This function is `O(1)`.
    #[inline]
    pub fn new_full_null(length: usize) -> Self {
        Self {
            values: Buffer::new(),
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

    /// The backing values buffer, holding the bytes the offsets cut the elements out of.
    ///
    /// This is *not* trimmed to what the offsets reach: it may hold bytes before the first offset
    /// and after the last. Read an element of this array with [`Self::value`] instead of indexing
    /// it directly.
    #[inline(always)]
    pub const fn values(&self) -> &Buffer<u8> {
        &self.values
    }

    /// The backing offsets buffer, if it holds the range of every element, laid end to end.
    ///
    /// Element `i` then covers `offsets[i]..offsets[i + 1]` of [`Self::values`], with no
    /// [`broadcast_index`] in the way, and the buffer holds [`Self::len`] `+ 1` offsets — the
    /// start of every element plus the end of the last. This is the `O(1)` counterpart of
    /// [`Self::to_flat`]: it materializes nothing, and returns `None` rather than writing out
    /// scalar offsets. Reach for the range scalar offsets share with [`Self::scalar_offsets`]
    /// instead — between them the two cover every array that has elements at all, so a `None` from
    /// both is an empty array.
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
    /// validity mask be scalar and reports the null the mask makes of those bytes. Returns `None`
    /// for offsets that are flat over more than one element, and for an empty array, which has no
    /// element to share a range. The range of a null element is undetermined (it can be any valid
    /// range).
    #[inline]
    pub fn scalar_offsets(&self) -> Option<Range<usize>> {
        // SAFETY: the array is not empty, so element 0 is in bounds.
        (self.offsets_are_scalar() && self.length > 0)
            .then(|| unsafe { self.value_range_unchecked(0) })
    }

    /// The bytes every element of this array reads, if the offsets hold a single range.
    ///
    /// This is [`Self::scalar_offsets`] as the byte string it cuts out of [`Self::values`].
    #[inline]
    pub fn scalar_values(&self) -> Option<&[u8]> {
        // SAFETY: the range comes from the offsets, so it is in bounds of the values.
        self.scalar_offsets()
            .map(|range| unsafe { self.values.get_unchecked(range) })
    }

    /// Consumes this array into its internal components.
    ///
    /// The offsets are *not* guaranteed to hold [`Self::len`] `+ 1` slots: they are either flat or
    /// scalar, which is why the length comes with them. See [`crate::broadcast`] for how to read
    /// them.
    #[inline]
    pub fn into_inner(self) -> (Buffer<u8>, Buffer<u64>, usize, Option<Bitmap>) {
        (self.values, self.offsets, self.length, self.validity)
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

    /// Whether the offsets hold the single range every element covers, so that every element is
    /// the same byte string.
    ///
    /// An array of one element is both scalar and [`flat`](Self::offsets_are_flat): the two
    /// representations coincide, and this reports them both.
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

    /// Whether this array's offsets hold the range of every element and its mask one bit per
    /// element.
    ///
    /// An array of one element is both flat and [`scalar`](Self::is_scalar).
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.offsets_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array is entirely stored in the scalar representation, and therefore stands
    /// for a single value repeated [`Self::len`] times in the memory of that value alone.
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
        // `i + 1` is in bounds. Every offset is at most the length of the values buffer, and
        // therefore fits in a `usize`.
        unsafe {
            let start = *self.offsets.get_unchecked(i) as usize;
            let end = *self.offsets.get_unchecked(i + 1) as usize;
            start..end
        }
    }

    /// The number of bytes in the element at `i`.
    ///
    /// The length of a null element is undetermined (it can be anything).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value_length(&self, i: usize) -> usize {
        self.value_range(i).len()
    }

    /// The number of bytes in the element at `i`.
    ///
    /// The length of a null element is undetermined (it can be anything).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_length_unchecked(&self, i: usize) -> usize {
        unsafe { self.value_range_unchecked(i) }.len()
    }

    /// Returns the bytes of the element at `i`.
    ///
    /// This function is `O(1)`. The value of a null element is undetermined (it can be any byte
    /// string).
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
    /// string).
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
    /// The values of null elements are undetermined (they can be any byte string).
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
    /// This array either has `length` elements — in which case this is [`Self::values_iter`] — or
    /// a single element, which the `length` values this yields are then all read from.
    /// Broadcasting is `O(1)`, and allocates nothing: the value is repeated as it is read, rather
    /// than materialized into an array to iterate the way [`Self::new_from_index`] would have to.
    /// The values of null elements are undetermined (they can be any byte string).
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

        // The values are left as they are: the offsets are what point into them, and they are not
        // normalized, so the bytes that fall outside the slice simply stop being reachable. Scalar
        // offsets are left alone as well, like a scalar mask: every element of the slice covers
        // the same range every element of this array does.
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

    /// Creates a [`PlBinaryArray`] of `length` copies of the element at `index`.
    ///
    /// This function is `O(1)`: the offsets of the result are scalar, so every one of its elements
    /// covers the range the element covers, of the very same values buffer. A null element repeats
    /// as `length` nulls.
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
    /// This function is `O(1)`.
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
    /// holds one bit per element.
    ///
    /// Materializing scalar offsets is what costs here: flat offsets lay the ranges of the
    /// elements end to end, so the one value every element of a scalar array covers has to be
    /// written out once per element, which is `O(len * value_length)`. It is only the offsets that
    /// are materialized, in `O(len)`, when every element covers no bytes or is null: the value of
    /// a null element is undetermined, so it need not be written out, and the empty byte string is
    /// the same value wherever the offsets point. The result carries its representation in its
    /// type: see [`Flat`] for what a flat array is a proof of.
    ///
    /// # Example
    /// ```
    /// use polars_array::PlBinaryArray;
    ///
    /// // Three copies of `ab`, over the bytes of that one value.
    /// let scalar = PlBinaryArray::new_scalar(b"ab", 3);
    /// assert_eq!(scalar.scalar_offsets(), Some(0..2));
    ///
    /// // Its flat counterpart holds the three values one after the other.
    /// let flat = scalar.to_flat();
    /// assert_eq!(flat.offsets().as_slice(), [0, 2, 4, 6]);
    /// assert_eq!(flat.as_slice(), b"ababab");
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

        // SAFETY: the offsets are ordered, there is one per element plus the end of the last, and
        // the last of them is within the values they were built for: the length of the value
        // repeated once per element, or zero, which every values buffer holds. The mask is the
        // flat counterpart of one that was flat or scalar for this array's length.
        Flat(unsafe { Self::new_unchecked(values, offsets, self.length, validity) })
    }

    /// Borrows this array as a [`Flat`] one, if its offsets already hold the range of every
    /// element and its mask one bit per element.
    ///
    /// This is the `O(1)` counterpart of [`Self::to_flat`]: it materializes nothing, and returns
    /// `None` rather than writing out a scalar buffer when this array is not
    /// [`flat`](Self::is_flat).
    ///
    /// # Example
    /// ```
    /// use polars_array::PlBinaryArray;
    ///
    /// let arr = PlBinaryArray::from_values_iter([b"foo".as_slice(), b"bar"]);
    /// assert_eq!(arr.as_flat().unwrap().as_slice(), b"foobar");
    ///
    /// // A billion copies of one value share two offsets, so they have to be written out.
    /// let scalar = PlBinaryArray::new_scalar(b"ab", 1_000_000_000);
    /// assert!(scalar.as_flat().is_none());
    /// ```
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: the offsets of a flat array hold the range of every element, and its mask one
        // bit per element.
        self.is_flat()
            .then(|| unsafe { Flat::new_ref(self) })
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
        // SAFETY: the offsets are the ends of the values appended so far, so they are ordered,
        // there is one per element plus the end of the last, and the last of them is the length of
        // the bytes they were built over. The mask holds one bit per element.
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

/// Compares two arrays element-wise; neither the offsets nor the bytes behind the offsets of null
/// elements are part of a value, so an array compares equal to any other one holding the same byte
/// strings.
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

/// Compares an array of unknown representation against a flat one; see
/// [`PartialEq<PlBinaryArray> for Flat<PlBinaryArray>`](Flat).
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
///
/// This is the half of the validation that both families of constructors share; how many offsets
/// there are is what tells the flat representation from the scalar one, and is checked separately.
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
    use crate::PlBinaryViewArray;

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
    fn the_offsets_are_reached_through_their_representation() {
        let arr = arr();

        assert_eq!(
            arr.flat_offsets().map(Buffer::as_slice),
            Some([0, 3, 3, 6].as_slice()),
        );
        assert_eq!(arr.scalar_offsets(), None);
        assert_eq!(arr.scalar_values(), None);

        // Scalar offsets hand out no flat buffer; it is the one range they hold that is reached.
        let arr = PlBinaryArray::new_scalar(b"ab", 1_000_000_000);

        assert_eq!(arr.flat_offsets(), None);
        assert_eq!(arr.scalar_offsets(), Some(0..2));
        assert_eq!(arr.scalar_values(), Some(b"ab".as_slice()));

        // The offsets are read whether or not the mask makes the elements null.
        let arr = arr.sliced(0, 3).with_validity(Some(Bitmap::new_zeroed(3)));

        assert_eq!(arr.scalar_offsets(), Some(0..2));
        assert!(
            arr.scalar_value().is_none(),
            "a flat mask leaves no shared element",
        );

        // An empty array has no range to share, and is flat unless it is backed by a stray one.
        assert_eq!(
            PlBinaryArray::new_empty()
                .flat_offsets()
                .map(Buffer::as_slice),
            Some([0].as_slice()),
        );

        let empty = PlBinaryArray::new_scalar(b"ab", 0);
        assert_eq!(empty.flat_offsets(), None);
        assert_eq!(empty.scalar_offsets(), None);
        assert_eq!(empty.scalar_values(), None);
    }

    #[test]
    fn the_bytes_outside_the_offsets_are_unreachable() {
        // The first element starts after the first byte and the last ends before the last one.
        let arr = PlBinaryArray::from_offsets(
            Buffer::from(b"foobar".to_vec()),
            Buffer::from(vec![1u64, 4]),
        );

        assert_eq!(arr.len(), 1);
        assert_eq!(arr.values().len(), 6);
        assert_eq!(arr.value(0), b"oob");
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
    fn scalar_validity_over_flat_offsets() {
        let arr = arr().with_validity_broadcast(Some(Bitmap::new_zeroed(1)));

        assert!(arr.validity_is_scalar());
        assert!(!arr.is_flat());
        assert_eq!(arr.null_count(), 3);
        assert!(arr.iter().all(|element| element.is_none()));

        // The offsets are untouched, so the bytes of the null elements are still there.
        assert_eq!(arr.value(0), b"foo");
        assert_eq!(
            arr.values_iter().collect::<Vec<_>>(),
            [b"foo".as_slice(), b"", b"bar"],
        );
    }

    #[test]
    fn flat_validity() {
        let arr = PlBinaryArray::new(
            Buffer::from(b"foobar".to_vec()),
            Buffer::from(vec![0u64, 3, 3, 6]),
            3,
            Some(Bitmap::from_iter([true, false, true])),
        );

        assert!(!arr.validity_is_scalar());
        assert!(arr.is_flat());
        assert_eq!(arr.null_count(), 1);
        assert!(arr.is_null(1));
        assert_eq!(
            arr.iter().collect::<Vec<_>>(),
            [Some(b"foo".as_slice()), None, Some(b"bar")],
        );
    }

    #[test]
    fn try_new_rejects_invalid_components() {
        let values = || Buffer::from(b"foobar".to_vec());

        // The offsets must hold one slot per element plus the end of the last.
        assert!(PlBinaryArray::try_new(values(), Buffer::from(vec![0u64, 3, 6]), 3, None).is_err());
        assert!(
            PlBinaryArray::try_new(values(), Buffer::from(vec![0u64, 3, 3, 6]), 3, None).is_ok()
        );

        // Two offsets are a scalar array rather than a flat one, and are never inferred to be
        // either — see `scalar_offsets_stand_for_the_range_every_element_covers`.
        assert!(PlBinaryArray::try_new(values(), Buffer::from(vec![0u64, 3]), 3, None).is_err());

        // They must be monotonically non-decreasing.
        assert!(
            PlBinaryArray::try_new(values(), Buffer::from(vec![0u64, 6, 3, 6]), 3, None).is_err()
        );

        // The last of them must not reach past the values.
        assert!(PlBinaryArray::try_new(values(), Buffer::from(vec![0u64, 7]), 1, None).is_err());
        assert!(PlBinaryArray::try_new(values(), Buffer::from(vec![0u64, 6]), 1, None).is_ok());

        // The validity mask has to be flat as well.
        let offsets = Buffer::from(vec![0u64, 3, 3, 6]);
        assert!(
            PlBinaryArray::try_new(values(), offsets.clone(), 3, Some(Bitmap::new_zeroed(2)))
                .is_err()
        );
        assert!(
            PlBinaryArray::try_new(values(), offsets.clone(), 3, Some(Bitmap::new_zeroed(1)))
                .is_err()
        );
        assert!(PlBinaryArray::try_new(values(), offsets, 3, Some(Bitmap::new_zeroed(3))).is_ok());
    }

    #[test]
    fn try_new_broadcast_requires_scalar_offsets() {
        let values = || Buffer::from(b"foobar".to_vec());
        let scalar = || Buffer::from(vec![0u64, 3]);

        // The offsets must hold the one range every element covers.
        assert!(PlBinaryArray::try_new_broadcast(values(), scalar(), 1_000_000, None).is_ok());
        assert!(
            PlBinaryArray::try_new_broadcast(values(), Buffer::from(vec![0u64, 3, 3, 6]), 3, None)
                .is_err()
        );

        // They are checked exactly as the flat ones are.
        assert!(
            PlBinaryArray::try_new_broadcast(values(), Buffer::from(vec![6u64, 3]), 3, None)
                .is_err()
        );
        assert!(
            PlBinaryArray::try_new_broadcast(values(), Buffer::from(vec![0u64, 7]), 3, None)
                .is_err()
        );

        // The validity mask has to be scalar as well.
        assert!(
            PlBinaryArray::try_new_broadcast(values(), scalar(), 3, Some(Bitmap::new_zeroed(3)))
                .is_err()
        );
        assert!(
            PlBinaryArray::try_new_broadcast(values(), scalar(), 3, Some(Bitmap::new_zeroed(1)))
                .is_ok()
        );

        // An array of no elements covers no range, so the single offset that begins no element
        // stands for it as well.
        assert!(
            PlBinaryArray::try_new_broadcast(values(), Buffer::from(vec![0u64]), 0, None).is_ok()
        );
        assert!(PlBinaryArray::try_new_broadcast(values(), scalar(), 0, None).is_ok());
        assert!(PlBinaryArray::try_new_broadcast(values(), Buffer::new(), 0, None).is_err());
    }

    #[test]
    fn slicing_only_slices_the_offsets() {
        let arr = PlBinaryArray::new(
            Buffer::from(b"foobar".to_vec()),
            Buffer::from(vec![0u64, 3, 3, 6]),
            3,
            Some(Bitmap::from_iter([true, false, true])),
        )
        .sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.flat_offsets().unwrap().as_slice(), [3, 3, 6]);
        assert_eq!(arr.validity().unwrap().flat_bitmap().unwrap().len(), 2);
        assert_eq!(arr.null_count(), 1);

        // The values are left alone; the offsets are what point into them.
        assert_eq!(arr.values().len(), 6);
        assert_eq!(arr.value(1), b"bar");

        // Slicing away every element leaves the end of the last value behind.
        let arr = arr.sliced(2, 0);
        assert!(arr.is_empty());
        assert_eq!(arr.flat_offsets().unwrap().as_slice(), [6]);
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
    fn equality_ignores_the_offsets_and_the_bytes_behind_them() {
        // The same three byte strings, laid out over different bytes with different offsets.
        let other = PlBinaryArray::from_offsets(
            Buffer::from(b"xfoobary".to_vec()),
            Buffer::from(vec![1u64, 4, 4, 7]),
        );

        assert_eq!(arr(), other);
        assert_eq!(arr(), arr());
        assert_ne!(arr(), arr().sliced(0, 2));

        // Same bytes, differently cut into elements.
        assert_ne!(
            arr(),
            PlBinaryArray::from_offsets(
                Buffer::from(b"foobar".to_vec()),
                Buffer::from(vec![0u64, 3, 4, 6]),
            ),
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
    }

    #[test]
    fn equality_ignores_the_values_of_null_elements() {
        // The byte strings `foo`, `bar` and `baz`, of which the second is null.
        let mask = Bitmap::from_iter([true, false, true]);
        let offsets = Buffer::from(vec![0u64, 3, 6, 9]);
        let lhs = PlBinaryArray::new(
            Buffer::from(b"foobarbaz".to_vec()),
            offsets.clone(),
            3,
            Some(mask.clone()),
        );

        // The second element holds different bytes, but it is null on both sides.
        let rhs = PlBinaryArray::new(
            Buffer::from(b"fooxxxbaz".to_vec()),
            offsets.clone(),
            3,
            Some(mask),
        );
        assert_eq!(lhs, rhs);

        // A byte of a valid element still counts.
        let other = PlBinaryArray::new(
            Buffer::from(b"fooxxxbay".to_vec()),
            offsets,
            3,
            Some(Bitmap::from_iter([true, false, true])),
        );
        assert_ne!(lhs, other);
    }

    #[test]
    fn equality_of_fully_null_arrays_ignores_their_values() {
        let null = PlBinaryArray::new_full_null(3);

        assert_eq!(null, null.clone());
        assert_eq!(
            null,
            arr().with_validity_broadcast(Some(Bitmap::new_zeroed(1))),
            "every element is null on both sides, so no value is determined",
        );
        assert_ne!(null, arr());
        assert_ne!(null, PlBinaryArray::new_full_null(4));
    }

    #[test]
    fn empty() {
        let arr = PlBinaryArray::new_empty();

        assert!(arr.is_empty());
        assert!(arr.is_flat());
        assert_eq!(arr.null_count(), 0);
        assert_eq!(arr.flat_offsets().unwrap().as_slice(), [0]);
        assert_eq!(arr.iter().next(), None);
        assert_eq!(arr, PlBinaryArray::default());
        assert_eq!(arr, PlBinaryArray::from_values_iter::<&[u8], _>([]));
    }

    #[test]
    #[should_panic(expected = "at least one offset")]
    fn an_array_without_offsets_has_no_length_to_take() {
        let _ = PlBinaryArray::from_offsets(Buffer::new(), Buffer::new());
    }

    #[test]
    fn collecting_the_values_and_the_optional_values() {
        let values = PlBinaryArray::from_values_iter([b"foo".as_slice(), b"", b"bar"]);

        assert_eq!(values, arr());
        assert!(values.is_flat());
        assert_eq!(values.null_count(), 0);

        let options: PlBinaryArray = [Some(b"foo".as_slice()), None, Some(b"bar")]
            .into_iter()
            .collect();

        assert!(options.is_flat());
        assert_eq!(options.null_count(), 1);
        assert_eq!(
            options.iter().collect::<Vec<_>>(),
            [Some(b"foo".as_slice()), None, Some(b"bar")],
        );
        // The bytes of the null element are never written out.
        assert_eq!(options.values().as_slice(), b"foobar");
        assert_eq!(options.flat_offsets().unwrap().as_slice(), [0, 3, 3, 6]);

        assert!(
            std::iter::empty::<Option<&[u8]>>()
                .collect::<PlBinaryArray>()
                .is_empty()
        );
    }

    #[test]
    fn iterators_are_exact_sized() {
        let arr = arr();

        assert_eq!(arr.iter().len(), 3);
        assert_eq!(arr.values_iter().len(), 3);
        assert_eq!(arr.iter().size_hint(), (3, Some(3)));
        assert_eq!(
            arr.values_iter().rev().collect::<Vec<_>>(),
            [b"bar".as_slice(), b"", b"foo"],
        );
        assert_eq!(
            (&arr).into_iter().collect::<Vec<_>>(),
            [Some(b"foo".as_slice()), Some(b""), Some(b"bar")],
        );
    }

    #[test]
    fn into_inner_returns_the_components() {
        let (values, offsets, length, validity) = PlBinaryArray::new_full_null(3).into_inner();

        assert!(values.is_empty());
        assert_eq!(offsets.as_slice(), [0, 0]);
        assert_eq!(length, 3);
        assert_eq!(validity, Some(Bitmap::new_zeroed(1)));
    }

    #[test]
    fn debug_lists_the_offsets_and_the_values() {
        assert_eq!(
            format!("{:?}", arr()),
            "PlBinaryArray { length: 3, offsets: [0, 3, 3, 6], \
             values: [102, 111, 111, 98, 97, 114] }",
        );

        // Neither a scalar validity mask nor scalar offsets are materialized: they are listed as
        // they are backed, which is the two of them every element of a billion shares.
        let arr = PlBinaryArray::new_scalar(b"ab", 1_000_000_000)
            .with_validity_broadcast(Some(Bitmap::new_zeroed(1)));
        assert_eq!(
            format!("{arr:?}"),
            "PlBinaryArray { length: 1000000000, validity: PlBitmapRef[false; 1000000000], \
             offsets: [0, 2], values: [97, 98] }",
        );
    }

    #[test]
    fn behind_the_trait_object() {
        let arr: Box<dyn PlArray> = Box::new(arr());

        assert_eq!(arr.array_type(), PlArrayType::Binary);
        assert!(arr.array_type().is_binary());
        assert!(!arr.array_type().is_binary_view());
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
                .downcast_ref::<PlBinaryArray>()
                .unwrap()
                .value(1),
            b"bar",
        );

        // An offset-based binary array is not a binary view array, however its bytes are reached.
        let view: Box<dyn PlArray> = Box::new(PlBinaryViewArray::from_values_iter([
            b"foo".as_slice(),
            b"",
            b"bar",
        ]));
        assert_ne!(&arr, &view);
    }

    #[test]
    fn new_from_index_repeats_one_value() {
        let arr = arr();

        // Nothing is copied: every element of the result covers the range the element covers, of
        // the very same values buffer.
        let repeated = arr.new_from_index(2, 3);
        assert_eq!(repeated.len(), 3);
        assert!(repeated.is_scalar());
        assert_eq!(repeated.scalar_offsets(), Some(3..6));
        assert_eq!(repeated.values().len(), 6);
        assert_eq!(repeated.null_count(), 0);
        assert!(repeated.values_iter().all(|value| value == b"bar"));

        // An empty element repeats as the empty range it covers.
        let repeated = arr.new_from_index(1, 4);
        assert_eq!(repeated.scalar_offsets(), Some(3..3));
        assert_eq!(repeated.value(3), b"");

        // A null element repeats as nulls, under a mask of a single bit, over no bytes at all.
        let nulls = arr
            .clone()
            .with_validity(Some(Bitmap::from_iter([false, true, true])))
            .new_from_index(0, 4);
        assert_eq!(nulls.null_count(), 4);
        assert!(nulls.is_scalar());
        assert!(nulls.values().is_empty());
        assert_eq!(nulls.get(3), None);

        assert!(arr.new_from_index(0, 0).is_empty());

        // The representation is not part of a value: the repetition of `foo` compares equal to the
        // flat array of the two values it stands for.
        assert_eq!(
            unsafe { arr.new_from_index_unchecked(0, 2) },
            PlBinaryArray::from_values_iter([b"foo".as_slice(), b"foo"]),
        );
    }

    #[test]
    fn a_repeated_value_costs_that_one_value() {
        // A single value of a thousand bytes, repeated a billion times.
        let arr = PlBinaryArray::from_values_iter([vec![7u8; 1_000]]);
        let repeated = arr.new_from_index(0, 1_000_000_000);

        // Neither the bytes nor the offsets are written out: the offsets are the one range every
        // element covers, of the values buffer the element was already taken over.
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
    fn new_scalar_repeats_one_value() {
        let arr = PlBinaryArray::new_scalar(b"ab", 1_000_000_000);

        assert_eq!(arr.len(), 1_000_000_000);
        assert!(arr.is_scalar());
        assert_eq!(arr.scalar_offsets(), Some(0..2));
        assert_eq!(arr.values().len(), 2);
        assert_eq!(arr.value_length(999_999_999), 2);
        assert_eq!(arr.null_count(), 0);

        // A single copy is one element, which the two representations both stand for.
        let one = PlBinaryArray::new_scalar(b"ab", 1);
        assert!(one.is_scalar());
        assert!(one.is_flat());
        assert_eq!(one.flat_offsets().unwrap().as_slice(), [0, 2]);
        assert_eq!(one, PlBinaryArray::from_values_iter([b"ab".as_slice()]));

        // No copies at all is an empty array, which reads no offset.
        assert!(PlBinaryArray::new_scalar(b"ab", 0).is_empty());

        // A value of no bytes at all is a value like any other.
        let empty = PlBinaryArray::new_scalar(b"", 3);
        assert!(empty.is_scalar());
        assert_eq!(empty.scalar_offsets(), Some(0..0));
        assert_eq!(empty.value(2), b"");
    }

    #[test]
    fn scalar_offsets_stand_for_the_range_every_element_covers() {
        // Two offsets are valid for any length, and are the range every element covers.
        let arr = PlBinaryArray::new_broadcast(
            Buffer::from(b"foobar".to_vec()),
            Buffer::from(vec![3u64, 6]),
            3,
            None,
        );

        assert!(arr.offsets_are_scalar());
        assert_eq!(arr.len(), 3);
        assert_eq!(
            arr.values_iter().collect::<Vec<_>>(),
            [b"bar".as_slice(); 3],
        );
        assert_eq!(arr.value(2), b"bar");

        // Anything between the two representations is rejected, and so is an empty buffer.
        assert!(
            PlBinaryArray::try_new(
                Buffer::from(b"foobar".to_vec()),
                Buffer::from(vec![0u64, 3, 6]),
                3,
                None,
            )
            .is_err()
        );
        assert!(PlBinaryArray::try_new(Buffer::new(), Buffer::new(), 0, None).is_err());

        // A scalar array compares equal to the flat one it stands for.
        assert_eq!(
            arr,
            PlBinaryArray::from_values_iter([b"bar".as_slice(), b"bar", b"bar"]),
            "the same three values, written out one after the other",
        );
        assert_ne!(arr, arr.clone().sliced(0, 2));
    }

    #[test]
    fn a_scalar_array_is_never_walked_element_by_element() {
        let arr = PlBinaryArray::new_scalar(b"ab", usize::MAX);

        // Equality reads the one element the arrays stand for, and never their length.
        assert_eq!(arr, PlBinaryArray::new_scalar(b"ab", usize::MAX));
        assert_ne!(arr, PlBinaryArray::new_scalar(b"ab", usize::MAX - 1));
        assert_ne!(arr, PlBinaryArray::new_scalar(b"ba", usize::MAX));

        // A flat mask over more than one element leaves the elements to be compared one by one, so
        // there is no shared element to read.
        let masked = arr
            .sliced(0, 3)
            .with_validity(Some(Bitmap::from_iter([true, false, true])));
        assert!(masked.offsets_are_scalar());
        assert_eq!(masked.scalar_value(), None);
        assert_eq!(masked.null_count(), 1);
        assert_eq!(masked.get(1), None);
        assert_eq!(masked.value(1), b"ab");
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
        assert!(
            flat.values().is_same_buffer(arr.values()),
            "and so must the values",
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
    fn to_flat_writes_out_no_undetermined_value() {
        // Every element is null, so its value is undetermined: it is only the offsets that are
        // written out, and there are no bytes to leave behind either.
        let all_null = PlBinaryArray::new_full_null(4);
        let flat = all_null.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.offsets().as_slice(), [0, 0, 0, 0, 0]);
        assert!(flat.as_slice().is_empty());
        assert_eq!(flat.null_count(), 4);
        assert_eq!(flat, all_null);

        // The same holds for a scalar array over a range every element covers, once a flat mask
        // makes every one of them null.
        let all_null =
            PlBinaryArray::new_scalar(b"ab", 3).with_validity(Some(Bitmap::new_zeroed(3)));
        let flat = all_null.to_flat();

        assert_eq!(flat.offsets().as_slice(), [0, 0, 0, 0]);
        assert_eq!(flat.values().len(), 2, "the bytes are left as they were");
        assert_eq!(flat, all_null);

        // And for a scalar array of empty values, which are the same value wherever they point.
        let empty = PlBinaryArray::new_broadcast(
            Buffer::from(b"foobar".to_vec()),
            Buffer::from(vec![3u64, 3]),
            3,
            None,
        );
        let flat = empty.to_flat();

        assert_eq!(flat.offsets().as_slice(), [0, 0, 0, 0]);
        assert_eq!(flat.null_count(), 0);
        assert_eq!(flat, empty);

        // An empty array reaches no element, and keeps the one offset that ends the last value.
        let flat = PlBinaryArray::new_scalar(b"ab", 0).to_flat();
        assert!(flat.is_empty());
        assert!(flat.is_flat());
        assert_eq!(flat.offsets().as_slice(), [0]);

        // Its mask, which was a single bit for no element at all, is emptied along with it.
        let flat = PlBinaryArray::new_full_null(0).to_flat();
        assert!(flat.is_empty());
        assert!(flat.is_flat());
        assert_eq!(flat.offsets().as_slice(), [0]);
        assert_eq!(flat.null_count(), 0);
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
        assert!(PlBinaryArray::new_scalar(b"ab", 3).as_flat().is_none());
        assert!(
            arr.with_validity_broadcast(Some(Bitmap::new_zeroed(1)))
                .as_flat()
                .is_none()
        );

        // A scalar array of unbounded length is still `O(1)` to reject.
        assert!(
            PlBinaryArray::new_scalar(b"ab", 1_000_000_000)
                .as_flat()
                .is_none()
        );

        // One element is both flat and scalar, so it is borrowed rather than written out.
        assert!(PlBinaryArray::new_scalar(b"ab", 1).as_flat().is_some());
    }

    #[test]
    fn broadcasting_one_value() {
        // A single value of two bytes, over a values buffer that holds more.
        let single = PlBinaryArray::from_offsets(
            Buffer::from(b"foobar".to_vec()),
            Buffer::from(vec![0u64, 2]),
        );

        // A billion copies of it are iterated without it ever being materialized: every element is
        // the same `O(1)` slice of the same values buffer.
        let mut iter = single.broadcast_values_iter(1_000_000_000);
        assert_eq!(iter.len(), 1_000_000_000);
        assert_eq!(iter.next(), Some(b"fo".as_slice()));
        assert_eq!(iter.nth(999_999_997), Some(b"fo".as_slice()));
        assert_eq!(iter.next_back(), Some(b"fo".as_slice()));
        assert!(iter.next().is_none());
        assert_eq!(
            single.broadcast_values_iter(1_000_000_000).nth(999_999_999),
            Some(b"fo".as_slice()),
        );

        // An array of the length asked for iterates as it is, whatever it is backed by.
        let arr = arr().with_validity(Some(Bitmap::from_iter([true, false, true])));
        assert_eq!(
            arr.broadcast_values_iter(3).collect::<Vec<_>>(),
            arr.values_iter().collect::<Vec<_>>(),
        );
        assert_eq!(
            arr.broadcast_values_iter(3).collect::<Vec<_>>(),
            [b"foo".as_slice(), b"", b"bar"],
        );

        // A scalar array broadcasts to the length it has like any other.
        let scalar = PlBinaryArray::new_scalar(b"ab", 3);
        assert_eq!(
            scalar.broadcast_values_iter(3).collect::<Vec<_>>(),
            [b"ab".as_slice(); 3],
        );

        // Broadcasting to nothing yields nothing, and an empty array broadcasts to nothing else:
        // it has no element to repeat.
        assert_eq!(single.broadcast_values_iter(0).len(), 0);
        assert_eq!(PlBinaryArray::new_empty().broadcast_values_iter(0).len(), 0);
    }

    #[test]
    #[should_panic(expected = "an array of length 3 does not broadcast to length 4")]
    fn broadcasting_more_than_one_value_panics() {
        let _ = arr().broadcast_values_iter(4);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn repeating_a_value_out_of_bounds_panics() {
        let _ = arr().new_from_index(3, 1);
    }
}
