use std::ops::Range;

use arrow::bitmap::Bitmap;
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmapRef, validity_eq};
use crate::broadcast::{assert_broadcastable, is_valid_buffer_len, is_valid_fixed_size_values_len};
use crate::concatenate::concatenate_repeated;
use crate::flat::Flat;

mod iterator;

pub use iterator::{PlFixedSizeListIter, PlFixedSizeListValuesIter};

/// An immutable, cheaply cloneable sequence of `length` optional lists of `width` values each,
/// over one values array.
///
/// This is the fixed-length nested array of this crate: it holds no values of its own, only a
/// validity mask, the width every one of its lists has, and the values array those lists cut into
/// consecutive slices. Element `i` is `values[i * width..(i + 1) * width]`, so there are no offsets
/// to store — the width is what a [`PlListArray`](crate::PlListArray) spends a buffer on. It
/// carries no logical type: the values array is a [`PlArray`], and what a caller thinks of as the
/// list's inner type lives at a higher level.
///
/// What the separate `length` field buys is what it buys everywhere else in this crate: each
/// backing buffer is either flat or scalar, so an array whose length is unbounded by its memory
/// use is representable. It is the *elements* the values array holds that are flat or scalar, and
/// each of them is `width` values wide:
///
/// * *flat*: `length * width` values, the values of every element laid end to end.
/// * *scalar*: `width` values, the one element every element covers, which is what lets a single
///   list repeated `length` times cost the memory of that one list.
///
/// An array of one element is both, as is one of width zero, whose elements hold no values at all;
/// an empty array is flat, since it has no element for a scalar values array to stand for. The
/// validity mask is flat (one bit per element) or scalar (a single bit shared by every element)
/// like every other mask in this crate, which is what lets a fully null array carry a one-bit
/// mask. See [`crate::broadcast`] for the full rules.
///
/// Unlike the values of a [`PlListArray`](crate::PlListArray), these are always trimmed to exactly
/// what the elements reach: there are no offsets to leave anything outside them, so slicing slices
/// the values as well.
///
/// # Example
/// ```
/// use polars_array::{PlArray, PlFixedSizeListArray, PlPrimitiveArray};
///
/// // Three lists of two values: `[1, 2]`, `[3, 4]` and `[5, 6]`.
/// let arr = PlFixedSizeListArray::from_values(
///     Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5, 6])),
///     2,
/// );
/// assert_eq!(arr.len(), 3);
/// assert_eq!(arr.width(), 2);
/// assert_eq!(arr.null_count(), 0);
/// assert_eq!(arr.value_range(2), 4..6);
///
/// // Reading an element slices the values array, which is `O(1)`.
/// let element = arr.value(2);
/// assert_eq!(element.len(), 2);
/// assert_eq!(
///     element
///         .as_any()
///         .downcast_ref::<PlPrimitiveArray<i32>>()
///         .unwrap()
///         .values()
///         .as_slice(),
///     [5, 6],
/// );
///
/// // A billion copies of one list cost that list: the values are the element they all share.
/// let scalar = PlFixedSizeListArray::new_scalar(
///     Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
///     1_000_000_000,
/// );
/// assert_eq!(scalar.len(), 1_000_000_000);
/// assert_eq!(scalar.values().len(), 2);
/// assert!(scalar.is_scalar());
/// assert_eq!(scalar.value_range(999_999_999), 0..2);
/// ```
#[derive(Clone)]
pub struct PlFixedSizeListArray {
    values: Box<dyn PlArray>,
    width: usize,
    length: usize,
    validity: Option<Bitmap>,
}

impl PlFixedSizeListArray {
    /// Creates a [`PlFixedSizeListArray`] out of its internal components.
    ///
    /// This function is `O(1)`: there are no offsets to walk, only the length of `values` to check
    /// against the width.
    ///
    /// # Errors
    /// This function errors if `values` is neither flat (`length * width` values) nor scalar (the
    /// `width` values of the one element every element covers, which an empty array has no element
    /// to share), or if `validity` is neither flat (length equal to `length`) nor scalar (length
    /// one).
    pub fn try_new(
        values: Box<dyn PlArray>,
        width: usize,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_valid_fixed_size_values_len(values.len(), width, length),
            ComputeError:
            "values array of length {} is neither flat nor scalar for a fixed size list array of \
             length {} and width {}: it needs the width of every element laid end to end, or the \
             width of the one element every element covers",
            values.len(), length, width,
        );

        if let Some(validity) = validity.as_ref() {
            polars_ensure!(
                is_valid_buffer_len(validity.len(), length),
                ComputeError:
                "validity mask of length {} is neither flat nor scalar for an array of length {}",
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

    /// Creates a [`PlFixedSizeListArray`] out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(
        values: Box<dyn PlArray>,
        width: usize,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        Self::try_new(values, width, length, validity).unwrap()
    }

    /// Creates a [`PlFixedSizeListArray`] out of its internal components without validating them.
    ///
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `values` must be either flat (`length * width` values) or scalar (`width` values, of which
    /// there is only an element to make sense when `length` is not zero); `validity` must be
    /// either flat (length equal to `length`) or scalar (length one).
    #[inline]
    pub unsafe fn new_unchecked(
        values: Box<dyn PlArray>,
        width: usize,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        if cfg!(debug_assertions) {
            assert!(is_valid_fixed_size_values_len(values.len(), width, length));
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_valid_buffer_len(v.len(), length))
            );
        }

        Self {
            values,
            width,
            length,
            validity,
        }
    }

    /// Creates an empty [`PlFixedSizeListArray`] of lists `width` values wide.
    ///
    /// The values array is what determines the type of the lists, of which there are none: it is
    /// kept as the type it is, sliced away to the empty values an array without elements has.
    #[inline]
    pub fn new_empty(values: Box<dyn PlArray>, width: usize) -> Self {
        Self {
            values: values.sliced(0, 0),
            width,
            length: 0,
            validity: None,
        }
    }

    /// Creates a fully valid, flat [`PlFixedSizeListArray`] by cutting `values` into lists of
    /// `width` values, taking its length from how many of them there are.
    ///
    /// The values are read as flat — `length * width` of them — so this never builds the scalar
    /// representation: [`Self::new_scalar`] is what does. This function is `O(1)`.
    ///
    /// # Panics
    /// Panics if `width` is zero, which leaves no number of lists to cut the values into, or if
    /// the length of `values` is not a multiple of `width`.
    pub fn from_values(values: Box<dyn PlArray>, width: usize) -> Self {
        assert!(
            width > 0,
            "the length of a fixed size list array of width zero cannot be taken from its values",
        );
        assert!(
            values.len().is_multiple_of(width),
            "the values of length {} do not divide into lists of width {}",
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

    /// Creates a [`PlFixedSizeListArray`] of `length` copies of the list `element`, in the memory
    /// of that one list.
    ///
    /// The list is given as the array of its values, whose length is the width of the result:
    /// every element covers all of it. This function is `O(1)`, and so is the result's memory use
    /// on top of `element`. Repeating a list that is already an element of a fixed size list array
    /// is [`Self::new_from_index`].
    #[inline]
    pub fn new_scalar(element: Box<dyn PlArray>, length: usize) -> Self {
        let width = element.len();

        // There is no element for the values to be shared by when there are no elements at all,
        // which is why an empty array is the one that keeps nothing of the list it repeats.
        let values = if length == 0 {
            element.sliced(0, 0)
        } else {
            element
        };

        Self {
            values,
            width,
            length,
            validity: None,
        }
    }

    /// Creates a [`PlFixedSizeListArray`] of `length` nulls whose lists are as wide as `element`.
    ///
    /// Every element is null, so its value is undetermined; each is given `element`, which is what
    /// keeps both the validity mask and the values a single shared slot. This function is `O(1)`.
    #[inline]
    pub fn new_full_null(element: Box<dyn PlArray>, length: usize) -> Self {
        Self {
            validity: Some(Bitmap::new_zeroed(1)),
            ..Self::new_scalar(element, length)
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

    /// The number of values in every element.
    ///
    /// This is as much a part of the array as its length: an element of a null list is as wide as
    /// any other, since it is the mask and not the width that makes it null.
    #[inline(always)]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// The values array the lists are taken over.
    ///
    /// This is *not* guaranteed to hold [`Self::len`] `*` [`Self::width`] values: it is either
    /// flat or scalar. Read an element of this array with [`Self::value`] instead of indexing it
    /// directly.
    #[inline]
    pub fn values(&self) -> &dyn PlArray {
        &*self.values
    }

    /// Consumes this array into its internal components.
    #[inline]
    pub fn into_inner(self) -> (Box<dyn PlArray>, usize, usize, Option<Bitmap>) {
        (self.values, self.width, self.length, self.validity)
    }

    /// The validity mask, if any element may be null.
    ///
    /// The returned [`PlBitmapRef`] has [`Self::len`] bits regardless of whether the backing bitmap
    /// is flat or scalar, so reading validity through it needs no knowledge of which representation
    /// this array is in. This mask says nothing about the values: a valid list may still hold null
    /// values.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_unchecked(validity, self.length) })
    }

    /// Whether the values hold the single element every element of this array covers, so that
    /// every element is the same list.
    ///
    /// An array of one element is both scalar and [`flat`](Self::values_are_flat): the two
    /// representations coincide, and this reports them both. So is an array of width zero, whose
    /// elements hold no values to lay out either way.
    #[inline]
    pub fn values_are_scalar(&self) -> bool {
        self.values.len() == self.width && self.length >= 1
    }

    /// Whether the values hold the values of every element, laid end to end.
    ///
    /// An array of one element is both flat and [`scalar`](Self::values_are_scalar).
    #[inline]
    pub fn values_are_flat(&self) -> bool {
        // A length times a width that overflows a `usize` is longer than any values array can be,
        // so such an array is never flat.
        self.length.checked_mul(self.width) == Some(self.values.len())
    }

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether this array's values hold the values of every element and its mask one bit per
    /// element.
    ///
    /// The values array carries its own representation, which this says nothing about. An array of
    /// one element is both flat and [`scalar`](Self::is_scalar).
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.values_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array's own backing buffers are entirely in the scalar representation, and
    /// therefore stand for a single list repeated [`Self::len`] times in the memory of that list
    /// alone.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.values_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element of this array equals, if its own backing buffers both hold
    /// one slot.
    ///
    /// The inner [`Option`] is that element, so an array of nothing but nulls yields `Some(None)`.
    /// Returns `None` for an empty array, and whenever a buffer is flat over more than one element
    /// — its elements need not be equal, even if the other buffer is scalar.
    ///
    /// This is what lets equality avoid walking a scalar array of unbounded length.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<Box<dyn PlArray>>> {
        let is_shared = self.values.len() == self.width
            && self
                .validity
                .as_ref()
                .is_none_or(|validity| validity.len() == 1);

        // SAFETY: the array is not empty, so element 0 is in bounds.
        (is_shared && self.length > 0).then(|| unsafe { self.get_unchecked(0) })
    }

    /// The range of [`Self::values`] the element at `i` covers, which is always [`Self::width`]
    /// values wide.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value_range(&self, i: usize) -> Range<usize> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.value_range_unchecked(i) }
    }

    /// The range of [`Self::values`] the element at `i` covers, which is always [`Self::width`]
    /// values wide.
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

    /// Returns the element at `i`: the values array sliced to the range the element covers.
    ///
    /// This function is `O(1)`. The value of a null element is undetermined (it can be any list of
    /// the width).
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
    /// This function is `O(1)`. The value of a null element is undetermined (it can be any list of
    /// the width).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> Box<dyn PlArray> {
        let range = unsafe { self.value_range_unchecked(i) };
        // SAFETY: the values hold the width of every element, or the one they all cover.
        unsafe { self.values.sliced_unchecked(range.start, self.width) }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// A valid list may still hold null values.
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
    /// The values of null elements are undetermined (they can be any list of the width).
    #[inline]
    pub fn values_iter(&self) -> PlFixedSizeListValuesIter<'_> {
        PlFixedSizeListValuesIter::new(self)
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlFixedSizeListIter<'_> {
        PlFixedSizeListIter::new(self)
    }

    /// Returns an iterator over `length` elements, repeating the single element of this array if
    /// that is all it holds, and ignoring validity.
    ///
    /// This is [`Self::broadcast_iter`] without the validity check, exactly as
    /// [`Self::values_iter`] is [`Self::iter`] without it. The values of null elements are
    /// undetermined (they can be any list of the width).
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlFixedSizeListValuesIter<'_> {
        assert_broadcastable(self.length, length);
        // SAFETY: this array broadcasts to `length`, which is what was just asserted.
        PlFixedSizeListValuesIter::new_broadcast(self, length)
    }

    /// Returns an iterator over `length` optional elements, repeating the single element of this
    /// array if that is all it holds.
    ///
    /// This array either has `length` elements — in which case this is [`Self::iter`] — or a
    /// single element, which the `length` elements this yields then all read. Broadcasting is
    /// `O(1)`, and allocates nothing: the element is repeated as it is read, rather than
    /// materialized into an array to iterate the way [`Self::new_from_index`] would have to.
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast_iter(&self, length: usize) -> PlFixedSizeListIter<'_> {
        assert_broadcastable(self.length, length);
        // SAFETY: this array broadcasts to `length`, which is what was just asserted.
        PlFixedSizeListIter::new_broadcast(self, length)
    }

    /// Replaces the validity mask.
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor scalar for this array's length.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Replaces the validity mask.
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor scalar for this array's length.
    pub fn set_validity(&mut self, validity: Option<Bitmap>) {
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

        if self.values_are_flat() {
            // There are no offsets to leave the values outside the slice behind, so they are
            // sliced along with it, a width at a time.
            unsafe {
                self.values
                    .slice_unchecked(offset * self.width, length * self.width)
            };
        } else if length == 0 {
            // Scalar values are unaffected by slicing — every element covers the same ones — with
            // the one exception of an empty slice, which keeps no element to share them.
            unsafe { self.values.slice_unchecked(0, 0) };
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

    /// Creates a [`PlFixedSizeListArray`] of `length` copies of the element at `index`.
    ///
    /// This function is `O(1)`: the values of the result are the element being repeated, sliced
    /// out of the very same values array, and every one of its elements covers all of them. A null
    /// element repeats as `length` nulls.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        assert!(index < self.length, "index out of bounds");
        unsafe { self.new_from_index_unchecked(index, length) }
    }

    /// Creates a [`PlFixedSizeListArray`] of `length` copies of the element at `index`.
    ///
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        // The values of a null element are undetermined, so they are repeated as they are found:
        // it is the mask that makes every element of the result null.
        let element = unsafe { self.value_unchecked(index) };

        if unsafe { self.is_null_unchecked(index) } {
            Self::new_full_null(element, length)
        } else {
            Self::new_scalar(element, length)
        }
    }

    /// Returns an equivalent array whose values hold the values of every element and whose mask
    /// holds one bit per element.
    ///
    /// The values array is left as it is: it carries its own representation, which being
    /// [`flat`](Self::is_flat) says nothing about. The result carries its representation in its
    /// type: see [`Flat`] for what a flat array is a proof of.
    ///
    /// Materializing scalar values is what costs here, and it costs more than it does for the
    /// arrays whose elements are a single value: the one list every element covers has to be
    /// written out once per element, which is `O(len * width)`, and it is [`concatenate_repeated`]
    /// that does it, so the values of the result keep whatever representation copies of that list
    /// concatenate into. Nothing is written out when every element is null, since the values of a
    /// null element are undetermined and one of them repeated is a values array like any other.
    ///
    /// # Example
    /// ```
    /// use polars_array::{PlArray, PlFixedSizeListArray, PlPrimitiveArray};
    ///
    /// // Three copies of `[1, 2]`, over the values of that one list.
    /// let scalar = PlFixedSizeListArray::new_scalar(
    ///     Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
    ///     3,
    /// );
    /// assert_eq!(scalar.values().len(), 2);
    ///
    /// // Its flat counterpart holds the three lists one after the other.
    /// let flat = scalar.to_flat();
    /// assert_eq!(flat.values().len(), 6);
    /// assert_eq!(flat, scalar);
    /// ```
    pub fn to_flat(&self) -> Flat<Self> {
        if self.is_flat() {
            return Flat(self.clone());
        }

        let validity = self.validity().map(|validity| validity.to_flat());

        let values = if self.values_are_flat() {
            self.values.clone()
        } else if self.length > 0 && self.null_count() == self.length {
            // Every element is null, so every list is undetermined: repeating one value of the
            // element they share is a values array of the right length like any other, and it is
            // `O(1)` for every values array but a struct one.
            let flat_len = self.flat_values_len();
            self.values.new_from_index(0, flat_len)
        } else {
            // The one list every element covers, written out once per element. Concatenating it
            // with copies of itself is what repeats it, and that keeps the values of the result
            // scalar when the list is itself a single repeated value.
            concatenate_repeated(&*self.values, self.length)
                .expect("copies of one array always concatenate")
        };

        // SAFETY: the values are the width of the element every element covers, repeated once per
        // element, and the mask is the flat counterpart of one that was flat or scalar for this
        // array's length.
        Flat(unsafe { Self::new_unchecked(values, self.width, self.length, validity) })
    }

    /// Borrows this array as a [`Flat`] one, if its values already hold the values of every
    /// element and its mask one bit per element.
    ///
    /// This is the `O(1)` counterpart of [`Self::to_flat`]: it materializes nothing, and returns
    /// `None` rather than writing out a scalar buffer when this array is not
    /// [`flat`](Self::is_flat).
    ///
    /// # Example
    /// ```
    /// use polars_array::{PlFixedSizeListArray, PlPrimitiveArray};
    ///
    /// let arr = PlFixedSizeListArray::from_values(
    ///     Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4])),
    ///     2,
    /// );
    /// assert!(arr.as_flat().is_some());
    ///
    /// // A billion copies of one list share its values, so they have to be written out.
    /// let scalar = PlFixedSizeListArray::new_scalar(
    ///     Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
    ///     1_000_000_000,
    /// );
    /// assert!(scalar.as_flat().is_none());
    /// ```
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: the values of a flat array hold the width of every element, and its mask one bit
        // per element.
        self.is_flat()
            .then(|| unsafe { Flat::from_ref_unchecked(self) })
    }

    /// The number of values a flat counterpart of this array holds.
    ///
    /// # Panics
    /// Panics if that overflows a `usize`, which no values array has the memory to back.
    #[inline]
    fn flat_values_len(&self) -> usize {
        self.length.checked_mul(self.width).expect(
            "the values of the flat counterpart of the fixed size list array overflow a `usize`",
        )
    }
}

impl<'a> IntoIterator for &'a PlFixedSizeListArray {
    type Item = Option<Box<dyn PlArray>>;
    type IntoIter = PlFixedSizeListIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares two arrays element-wise; the values of null elements are not part of a value, so an
/// array compares equal to any other one holding the same lists. The width is part of one: lists
/// of different widths are different lists, and an array holds nothing else.
impl PartialEq for PlFixedSizeListArray {
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

        (0..self.length).all(|i| unsafe {
            self.is_null_unchecked(i) || self.value_unchecked(i).eq_dyn(&*other.value_unchecked(i))
        })
    }
}

impl Eq for PlFixedSizeListArray {}

/// Compares an array of unknown representation against a flat one; see
/// [`PartialEq<PlFixedSizeListArray> for Flat<PlFixedSizeListArray>`](Flat).
impl PartialEq<Flat<PlFixedSizeListArray>> for PlFixedSizeListArray {
    #[inline]
    fn eq(&self, other: &Flat<PlFixedSizeListArray>) -> bool {
        *self == other.0
    }
}

impl std::fmt::Debug for PlFixedSizeListArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The values array formats its own scalar representation, so this never materializes one:
        // it is listed as it is backed, which is one element's worth for a scalar array.
        let mut s = f.debug_struct("PlFixedSizeListArray");
        s.field("length", &self.length);
        s.field("width", &self.width);
        if let Some(validity) = self.validity() {
            s.field("validity", &validity);
        }
        s.field("values", &self.values).finish()
    }
}

impl PlArray for PlFixedSizeListArray {
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
        PlArrayType::FixedSizeList
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
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
    use crate::{PlListArray, PlPrimitiveArray, PlStructArray};

    /// The values array every test builds its lists over.
    fn values() -> Box<dyn PlArray> {
        Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5, 6]))
    }

    /// The three lists `[1, 2]`, `[3, 4]` and `[5, 6]` over [`values`].
    fn arr() -> PlFixedSizeListArray {
        PlFixedSizeListArray::from_values(values(), 2)
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
        assert_eq!(arr.values().len(), 6);

        assert_eq!(arr.value_range(0), 0..2);
        assert_eq!(arr.value_range(1), 2..4);
        assert_eq!(arr.value_range(2), 4..6);

        assert_eq!(elements(&*arr.value(0)), [Some(1), Some(2)]);
        assert_eq!(elements(&*arr.value(2)), [Some(5), Some(6)]);
        assert_eq!(elements(&*arr.get(1).unwrap()), [Some(3), Some(4)]);
    }

    #[test]
    fn an_empty_array_holds_no_values() {
        let arr = PlFixedSizeListArray::new_empty(values(), 2);

        assert!(arr.is_empty());
        assert_eq!(arr.width(), 2);
        assert_eq!(arr.values().len(), 0);
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert_eq!(arr.scalar_value(), None);

        // The values are what determines the type of the lists, of which there are none.
        assert_eq!(arr.values().array_type(), values().array_type());
    }

    #[test]
    fn null_scalar() {
        // Every element is null, over the two values they all share: the mask is a single bit, and
        // the values are one list.
        let arr = PlFixedSizeListArray::new_full_null(
            Box::new(PlPrimitiveArray::<i32>::new_full_null(2)),
            1_000_000,
        );

        assert!(arr.is_scalar());
        assert!(arr.validity_is_scalar());
        assert!(arr.values_are_scalar());
        assert!(!arr.values_are_flat());
        assert_eq!(arr.width(), 2);
        assert_eq!(arr.validity().unwrap().len(), 1_000_000);
        assert_eq!(arr.validity().unwrap().bitmap().len(), 1);
        assert_eq!(arr.values().len(), 2);
        assert_eq!(arr.null_count(), 1_000_000);
        assert!(arr.has_nulls());
        assert!(arr.is_null(999_999));
        assert_eq!(arr.get(999_999), None);
        assert_eq!(arr.value_range(999_999), 0..2);

        let valid = arr.without_validity();
        assert_eq!(valid.null_count(), 0);
        assert!(valid.validity().is_none());
    }

    #[test]
    fn scalar_validity_over_flat_values() {
        let arr = arr().with_validity(Some(Bitmap::new_zeroed(1)));

        assert!(arr.validity_is_scalar());
        assert!(!arr.is_flat());
        // The values are still flat over three elements, so the array as a whole is not scalar.
        assert!(!arr.is_scalar());
        assert_eq!(arr.null_count(), 3);
        assert!(arr.iter().all(|element| element.is_none()));

        // The values are untouched, so the lists of the null elements are still there.
        assert_eq!(elements(&*arr.value(0)), [Some(1), Some(2)]);
        assert_eq!(
            arr.values_iter().map(|list| list.len()).collect::<Vec<_>>(),
            [2, 2, 2],
        );
    }

    #[test]
    fn flat_validity() {
        let arr =
            PlFixedSizeListArray::new(values(), 2, 3, Some(Bitmap::from_iter([true, false, true])));

        assert!(!arr.validity_is_scalar());
        assert!(arr.is_flat());
        assert_eq!(arr.null_count(), 1);
        assert!(arr.is_null(1));
        assert_eq!(
            arr.iter()
                .map(|element| element.map(|list| list.len()))
                .collect::<Vec<_>>(),
            [Some(2), None, Some(2)],
        );
    }

    #[test]
    fn try_new_rejects_invalid_components() {
        // The values must hold the width of every element, or the width of the one element they
        // all cover.
        assert!(PlFixedSizeListArray::try_new(values(), 2, 4, None).is_err());
        assert!(PlFixedSizeListArray::try_new(values(), 2, 3, None).is_ok());
        assert!(PlFixedSizeListArray::try_new(values(), 6, 1, None).is_ok());
        assert!(PlFixedSizeListArray::try_new(values(), 6, 1_000_000, None).is_ok());

        // An empty array has no element for a scalar values array to stand for.
        assert!(PlFixedSizeListArray::try_new(values(), 6, 0, None).is_err());
        assert!(PlFixedSizeListArray::try_new(values().sliced(0, 0), 6, 0, None).is_ok());

        // A width that does not divide the values is one no number of elements adds up to.
        assert!(PlFixedSizeListArray::try_new(values(), 4, 1, None).is_err());

        // The validity mask must be flat or scalar.
        assert!(
            PlFixedSizeListArray::try_new(values(), 2, 3, Some(Bitmap::new_zeroed(2))).is_err()
        );
        assert!(PlFixedSizeListArray::try_new(values(), 2, 3, Some(Bitmap::new_zeroed(1))).is_ok());
        assert!(PlFixedSizeListArray::try_new(values(), 2, 3, Some(Bitmap::new_zeroed(3))).is_ok());
    }

    #[test]
    fn slicing_slices_the_values() {
        let arr =
            PlFixedSizeListArray::new(values(), 2, 3, Some(Bitmap::from_iter([true, false, true])))
                .sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.width(), 2);
        assert_eq!(arr.null_count(), 1);
        assert!(arr.is_flat());

        // There are no offsets to leave the values outside the slice behind.
        assert_eq!(arr.values().len(), 4);
        assert_eq!(elements(arr.values()), [Some(3), Some(4), Some(5), Some(6)],);
        assert_eq!(elements(&*arr.value(1)), [Some(5), Some(6)]);

        // Slicing away every element leaves no values behind either.
        let arr = arr.sliced(2, 0);
        assert!(arr.is_empty());
        assert_eq!(arr.values().len(), 0);
        assert_eq!(arr.width(), 2);
    }

    #[test]
    fn slicing_keeps_scalar_validity() {
        let arr = arr()
            .with_validity(Some(Bitmap::new_zeroed(1)))
            .sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.validity().unwrap().len(), 2);
        assert_eq!(arr.validity().unwrap().bitmap().len(), 1);
        assert!(arr.validity_is_scalar());
        assert_eq!(arr.null_count(), 2);
    }

    #[test]
    fn slicing_a_scalar_is_free() {
        let arr = PlFixedSizeListArray::new_scalar(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            1_000_000_000,
        );

        let sliced = arr.clone().sliced(500, 2);
        assert_eq!(sliced.len(), 2);
        assert!(sliced.is_scalar());
        assert_eq!(sliced.values().len(), 2);
        assert_eq!(elements(&*sliced.value(1)), [Some(1), Some(2)]);

        // An empty slice keeps no element to share the values, so they go with it.
        let empty = arr.sliced(0, 0);
        assert!(empty.is_empty());
        assert_eq!(empty.width(), 2);
        assert_eq!(empty.values().len(), 0);
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
    #[should_panic(expected = "neither flat nor scalar")]
    fn setting_a_validity_mask_that_is_neither_flat_nor_scalar_panics() {
        let _ = arr().with_validity(Some(Bitmap::new_zeroed(2)));
    }

    #[test]
    #[should_panic(expected = "do not divide into lists of width")]
    fn from_values_rejects_a_width_that_does_not_divide_the_values() {
        let _ = PlFixedSizeListArray::from_values(values(), 4);
    }

    #[test]
    #[should_panic(expected = "width zero")]
    fn from_values_rejects_a_width_of_zero() {
        let _ = PlFixedSizeListArray::from_values(values(), 0);
    }

    #[test]
    fn new_scalar_repeats_one_list() {
        let arr = PlFixedSizeListArray::new_scalar(
            Box::new(PlPrimitiveArray::new_scalar(7i32, 1_000_000_000)),
            1_000_000_000,
        );

        assert_eq!(arr.len(), 1_000_000_000);
        assert_eq!(arr.width(), 1_000_000_000);
        assert!(arr.is_scalar());
        assert!(!arr.values_are_flat());
        assert_eq!(arr.values().len(), 1_000_000_000);
        assert_eq!(arr.value_range(999_999_999), 0..1_000_000_000);
        assert_eq!(arr.null_count(), 0);

        // A single copy is one element, which the two representations both stand for.
        let one = PlFixedSizeListArray::new_scalar(values(), 1);
        assert!(one.is_scalar());
        assert!(one.is_flat());
        assert_eq!(one.width(), 6);
        assert_eq!(one, PlFixedSizeListArray::from_values(values(), 6));

        // No copies at all is an empty array, which keeps the width but not the values.
        let none = PlFixedSizeListArray::new_scalar(values(), 0);
        assert!(none.is_empty());
        assert_eq!(none.width(), 6);
        assert_eq!(none.values().len(), 0);

        // A length times a width that overflows a `usize` is a length no flat array can have, and
        // the scalar representation is all that is left.
        let huge = PlFixedSizeListArray::new_scalar(values(), usize::MAX);
        assert!(huge.is_scalar());
        assert!(!huge.values_are_flat());
        assert_eq!(huge.value_range(usize::MAX - 1), 0..6);
        assert_eq!(
            elements(&*huge.value(usize::MAX - 1)),
            [Some(1), Some(2), Some(3), Some(4), Some(5), Some(6)]
        );
    }

    #[test]
    fn new_from_index_repeats_one_element() {
        let arr = arr();

        let repeated = arr.new_from_index(1, 1_000_000_000);
        assert_eq!(repeated.len(), 1_000_000_000);
        assert_eq!(repeated.width(), 2);
        assert!(repeated.is_scalar());
        assert_eq!(repeated.values().len(), 2);
        assert_eq!(repeated.null_count(), 0);
        assert_eq!(elements(&*repeated.value(999_999_999)), [Some(3), Some(4)]);

        // A null element repeats as nulls, over the values it was found with.
        let nulls = arr
            .clone()
            .with_validity(Some(Bitmap::from_iter([true, false, true])))
            .new_from_index(1, 1_000_000);
        assert_eq!(nulls.len(), 1_000_000);
        assert_eq!(nulls.width(), 2);
        assert_eq!(nulls.null_count(), 1_000_000);
        assert!(nulls.validity().unwrap().is_scalar());

        // The repetition of a repeated element is that same element again.
        assert_eq!(repeated.new_from_index(0, 1_000_000_000), repeated);

        assert!(arr.new_from_index(0, 0).is_empty());
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn repeating_an_element_out_of_bounds_panics() {
        let _ = arr().new_from_index(3, 1);
    }

    #[test]
    fn to_flat_writes_the_lists_out() {
        let scalar = PlFixedSizeListArray::new_scalar(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            3,
        );
        let flat = scalar.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.values().len(), 6);
        assert_eq!(
            elements(flat.values()),
            [Some(1), Some(2), Some(1), Some(2), Some(1), Some(2)],
        );
        assert_eq!(flat, scalar);
    }

    #[test]
    fn to_flat_of_an_already_flat_array_only_clones() {
        let arr = arr();
        let flat = arr.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.values().len(), 6);
        assert_eq!(flat, arr);

        // Flat values under a scalar mask are not a flat array: the mask is written out, and the
        // values are left alone.
        let arr = arr.with_validity(Some(Bitmap::new_zeroed(1)));
        let flat = arr.to_flat();
        assert!(flat.is_flat());
        assert_eq!(flat.validity().unwrap().bitmap().len(), 3);
        assert_eq!(flat.values().len(), 6);
        assert_eq!(flat, arr);
    }

    #[test]
    fn to_flat_of_a_list_of_scalar_values_does_not_materialize_them() {
        // A billion copies of one list of a repeated value: the lists are laid out one per
        // element, but the value they all hold is still a single slot.
        let arr = PlFixedSizeListArray::new_scalar(
            Box::new(PlPrimitiveArray::new_scalar(7i32, 4)),
            1_000_000,
        );
        let flat = arr.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.values().len(), 4_000_000);
        assert!(
            flat.values()
                .as_any()
                .downcast_ref::<PlPrimitiveArray<i32>>()
                .unwrap()
                .is_scalar()
        );
    }

    #[test]
    fn to_flat_writes_out_no_undetermined_list() {
        // Every element is null, so its list is undetermined: the values of the result are as long
        // as they have to be, and nothing is written out to fill them.
        let arr = PlFixedSizeListArray::new_full_null(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            1_000_000,
        );
        let flat = arr.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.len(), 1_000_000);
        assert_eq!(flat.values().len(), 2_000_000);
        assert_eq!(flat.null_count(), 1_000_000);
        assert!(
            flat.values()
                .as_any()
                .downcast_ref::<PlPrimitiveArray<i32>>()
                .unwrap()
                .is_scalar()
        );
        assert_eq!(flat, arr);
    }

    #[test]
    fn as_flat_borrows_an_already_flat_array() {
        let arr = arr();
        assert_eq!(arr.as_flat().unwrap().values().len(), 6);

        // A billion copies of one list share its values, so they have to be written out.
        let scalar = PlFixedSizeListArray::new_scalar(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            1_000_000_000,
        );
        assert!(scalar.as_flat().is_none());

        // One element is both, so it is borrowed rather than written out.
        let one = PlFixedSizeListArray::new_scalar(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            1,
        );
        assert!(one.is_scalar());
        assert!(one.as_flat().is_some());
    }

    #[test]
    fn equality_ignores_the_representation_but_not_the_width() {
        assert_eq!(arr(), arr());
        assert_ne!(arr(), arr().sliced(0, 2));

        // The same values, cut into lists of a different width.
        assert_ne!(arr(), PlFixedSizeListArray::from_values(values(), 3));

        // The same three lists, one shared and one written out per element.
        let scalar = PlFixedSizeListArray::new_scalar(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            3,
        );
        let flat = PlFixedSizeListArray::from_values(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 1, 2, 1, 2])),
            2,
        );
        assert_eq!(scalar, flat);

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
            PlFixedSizeListArray::from_values(
                Box::new(PlPrimitiveArray::from_vec(vec![1i64, 2, 3, 4, 5, 6])),
                2,
            ),
        );
    }

    #[test]
    fn equality_ignores_the_values_of_null_elements() {
        let mask = Bitmap::from_iter([true, false, true]);
        let lhs = arr().with_validity(Some(mask.clone()));
        let rhs = PlFixedSizeListArray::from_values(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 9, 9, 5, 6])),
            2,
        )
        .with_validity(Some(mask));

        assert_eq!(lhs, rhs);
    }

    #[test]
    fn equality_of_scalars_does_not_walk_elements() {
        let arr = PlFixedSizeListArray::new_scalar(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            1_000_000_000,
        );

        assert_eq!(arr, arr.clone());
        assert_ne!(
            arr,
            PlFixedSizeListArray::new_scalar(
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 3])),
                1_000_000_000,
            ),
        );

        // Two fully null arrays hold nothing determined to compare.
        let nulls = PlFixedSizeListArray::new_full_null(
            Box::new(PlPrimitiveArray::<i32>::new_full_null(2)),
            1_000_000_000,
        );
        assert_eq!(nulls, nulls.clone());
        assert_ne!(nulls, arr);
    }

    #[test]
    fn lists_of_width_zero_hold_no_values() {
        let arr =
            PlFixedSizeListArray::new(Box::new(PlPrimitiveArray::<i32>::new_empty()), 0, 5, None);

        assert_eq!(arr.len(), 5);
        assert_eq!(arr.width(), 0);
        assert_eq!(arr.values().len(), 0);
        // Every representation of no values at all is the same one.
        assert!(arr.is_flat());
        assert!(arr.is_scalar());
        assert_eq!(arr.value_range(4), 0..0);
        assert!(arr.value(4).is_empty());
        assert_eq!(arr.to_flat().values().len(), 0);
    }

    #[test]
    fn lists_of_lists() {
        // Two lists of two lists of one value each.
        let inner = PlFixedSizeListArray::from_values(values(), 1);
        let outer = PlFixedSizeListArray::from_values(Box::new(inner), 3);

        assert_eq!(outer.len(), 2);
        assert_eq!(outer.width(), 3);
        assert_eq!(outer.values().array_type(), PlArrayType::FixedSizeList);

        let element = outer.value(1);
        assert_eq!(element.len(), 3);
        assert_eq!(
            elements(
                &*element
                    .as_any()
                    .downcast_ref::<PlFixedSizeListArray>()
                    .unwrap()
                    .value(0),
            ),
            [Some(4)],
        );

        // The values need not be a values array of their own: any array will do.
        let structs = PlFixedSizeListArray::from_values(
            Box::new(PlStructArray::from_fields(vec![values()])),
            2,
        );
        assert_eq!(structs.len(), 3);
        assert_eq!(structs.value(1).array_type(), PlArrayType::Struct);

        let lists =
            PlFixedSizeListArray::from_values(Box::new(PlListArray::new_scalar(values(), 6)), 3);
        assert_eq!(lists.len(), 2);
        assert_eq!(lists.value(0).array_type(), PlArrayType::List);
    }

    #[test]
    fn iterators_are_exact_sized() {
        let arr = arr().with_validity(Some(Bitmap::from_iter([true, false, true])));

        assert_eq!(arr.iter().len(), 3);
        assert_eq!(arr.values_iter().len(), 3);
        assert_eq!((&arr).into_iter().count(), 3);
        assert_eq!(
            arr.iter()
                .map(|element| element.map(|list| list.len()))
                .collect::<Vec<_>>(),
            [Some(2), None, Some(2)],
        );

        // The values of a null element are still there to be walked.
        assert_eq!(
            elements(&*arr.values_iter().next_back().unwrap()),
            [Some(5), Some(6)],
        );
        assert_eq!(
            elements(&*arr.values_iter().nth(1).unwrap()),
            [Some(3), Some(4)]
        );
    }

    #[test]
    fn broadcasting_one_list() {
        // The single list of a billion sevens, over values that cost `O(1)`.
        let single = PlFixedSizeListArray::from_values(
            Box::new(PlPrimitiveArray::<i32>::new_scalar(7, 1_000_000_000)),
            1_000_000_000,
        );
        assert_eq!(single.len(), 1);

        // A billion copies of that list are iterated without the list ever being materialized:
        // every element is the same `O(1)` slice of the same values array.
        let mut iter = single.broadcast_iter(1_000_000_000);
        assert_eq!(iter.len(), 1_000_000_000);
        assert_eq!(iter.next().unwrap().unwrap().len(), 1_000_000_000);
        assert_eq!(iter.nth(999_999_997).unwrap().unwrap().len(), 1_000_000_000);
        assert_eq!(iter.next_back().unwrap().unwrap().len(), 1_000_000_000);
        assert!(iter.next().is_none());
        assert_eq!(
            single
                .broadcast_values_iter(1_000_000_000)
                .nth(999_999_999)
                .unwrap()
                .len(),
            1_000_000_000,
        );

        // A null element broadcasts as nulls.
        let nulls = PlFixedSizeListArray::new_full_null(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            1,
        );
        assert!(nulls.broadcast_iter(3).all(|element| element.is_none()));
        assert_eq!(nulls.broadcast_iter(3).count(), 3);

        // An array of the length asked for iterates as it is, whatever it is backed by.
        let arr = arr().with_validity(Some(Bitmap::from_iter([true, false, true])));
        let lists = |iter: PlFixedSizeListIter<'_>| {
            iter.map(|element| element.map(|list| elements(&*list)))
                .collect::<Vec<_>>()
        };
        assert_eq!(lists(arr.broadcast_iter(3)), lists(arr.iter()));
        assert_eq!(
            elements(&*arr.broadcast_values_iter(3).next_back().unwrap()),
            [Some(5), Some(6)],
        );

        // A scalar array broadcasts to the length it has like any other.
        let scalar = PlFixedSizeListArray::new_scalar(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            3,
        );
        assert_eq!(
            lists(scalar.broadcast_iter(3)),
            vec![Some(vec![Some(1), Some(2)]); 3],
        );

        // Broadcasting to nothing yields nothing, and an empty array broadcasts to nothing
        // else: it has no element to repeat.
        assert_eq!(single.broadcast_iter(0).len(), 0);
        assert_eq!(
            PlFixedSizeListArray::new_empty(values(), 2)
                .broadcast_iter(0)
                .len(),
            0,
        );
    }

    #[test]
    #[should_panic(expected = "an array of length 3 does not broadcast to length 4")]
    fn broadcasting_more_than_one_list_panics() {
        let _ = arr().broadcast_iter(4);
    }

    #[test]
    fn into_inner_returns_the_components() {
        let (values, width, length, validity) = arr()
            .with_validity(Some(Bitmap::new_zeroed(1)))
            .into_inner();

        assert_eq!(values.len(), 6);
        assert_eq!(width, 2);
        assert_eq!(length, 3);
        assert_eq!(validity.unwrap().len(), 1);
    }

    #[test]
    fn debug_lists_the_width_and_the_values() {
        assert_eq!(
            format!("{:?}", arr()),
            "PlFixedSizeListArray { length: 3, width: 2, \
             values: PlPrimitiveArray[1, 2, 3, 4, 5, 6] }",
        );

        // Neither a scalar validity mask nor a scalar values array is materialized.
        let arr = PlFixedSizeListArray::new_full_null(
            Box::new(PlPrimitiveArray::new_scalar(7i32, 2)),
            1_000_000_000,
        );
        assert_eq!(
            format!("{arr:?}"),
            "PlFixedSizeListArray { length: 1000000000, width: 2, \
             validity: PlBitmapRef[false; 1000000000], values: PlPrimitiveArray[7; 2] }",
        );
    }

    #[test]
    fn behind_the_trait_object() {
        let arr: Box<dyn PlArray> = Box::new(arr());

        assert_eq!(arr.array_type(), PlArrayType::FixedSizeList);
        assert!(arr.array_type().is_fixed_size_list());
        assert!(!arr.array_type().is_list());
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 0);
        assert!(arr.is_valid(2));

        let sliced = arr.sliced(1, 2);
        assert_eq!(sliced.len(), 2);
        assert_eq!(sliced.array_type(), arr.array_type());

        let nulled = arr.with_validity(Some(Bitmap::new_zeroed(1)));
        assert_eq!(nulled.null_count(), 3);
        assert!(nulled.without_validity().validity().is_none());

        // A billion copies of one element stay `O(1)` behind the trait object.
        let repeated = arr.new_from_index(1, 1_000_000_000);
        assert_eq!(repeated.len(), 1_000_000_000);
        assert_eq!(repeated.array_type(), arr.array_type());
        assert_eq!(&repeated.sliced(999_999_999, 1), &arr.sliced(1, 1));

        assert_eq!(&arr, &arr.clone());
    }
}
