use std::ops::Range;

use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmapRef, validity_eq};
use crate::broadcast::is_valid_buffer_len;
use crate::concatenate::concatenate;

mod iterator;

pub use iterator::{PlListIter, PlListValuesIter};

/// An immutable, cheaply cloneable sequence of `length` optional lists over one values array.
///
/// This is the variable-length nested array of this crate: it holds no values of its own, only a
/// validity mask and the offsets that cut a single values array into `length` consecutive slices.
/// Element `i` is `values[offsets[i]..offsets[i + 1]]`. It carries no logical type — the values
/// array is a [`PlArray`], and what a caller thinks of as the list's inner type lives at a higher
/// level.
///
/// The offsets are always `u64` and always *flat*: a list array of `length` elements is backed by
/// exactly `length + 1` offsets, since the end of one list is the start of the next and the last
/// list needs an end of its own. There is therefore no scalar representation of a list repeated
/// `length` times, and a list array always costs `O(len)` memory in its offsets — unlike every
/// other array in this crate, whose backing buffers may each be scalar.
///
/// What the separate `length` field buys here is the same as everywhere else for the *validity
/// mask*: it is read through
/// [`broadcast_index(i, bitmap.len())`](crate::broadcast::broadcast_index), so it is either flat
/// (one bit per element) or scalar (a single bit shared by every element), which is what lets a
/// fully null list array carry a one-bit mask. See [`crate::broadcast`] for the full rules.
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
///         .values()
///         .as_slice(),
///     [3, 4, 5],
/// );
/// ```
#[derive(Clone)]
pub struct PlListArray {
    values: Box<dyn PlArray>,
    offsets: Buffer<u64>,
    length: usize,
    validity: Option<Bitmap>,
}

impl PlListArray {
    /// Creates a [`PlListArray`] out of its internal components.
    ///
    /// This function is `O(len)`: it walks the offsets to check that they are ordered.
    ///
    /// # Errors
    /// This function errors if `offsets` does not hold exactly `length + 1` offsets, if the offsets
    /// are not monotonically non-decreasing, if the last offset exceeds the length of `values`, or
    /// if `validity` is neither flat (length equal to `length`) nor scalar (length one).
    pub fn try_new(
        values: Box<dyn PlArray>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            offsets.len().checked_sub(1) == Some(length),
            ComputeError:
            "offsets buffer of length {} is not valid for a list array of length {}: it needs one \
             offset per element plus the end of the last",
            offsets.len(), length,
        );

        // The offsets are ordered, so checking the last one against the values array covers them
        // all — including that every one of them fits in a `usize`.
        for (i, window) in offsets.windows(2).enumerate() {
            polars_ensure!(
                window[0] <= window[1],
                ComputeError:
                "offset {} of the list array is {}, which is smaller than the offset {} before it",
                i + 1, window[1], window[0],
            );
        }
        polars_ensure!(
            offsets[length] <= values.len() as u64,
            ComputeError:
            "the last offset of the list array is {}, which exceeds the length {} of its values",
            offsets[length], values.len(),
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
            offsets,
            length,
            validity,
        })
    }

    /// Creates a [`PlListArray`] out of its internal components.
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

    /// Creates a [`PlListArray`] out of its internal components without validating them.
    ///
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `offsets` must hold exactly `length + 1` monotonically non-decreasing offsets, the last of
    /// which does not exceed the length of `values`, and `validity` must be either flat (length
    /// equal to `length`) or scalar (length one).
    #[inline]
    pub unsafe fn new_unchecked(
        values: Box<dyn PlArray>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        if cfg!(debug_assertions) {
            assert_eq!(offsets.len().checked_sub(1), Some(length));
            assert!(offsets.windows(2).all(|window| window[0] <= window[1]));
            assert!(offsets[length] <= values.len() as u64);
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_valid_buffer_len(v.len(), length))
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

    /// Creates a fully valid [`PlListArray`] from `values` and `offsets`, taking its length from
    /// the offsets.
    ///
    /// This function is `O(len)`.
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

    /// Creates a [`PlListArray`] of `length` nulls over `values`.
    ///
    /// Every element is null, so its value is undetermined; each is given the empty list, which is
    /// what keeps the validity mask a single bit and the values array untouched. The offsets still
    /// hold one slot per element, so unlike the other arrays this is `O(length)` and not `O(1)`.
    #[inline]
    pub fn new_full_null(values: Box<dyn PlArray>, length: usize) -> Self {
        Self {
            values,
            offsets: Buffer::zeroed(length + 1),
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

    /// The backing offsets buffer, which holds [`Self::len`] `+ 1` offsets into [`Self::values`].
    ///
    /// The offsets are always flat, and are not normalized: the first one is whatever slicing left
    /// it, not necessarily zero.
    #[inline(always)]
    pub const fn offsets(&self) -> &Buffer<u64> {
        &self.offsets
    }

    /// Consumes this array into its internal components.
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
            .map(|validity| unsafe { PlBitmapRef::new_unchecked(validity, self.length) })
    }

    /// Whether the validity mask holds a single bit shared by every element.
    ///
    /// There is no such question to ask of the offsets, which are always flat.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
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
        // SAFETY: there is one offset per element plus one, so `i + 1` is in bounds. Every offset
        // is at most the length of the values array, and therefore fits in a `usize`.
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

        // The values array is left as it is: the offsets are what point into it, and they are not
        // normalized, so the elements that fall outside the slice simply stop being reachable.
        unsafe {
            self.offsets
                .slice_in_place_unchecked(offset..offset + length + 1)
        };

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
    /// There is no scalar representation of a repeated list — the offsets are always flat, and
    /// they have to be ordered — so the values of the element are repeated as well: this is
    /// `O(length * value_length(index))`, and `O(length)` for a null or empty element, whose values
    /// need not be written out. The values are themselves a [`PlArray`], so a scalar one is
    /// repeated in `O(1)`.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`, or if the total length of the values of the result
    /// overflows a `usize`.
    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        assert!(index < self.length, "index out of bounds");
        unsafe { self.new_from_index_unchecked(index, length) }
    }

    /// Creates a [`PlListArray`] of `length` copies of the element at `index`.
    ///
    /// # Panics
    /// Panics if the total length of the values of the result overflows a `usize`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        if unsafe { self.is_null_unchecked(index) } {
            return Self::new_full_null(self.values.clone(), length);
        }

        let range = unsafe { self.value_range_unchecked(index) };
        if length == 0 || range.is_empty() {
            // The values array is kept as it is — it is what determines the type of the lists —
            // but no element of it is reachable.
            return Self {
                values: self.values.clone(),
                offsets: Buffer::zeroed(length + 1),
                length,
                validity: None,
            };
        }

        // SAFETY: the offsets are ordered and bounded by the length of the values array.
        let element = unsafe { self.values.sliced_unchecked(range.start, range.len()) };

        // Concatenation is what materializes the repetition, and it keeps the values scalar when
        // the element is itself a single repeated value.
        let values = concatenate(&vec![&*element; length])
            .expect("an array concatenates with copies of itself");

        let value_length = range.len() as u64;
        let offsets = (0..=length as u64)
            .map(|i| i * value_length)
            .collect::<Vec<_>>();

        // SAFETY: every list of the result has the length of the element, so the offsets are
        // ordered and there is one per element plus the end of the last, and the last of them is
        // the total length the element was repeated over. Every element is valid.
        unsafe { Self::new_unchecked(values, Buffer::from(offsets), length, None) }
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
        // left to compare.
        if self.length > 0 && self.null_count() == self.length {
            return true;
        }

        (0..self.length).all(|i| unsafe {
            self.is_null_unchecked(i) || self.value_unchecked(i).eq_dyn(&*other.value_unchecked(i))
        })
    }
}

impl Eq for PlListArray {}

impl std::fmt::Debug for PlListArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The values array formats its own scalar representation, so this never materializes one;
        // the offsets are always flat, and are listed like the values of a flat array.
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
        assert_eq!(arr.offsets().as_slice(), [0, 2, 2, 5]);
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
    fn the_values_outside_the_offsets_are_unreachable() {
        // The first list starts after the first value and the last ends before the last one.
        let arr = PlListArray::from_offsets(values(), Buffer::from(vec![1u64, 3]));

        assert_eq!(arr.len(), 1);
        assert_eq!(arr.values().len(), 5);
        assert_eq!(elements(&*arr.value(0)), [Some(2), Some(3)]);
    }

    #[test]
    fn null_scalar() {
        // Every element is null, and every list is empty; the mask is a single bit, but there is
        // still one offset per element.
        let arr = PlListArray::new_full_null(values(), 1_000_000);

        assert!(arr.validity_is_scalar());
        assert_eq!(arr.validity().unwrap().len(), 1_000_000);
        assert_eq!(arr.validity().unwrap().bitmap().len(), 1);
        assert_eq!(arr.offsets().len(), 1_000_001);
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
        let arr = arr().with_validity(Some(Bitmap::new_zeroed(1)));

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

        // They must be monotonically non-decreasing.
        assert!(
            PlListArray::try_new(values(), Buffer::from(vec![0u64, 5, 2, 5]), 3, None).is_err()
        );

        // The last of them must not reach past the values array.
        assert!(PlListArray::try_new(values(), Buffer::from(vec![0u64, 6]), 1, None).is_err());
        assert!(PlListArray::try_new(values(), Buffer::from(vec![0u64, 5]), 1, None).is_ok());

        // The validity mask must be flat or scalar.
        let offsets = Buffer::from(vec![0u64, 2, 2, 5]);
        assert!(
            PlListArray::try_new(values(), offsets.clone(), 3, Some(Bitmap::new_zeroed(2)))
                .is_err()
        );
        assert!(
            PlListArray::try_new(values(), offsets.clone(), 3, Some(Bitmap::new_zeroed(1))).is_ok()
        );
        assert!(PlListArray::try_new(values(), offsets, 3, Some(Bitmap::new_zeroed(3))).is_ok());
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
        assert_eq!(arr.offsets().as_slice(), [2, 2, 5]);
        assert_eq!(arr.validity().unwrap().bitmap().len(), 2);
        assert_eq!(arr.null_count(), 1);

        // The values array is left alone; the offsets are what point into it.
        assert_eq!(arr.values().len(), 5);
        assert_eq!(elements(&*arr.value(1)), [Some(3), Some(4), Some(5)]);

        // Slicing away every element leaves the end of the last list behind.
        let arr = arr.sliced(2, 0);
        assert!(arr.is_empty());
        assert_eq!(arr.offsets().as_slice(), [5]);
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
            arr().with_validity(Some(Bitmap::new_zeroed(1))),
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
        assert_eq!(arr.offsets().as_slice(), [0]);
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
            .with_validity(Some(Bitmap::new_zeroed(1)))
            .into_inner();

        assert_eq!(values.len(), 5);
        assert_eq!(offsets.as_slice(), [0, 0, 0, 0]);
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
        .with_validity(Some(Bitmap::new_zeroed(1)));
        assert_eq!(
            format!("{arr:?}"),
            "PlListArray { length: 1, validity: PlBitmapRef[false], \
             offsets: [0, 1000000000], values: PlPrimitiveArray[7; 1000000000] }",
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

        let nulled = arr.with_validity(Some(Bitmap::new_zeroed(1)));
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

        // The values of the element are repeated along with it: three copies of `[3, 4, 5]`.
        let repeated = arr.new_from_index(2, 3);
        assert_eq!(repeated.len(), 3);
        assert_eq!(repeated.offsets().as_slice(), [0, 3, 6, 9]);
        assert_eq!(repeated.values().len(), 9);
        assert_eq!(repeated.null_count(), 0);
        for i in 0..repeated.len() {
            assert_eq!(elements(&*repeated.value(i)), [Some(3), Some(4), Some(5)]);
        }

        // An empty element has no values to repeat, so the values array is left as it is.
        let repeated = arr.new_from_index(1, 4);
        assert_eq!(repeated.offsets().as_slice(), [0, 0, 0, 0, 0]);
        assert_eq!(repeated.values().len(), 5);
        assert!(elements(&*repeated.value(3)).is_empty());

        // A null element repeats as nulls, under a mask of a single bit, and every list is empty.
        let nulls = arr
            .clone()
            .with_validity(Some(Bitmap::from_iter([false, true, true])))
            .new_from_index(0, 4);
        assert_eq!(nulls.null_count(), 4);
        assert!(nulls.validity_is_scalar());
        assert_eq!(nulls.offsets().as_slice(), [0, 0, 0, 0, 0]);
        assert_eq!(nulls.get(3), None);

        assert!(arr.new_from_index(0, 0).is_empty());
        assert_eq!(
            unsafe { arr.new_from_index_unchecked(0, 2) },
            PlListArray::from_offsets(
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 1, 2])),
                Buffer::from(vec![0u64, 2, 4]),
            ),
        );
    }

    #[test]
    fn repeating_a_list_of_scalar_values_does_not_materialize_them() {
        // A single list of a billion sevens, over values that cost `O(1)`.
        let arr = PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::<i32>::new_scalar(7, 1_000_000_000)),
            Buffer::from(vec![0u64, 1_000_000_000]),
        );

        // Nothing here may walk the values: repeating them keeps them scalar.
        let repeated = arr.new_from_index(0, 2);
        assert_eq!(repeated.len(), 2);
        assert_eq!(repeated.values().len(), 2_000_000_000);
        assert_eq!(
            repeated.offsets().as_slice(),
            [0, 1_000_000_000, 2_000_000_000],
        );
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
    #[should_panic(expected = "index out of bounds")]
    fn repeating_a_list_out_of_bounds_panics() {
        let _ = arr().new_from_index(3, 1);
    }
}
