use arrow::bitmap::{Bitmap, MutableBitmap};
use polars_error::{PolarsResult, polars_ensure};

use crate::bitmap::{PlBitmapIter, PlBitmapRef};
use crate::broadcast::is_valid_buffer_len;

mod iterator;

pub use iterator::PlBooleanIter;

/// An immutable, cheaply cloneable sequence of `length` optional [`bool`] values.
///
/// This is the boolean counterpart of [`PlPrimitiveArray`](crate::PlPrimitiveArray): the values are
/// packed one bit per element into a [`Bitmap`] rather than one slot per element into a buffer. It
/// carries no logical type — only the physical values and their validity.
///
/// The logical length is stored separately from the backing bitmaps, which lets a *scalar* array —
/// one value repeated `length` times — be represented in `O(1)` memory. Element `i` reads slot
/// [`broadcast_index(i, bitmap.len())`](crate::broadcast::broadcast_index) of each backing bitmap,
/// so both `values` and `validity` are independently either dense (one bit per element) or
/// broadcast (a single shared bit). See [`crate::broadcast`] for the full rules.
///
/// # Example
/// ```
/// use polars_array::PlBooleanArray;
///
/// let dense = PlBooleanArray::from_vec(vec![true, false, true]);
/// assert_eq!(dense.len(), 3);
/// assert_eq!(
///     dense.iter().collect::<Vec<_>>(),
///     [Some(true), Some(false), Some(true)],
/// );
///
/// // A scalar array of a billion elements costs a single bit of memory.
/// let scalar = PlBooleanArray::new_scalar(true, 1_000_000_000);
/// assert_eq!(scalar.len(), 1_000_000_000);
/// assert_eq!(scalar.values().bitmap().len(), 1);
/// assert!(scalar.value(999_999_999));
/// ```
#[derive(Clone)]
pub struct PlBooleanArray {
    values: Bitmap,
    length: usize,
    validity: Option<Bitmap>,
}

impl PlBooleanArray {
    /// Creates a [`PlBooleanArray`] out of its internal components.
    ///
    /// This function is `O(1)`.
    ///
    /// # Errors
    /// This function errors if `values` or `validity` is neither dense (length equal to `length`)
    /// nor broadcast (length one).
    pub fn try_new(values: Bitmap, length: usize, validity: Option<Bitmap>) -> PolarsResult<Self> {
        polars_ensure!(
            is_valid_buffer_len(values.len(), length),
            ComputeError:
            "values bitmap of length {} is neither dense nor broadcast for an array of length {}",
            values.len(), length,
        );

        if let Some(validity) = validity.as_ref() {
            polars_ensure!(
                is_valid_buffer_len(validity.len(), length),
                ComputeError:
                "validity mask of length {} is neither dense nor broadcast for an array of length {}",
                validity.len(), length,
            );
        }

        Ok(Self {
            values,
            length,
            validity,
        })
    }

    /// Creates a [`PlBooleanArray`] out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(values: Bitmap, length: usize, validity: Option<Bitmap>) -> Self {
        Self::try_new(values, length, validity).unwrap()
    }

    /// Creates a [`PlBooleanArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// `values` and `validity` must each be either dense (length equal to `length`) or broadcast
    /// (length one).
    #[inline]
    pub unsafe fn new_unchecked(values: Bitmap, length: usize, validity: Option<Bitmap>) -> Self {
        if cfg!(debug_assertions) {
            assert!(is_valid_buffer_len(values.len(), length));
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_valid_buffer_len(v.len(), length))
            );
        }

        Self {
            values,
            length,
            validity,
        }
    }

    /// Creates an empty [`PlBooleanArray`].
    #[inline]
    pub fn new_empty() -> Self {
        Self {
            values: Bitmap::new(),
            length: 0,
            validity: None,
        }
    }

    /// Creates a dense, fully valid [`PlBooleanArray`] from `values`.
    #[inline]
    pub fn from_values(values: Bitmap) -> Self {
        let length = values.len();
        Self {
            values,
            length,
            validity: None,
        }
    }

    /// Creates a dense, fully valid [`PlBooleanArray`] from a [`Vec`].
    #[inline]
    pub fn from_vec(values: Vec<bool>) -> Self {
        Self::from_values(Bitmap::from(values))
    }

    /// Creates a dense, fully valid [`PlBooleanArray`] by packing `values`.
    #[inline]
    pub fn from_slice(values: &[bool]) -> Self {
        Self::from_values(Bitmap::from(values))
    }

    /// Creates a [`PlBooleanArray`] of `length` copies of `value`, in `O(1)` memory.
    #[inline]
    pub fn new_scalar(value: bool, length: usize) -> Self {
        Self {
            values: Bitmap::new_with_value(value, 1),
            length,
            validity: None,
        }
    }

    /// Creates a [`PlBooleanArray`] of `length` nulls, in `O(1)` memory.
    #[inline]
    pub fn new_full_null(length: usize) -> Self {
        Self {
            values: Bitmap::new_zeroed(1),
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

    /// The values, ignoring validity.
    ///
    /// The returned [`PlBitmapRef`] has [`Self::len`] bits regardless of whether the backing bitmap
    /// is dense or broadcast, so reading values through it needs no knowledge of which
    /// representation this array is in. Reach for the backing [`Bitmap`] — which is *not*
    /// guaranteed to have [`Self::len`] bits — with [`PlBitmapRef::bitmap`], or materialize a dense
    /// one with [`PlBitmapRef::to_dense`].
    #[inline]
    pub fn values(&self) -> PlBitmapRef<'_> {
        // SAFETY: the bitmap is dense or broadcast for `self.length`, upheld by every constructor.
        unsafe { PlBitmapRef::new_unchecked(&self.values, self.length) }
    }

    /// The validity mask, if any element may be null.
    ///
    /// The returned [`PlBitmapRef`] has [`Self::len`] bits regardless of whether the backing bitmap
    /// is dense or broadcast, exactly like [`Self::values`].
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is dense or broadcast for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_unchecked(validity, self.length) })
    }

    /// Whether the values bitmap holds a single bit shared by every element.
    ///
    /// This is `false` for a dense array of length one, where the two representations coincide.
    #[inline]
    pub fn values_are_broadcast(&self) -> bool {
        self.values.len() != self.length
    }

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    pub fn validity_is_broadcast(&self) -> bool {
        self.validity().is_some_and(|v| v.is_broadcast())
    }

    /// Whether every backing bitmap has one bit per element.
    #[inline]
    pub fn is_dense(&self) -> bool {
        !self.values_are_broadcast() && !self.validity_is_broadcast()
    }

    /// Whether this array is entirely stored in the broadcast representation, and therefore is a
    /// single logical value repeated [`Self::len`] times in `O(1)` memory.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.values_are_broadcast() && self.validity().is_none_or(|v| v.is_broadcast())
    }

    /// The value shared by every element, if the values bitmap is a broadcast bitmap.
    ///
    /// Returns `None` for a dense array and for an empty array. The value of a null element is
    /// undetermined, so this may return a value even when all elements are null.
    #[inline]
    pub fn broadcast_value(&self) -> Option<bool> {
        self.values().broadcast_value()
    }

    /// Returns the value at `i`.
    ///
    /// The value of a null element is undetermined (it can be anything).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value(&self, i: usize) -> bool {
        self.values().get(i)
    }

    /// Returns the value at `i`.
    ///
    /// The value of a null element is undetermined (it can be anything).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> bool {
        unsafe { self.values().get_unchecked(i) }
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
    pub fn get(&self, i: usize) -> Option<bool> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> Option<bool> {
        unsafe { self.is_valid_unchecked(i).then(|| self.value_unchecked(i)) }
    }

    /// The number of null elements.
    ///
    /// This is `O(1)` for a broadcast validity mask and `O(len)` for a dense one, amortized over
    /// repeated calls on the same [`Bitmap`].
    pub fn null_count(&self) -> usize {
        self.validity().map_or(0, |validity| validity.unset_bits())
    }

    /// Whether this array has at least one null element.
    #[inline]
    pub fn has_nulls(&self) -> bool {
        self.null_count() > 0
    }

    /// Returns an iterator over the values, ignoring validity.
    ///
    /// The values of null elements are undetermined (they can be anything).
    #[inline]
    pub fn values_iter(&self) -> PlBitmapIter<'_> {
        self.values().iter()
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlBooleanIter<'_> {
        PlBooleanIter::new(self.values(), self.validity(), self.length)
    }

    /// Replaces the validity mask.
    ///
    /// # Panics
    /// Panics if `validity` is neither dense nor broadcast for this array's length.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Replaces the validity mask.
    ///
    /// # Panics
    /// Panics if `validity` is neither dense nor broadcast for this array's length.
    pub fn set_validity(&mut self, validity: Option<Bitmap>) {
        if let Some(validity) = validity.as_ref() {
            assert!(
                is_valid_buffer_len(validity.len(), self.length),
                "validity mask of length {} is neither dense nor broadcast for an array of length {}",
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

        // Broadcast bitmaps are unaffected by slicing: every element reads the same bit.
        if !self.values_are_broadcast() {
            unsafe { self.values.slice_unchecked(offset, length) };
        }
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

    /// Returns an equivalent array whose backing bitmaps all hold one bit per element.
    ///
    /// This materializes any broadcast bitmap and is therefore `O(len)`; it is a no-op clone when
    /// this array [`is_dense`](Self::is_dense).
    pub fn to_dense(&self) -> Self {
        if self.is_dense() {
            return self.clone();
        }

        Self {
            values: self.values().to_dense(),
            length: self.length,
            validity: self.validity().map(|validity| validity.to_dense()),
        }
    }

    /// The single element every element of this array equals, if it is a non-empty scalar array.
    ///
    /// This is what lets equality and formatting avoid walking a scalar array of unbounded length.
    #[inline]
    fn scalar_element(&self) -> Option<Option<bool>> {
        (!self.is_empty() && self.is_scalar()).then(|| unsafe { self.get_unchecked(0) })
    }
}

impl Default for PlBooleanArray {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

impl From<Vec<bool>> for PlBooleanArray {
    #[inline]
    fn from(values: Vec<bool>) -> Self {
        Self::from_vec(values)
    }
}

impl From<Bitmap> for PlBooleanArray {
    #[inline]
    fn from(values: Bitmap) -> Self {
        Self::from_values(values)
    }
}

impl FromIterator<Option<bool>> for PlBooleanArray {
    fn from_iter<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut values = MutableBitmap::with_capacity(lower);
        let mut validity = MutableBitmap::with_capacity(lower);

        for item in iter {
            values.push(item.unwrap_or_default());
            validity.push(item.is_some());
        }

        let length = values.len();
        let validity = Bitmap::from(validity);
        let validity = (validity.unset_bits() > 0).then_some(validity);

        Self {
            values: values.into(),
            length,
            validity,
        }
    }
}

impl FromIterator<bool> for PlBooleanArray {
    #[inline]
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        Self::from_values(Bitmap::from_iter(iter))
    }
}

impl<'a> IntoIterator for &'a PlBooleanArray {
    type Item = Option<bool>;
    type IntoIter = PlBooleanIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares two arrays element-wise; the representation (dense or broadcast) is irrelevant.
impl PartialEq for PlBooleanArray {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }

        // Never walk two scalar arrays element by element: their length is unbounded by their
        // memory use.
        if let (Some(lhs), Some(rhs)) = (self.scalar_element(), other.scalar_element()) {
            return lhs == rhs;
        }

        self.iter().eq(other.iter())
    }
}

impl Eq for PlBooleanArray {}

impl std::fmt::Debug for PlBooleanArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// Renders nulls as `null` instead of `None`.
        struct Element(Option<bool>);

        impl std::fmt::Debug for Element {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match &self.0 {
                    Some(value) => value.fmt(f),
                    None => f.write_str("null"),
                }
            }
        }

        f.write_str("PlBooleanArray")?;

        // Never materialize a scalar array: its length is unbounded by its memory use.
        if self.length > 1 {
            if let Some(element) = self.scalar_element() {
                return write!(f, "[{:?}; {}]", Element(element), self.length);
            }
        }

        f.debug_list().entries(self.iter().map(Element)).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense() {
        let arr = PlBooleanArray::from_vec(vec![true, false, true]);

        assert_eq!(arr.len(), 3);
        assert!(arr.is_dense());
        assert!(!arr.is_scalar());
        assert_eq!(arr.null_count(), 0);
        assert!(!arr.value(1));
        assert_eq!(arr.get(2), Some(true));
        assert_eq!(
            arr.iter().collect::<Vec<_>>(),
            [Some(true), Some(false), Some(true)],
        );
        assert_eq!(arr.values_iter().collect::<Vec<_>>(), [true, false, true]);
    }

    #[test]
    fn scalar_broadcasts_values() {
        let arr = PlBooleanArray::new_scalar(true, 4);

        assert_eq!(arr.len(), 4);
        assert_eq!(arr.values().bitmap().len(), 1);
        assert_eq!(arr.values().len(), 4);
        assert!(arr.is_scalar());
        assert!(!arr.is_dense());
        assert_eq!(arr.broadcast_value(), Some(true));
        assert_eq!(arr.null_count(), 0);

        for i in 0..arr.len() {
            assert_eq!(arr.get(i), Some(true));
        }
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(true); 4]);
        assert_eq!(arr.values_iter().rev().collect::<Vec<_>>(), [true; 4]);
    }

    #[test]
    fn null_scalar() {
        let arr = PlBooleanArray::new_full_null(3);

        assert_eq!(arr.len(), 3);
        assert!(arr.is_scalar());
        assert_eq!(arr.null_count(), 3);
        assert!(arr.has_nulls());
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, None, None]);
    }

    #[test]
    fn dense_values_with_broadcast_validity() {
        let arr = PlBooleanArray::from_vec(vec![true, false, true])
            .with_validity(Some(Bitmap::new_zeroed(1)));

        assert!(arr.validity_is_broadcast());
        assert!(!arr.values_are_broadcast());
        assert!(!arr.is_dense());
        assert!(!arr.is_scalar());
        assert_eq!(arr.null_count(), 3);
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, None, None]);
    }

    #[test]
    fn broadcast_values_with_dense_validity() {
        let arr = PlBooleanArray::new_scalar(true, 3)
            .with_validity(Some(Bitmap::from_iter([true, false, true])));

        assert!(arr.values_are_broadcast());
        assert!(!arr.validity_is_broadcast());
        assert!(!arr.is_dense());
        assert!(!arr.is_scalar());
        assert_eq!(arr.null_count(), 1);
        assert_eq!(
            arr.iter().collect::<Vec<_>>(),
            [Some(true), None, Some(true)],
        );
    }

    #[test]
    fn values_hide_the_representation() {
        let scalar = PlBooleanArray::new_scalar(true, 1_000);
        let values = scalar.values();

        // The values cover every element even though they are backed by a single bit.
        assert_eq!(values.len(), 1_000);
        assert_eq!(values.bitmap().len(), 1);
        assert!(values.is_broadcast());
        assert_eq!(values.broadcast_value(), Some(true));
        assert!(values.get(999));
        assert_eq!(values.set_bits(), 1_000);

        // Materializing them yields exactly the bitmap a dense array would carry.
        assert_eq!(values.to_dense(), *scalar.to_dense().values().bitmap());
    }

    #[test]
    fn validity_of_a_fully_valid_array() {
        assert!(
            PlBooleanArray::from_vec(vec![true, false])
                .validity()
                .is_none()
        );
        assert!(PlBooleanArray::new_scalar(true, 1_000).validity().is_none());
    }

    #[test]
    fn from_iter_with_nulls() {
        let arr: PlBooleanArray = [Some(true), None, Some(false)].into_iter().collect();

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
        assert!(arr.is_valid(0));
        assert!(arr.is_null(1));
        assert_eq!(arr.get(1), None);
        assert_eq!(
            arr.iter().collect::<Vec<_>>(),
            [Some(true), None, Some(false)],
        );

        // A fully valid iterator carries no validity mask.
        let arr: PlBooleanArray = [Some(true), Some(false)].into_iter().collect();
        assert!(arr.validity().is_none());

        let arr: PlBooleanArray = [true, false, true].into_iter().collect();
        assert_eq!(arr.len(), 3);
        assert!(arr.is_dense());
        assert!(arr.validity().is_none());
    }

    #[test]
    fn slicing_a_scalar_is_free() {
        let arr = PlBooleanArray::new_scalar(true, 1_000).sliced(500, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.values().bitmap().len(), 1);
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(true), Some(true)]);
    }

    #[test]
    fn slicing_a_dense_array_slices_its_bitmaps() {
        let arr: PlBooleanArray = [Some(true), None, Some(false), Some(true)]
            .into_iter()
            .collect();
        let arr = arr.sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.values().bitmap().len(), 2);
        assert_eq!(arr.validity().unwrap().bitmap().len(), 2);
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, Some(false)]);
    }

    #[test]
    fn slicing_keeps_broadcast_validity() {
        let arr = PlBooleanArray::from_vec(vec![true, false, true])
            .with_validity(Some(Bitmap::new_zeroed(1)))
            .sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.values().bitmap().len(), 2);
        assert_eq!(arr.validity().unwrap().len(), 2);
        assert_eq!(arr.validity().unwrap().bitmap().len(), 1);
        assert!(arr.validity().unwrap().is_broadcast());
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, None]);
    }

    #[test]
    fn to_dense_materializes_broadcasts() {
        let scalar = PlBooleanArray::new_scalar(true, 3);
        let dense = scalar.to_dense();

        assert!(dense.is_dense());
        assert_eq!(dense.values().bitmap().len(), 3);
        assert_eq!(dense.values_iter().collect::<Vec<_>>(), [true; 3]);
        assert_eq!(dense, scalar);

        let null_scalar = PlBooleanArray::new_full_null(3);
        let dense = null_scalar.to_dense();

        assert!(dense.is_dense());
        assert_eq!(dense.validity().unwrap().bitmap().len(), 3);
        assert!(dense.validity().unwrap().is_dense());
        assert_eq!(dense.null_count(), 3);
        assert_eq!(dense, null_scalar);
    }

    #[test]
    fn to_dense_of_empty_scalar() {
        let dense = PlBooleanArray::new_scalar(true, 0).to_dense();

        assert!(dense.is_dense());
        assert!(dense.is_empty());
        assert_eq!(dense.values().bitmap().len(), 0);
    }

    #[test]
    fn equality_ignores_representation() {
        let scalar = PlBooleanArray::new_scalar(true, 3);
        let dense = PlBooleanArray::from_vec(vec![true, true, true]);

        assert_eq!(scalar, dense);
        assert_ne!(scalar, PlBooleanArray::new_scalar(true, 4));
        assert_ne!(scalar, PlBooleanArray::from_vec(vec![true, true, false]));
        assert_ne!(scalar, PlBooleanArray::new_full_null(3));
    }

    #[test]
    fn equality_of_scalars_does_not_walk_elements() {
        // Element-by-element comparison of a billion elements would not finish; the fast path must
        // hit.
        let arr = PlBooleanArray::new_scalar(true, 1_000_000_000);

        assert_eq!(arr, arr.clone());
        assert_ne!(arr, PlBooleanArray::new_scalar(false, 1_000_000_000));
        assert_ne!(arr, PlBooleanArray::new_full_null(1_000_000_000));
        assert_eq!(
            PlBooleanArray::new_full_null(1_000_000_000),
            PlBooleanArray::new_full_null(1_000_000_000),
        );
    }

    #[test]
    fn empty() {
        let arr = PlBooleanArray::new_empty();

        assert!(arr.is_empty());
        assert!(arr.is_dense());
        assert_eq!(arr.null_count(), 0);
        assert_eq!(arr.broadcast_value(), None);
        assert_eq!(arr.iter().next(), None);
    }

    #[test]
    fn try_new_rejects_mismatched_bitmaps() {
        assert!(PlBooleanArray::try_new(Bitmap::new_zeroed(2), 3, None).is_err());
        assert!(
            PlBooleanArray::try_new(Bitmap::new_zeroed(1), 3, Some(Bitmap::new_zeroed(2))).is_err()
        );
        assert!(
            PlBooleanArray::try_new(Bitmap::new_zeroed(1), 3, Some(Bitmap::new_zeroed(3))).is_ok()
        );
    }

    #[test]
    fn iterators_are_exact_sized() {
        let arr = PlBooleanArray::new_scalar(true, 5);

        assert_eq!(arr.iter().len(), 5);
        assert_eq!(arr.values_iter().len(), 5);
        assert_eq!(arr.iter().size_hint(), (5, Some(5)));
    }

    #[test]
    fn debug_does_not_materialize_scalars() {
        let arr = PlBooleanArray::new_scalar(true, 1_000_000_000);
        assert_eq!(format!("{arr:?}"), "PlBooleanArray[true; 1000000000]");

        let arr = PlBooleanArray::new_full_null(1_000_000_000);
        assert_eq!(format!("{arr:?}"), "PlBooleanArray[null; 1000000000]");

        let arr: PlBooleanArray = [Some(true), None].into_iter().collect();
        assert_eq!(format!("{arr:?}"), "PlBooleanArray[true, null]");
    }
}
