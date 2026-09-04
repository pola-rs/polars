use std::borrow::Cow;

use arrow::bitmap::{Bitmap, MutableBitmap};
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmap, PlBitmapIter, PlBitmapRef};
use crate::broadcast::{
    ArrayRepr, is_flat_buffer_len, is_scalar_buffer_len, is_valid_buffer_len, normalize_bitmap,
    normalize_validity, scalar_buffer_len, slice_bitmap, slice_validity,
};
use crate::flat::Flat;

mod builder;
mod flat;
mod iterator;

pub use builder::PlBooleanArrayBuilder;
pub use iterator::PlBooleanIter;

/// An immutable, cheaply cloneable sequence of `length` optional [`bool`] values.
#[derive(Clone)]
pub struct PlBooleanArray {
    /// Scalar: values.len() == 1
    values: Bitmap,
    length: usize,
    /// Scalar: validity.len() == 1
    validity: Option<Bitmap>,
}

impl PlBooleanArray {
    /// Creates a flat [`PlBooleanArray`] out of its internal components.
    ///
    /// # Errors
    /// This function errors if `values` or `validity` does not hold exactly `length` bits.
    pub fn try_new(values: Bitmap, length: usize, validity: Option<Bitmap>) -> PolarsResult<Self> {
        polars_ensure!(
            is_flat_buffer_len(values.len(), length),
            ComputeError:
            "values bitmap of length {} is not flat for an array of length {}",
            values.len(), length,
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
            length,
            validity,
        })
    }

    /// Creates a flat [`PlBooleanArray`] out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(values: Bitmap, length: usize, validity: Option<Bitmap>) -> Self {
        Self::try_new(values, length, validity).unwrap()
    }

    /// Creates a flat [`PlBooleanArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// `values` and `validity` must each hold exactly `length` bits.
    #[inline]
    pub unsafe fn new_unchecked(values: Bitmap, length: usize, validity: Option<Bitmap>) -> Self {
        if cfg!(debug_assertions) {
            assert!(is_flat_buffer_len(values.len(), length));
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_flat_buffer_len(v.len(), length))
            );
        }

        Self {
            values,
            length,
            validity,
        }
    }

    /// Creates a scalar [`PlBooleanArray`] of `length` elements out of its internal components.
    ///
    /// # Errors
    /// This function errors if `values` or `validity` is not scalar for `length`, per
    /// [`is_scalar_buffer_len`].
    pub fn try_new_broadcast(
        values: Bitmap,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_scalar_buffer_len(values.len(), length),
            ComputeError:
            "values bitmap of length {} is not the single bit the {} elements of a broadcast \
             array share",
            values.len(), length,
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
            values: normalize_bitmap(values, length),
            length,
            validity: normalize_validity(validity, length),
        })
    }

    /// Creates a scalar [`PlBooleanArray`] of `length` elements out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new_broadcast`] errors.
    #[inline]
    pub fn new_broadcast(values: Bitmap, length: usize, validity: Option<Bitmap>) -> Self {
        Self::try_new_broadcast(values, length, validity).unwrap()
    }

    /// Creates a scalar [`PlBooleanArray`] of `length` elements out of its internal components
    /// without validating them.
    ///
    /// # Safety
    /// `values` and `validity` must each be scalar for `length`, per [`is_scalar_buffer_len`].
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        values: Bitmap,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        if cfg!(debug_assertions) {
            assert!(is_scalar_buffer_len(values.len(), length));
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_scalar_buffer_len(v.len(), length))
            );
        }

        Self {
            values: normalize_bitmap(values, length),
            length,
            validity: normalize_validity(validity, length),
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

    /// Creates a flat, fully valid [`PlBooleanArray`] from `values`.
    #[inline]
    pub fn from_values(values: Bitmap) -> Self {
        let length = values.len();
        Self {
            values,
            length,
            validity: None,
        }
    }

    /// Creates a flat, fully valid [`PlBooleanArray`] from a [`Vec`].
    #[inline]
    pub fn from_vec(values: Vec<bool>) -> Self {
        Self::from_values(Bitmap::from(values))
    }

    /// Creates a flat, fully valid [`PlBooleanArray`] by packing `values`.
    #[inline]
    pub fn from_slice(values: &[bool]) -> Self {
        Self::from_values(Bitmap::from(values))
    }

    /// Creates a fully valid [`PlBooleanArray`] whose values are the bits of `values`, in whatever
    /// representation that mask is in.
    #[inline]
    pub fn from_pl_bitmap(values: PlBitmap) -> Self {
        let (values, length) = values.into_inner();
        Self {
            values,
            length,
            validity: None,
        }
    }

    /// Creates a [`PlBooleanArray`] of `length` copies of `value`, in `O(1)` memory.
    #[inline]
    pub fn new_scalar(value: bool, length: usize) -> Self {
        Self {
            values: Bitmap::new_with_value(value, scalar_buffer_len(length)),
            length,
            validity: None,
        }
    }

    /// Creates a [`PlBooleanArray`] of `length` nulls, in `O(1)` memory.
    #[inline]
    pub fn new_full_null(length: usize) -> Self {
        Self {
            values: Bitmap::new_zeroed(scalar_buffer_len(length)),
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

    /// The values, ignoring validity.
    #[inline]
    pub fn values(&self) -> PlBitmapRef<'_> {
        // SAFETY: the bitmap is flat or scalar for `self.length`, upheld by every constructor.
        unsafe { PlBitmapRef::new_broadcast_unchecked(&self.values, self.length) }
    }

    /// Which representation the backing values bitmap is in, along with what it holds.
    #[inline]
    pub fn values_repr(&self) -> ArrayRepr<&Bitmap, bool> {
        if self.values_are_scalar() {
            // SAFETY: the bitmap holds a single bit, so bit 0 is in bounds.
            ArrayRepr::Scalar(unsafe { self.values.get_bit_unchecked(0) })
        } else {
            ArrayRepr::Flat(&self.values)
        }
    }

    /// The backing values bitmap, if it holds one bit per element.
    #[inline]
    pub fn flat_values(&self) -> Option<&Bitmap> {
        self.values_repr().flat()
    }

    /// The value every element of this array reads, if the values bitmap holds a single bit.
    #[inline]
    pub fn scalar_values(&self) -> Option<bool> {
        self.values_repr().scalar()
    }

    /// The validity mask, if any element may be null.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_broadcast_unchecked(validity, self.length) })
    }

    /// Whether the values bitmap holds a single bit shared by every element.
    ///
    /// An array of no elements holds no such bit: it keeps the empty bitmap in place of the one
    /// bit a scalar bitmap would, and is flat.
    #[inline]
    pub fn values_are_scalar(&self) -> bool {
        self.values.len() == 1 && self.length > 0
    }

    /// Whether the values bitmap holds one bit per element.
    #[inline]
    pub fn values_are_flat(&self) -> bool {
        self.values.len() == self.length
    }

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether every backing bitmap has one bit per element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.values_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array is entirely stored in the scalar representation, and therefore is a
    /// single logical value repeated [`Self::len`] times in `O(1)` memory.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.values_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element of this array equals, if every backing bitmap holds one
    /// bit.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<bool>> {
        let is_shared = self.values.len() == 1
            && self
                .validity
                .as_ref()
                .is_none_or(|validity| validity.len() == 1);

        // SAFETY: the array is not empty, so element 0 is in bounds.
        (is_shared && self.length > 0).then(|| unsafe { self.get_unchecked(0) })
    }

    /// Returns the value at `i`.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value(&self, i: usize) -> bool {
        self.values().get(i)
    }

    /// Returns the value at `i`.
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

    /// Returns an iterator over the values, ignoring validity.
    #[inline]
    pub fn values_iter(&self) -> PlBitmapIter<'_> {
        self.values().iter()
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlBooleanIter<'_> {
        PlBooleanIter::new(self.values(), self.validity(), self.length)
    }

    /// Returns an iterator over `length` values, repeating the single value of this array if that
    /// is all it holds, and ignoring validity.
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlBitmapIter<'_> {
        self.values().broadcast(length).iter()
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

        unsafe {
            slice_bitmap(&mut self.values, self.length, offset, length);
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

    /// Creates a [`PlBooleanArray`] of `length` copies of the element at `index`.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        assert!(index < self.length, "index out of bounds");
        unsafe { self.new_from_index_unchecked(index, length) }
    }

    /// Creates a [`PlBooleanArray`] of `length` copies of the element at `index`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        if unsafe { self.is_null_unchecked(index) } {
            return Self::new_full_null(length);
        }

        // The value of a null element is undetermined, so it is repeated as it is found: it is the
        // mask that makes every element of the result null.
        let value = unsafe { self.value_unchecked(index) };

        Self::new_scalar(value, length)
    }

    /// Returns an equivalent array whose backing bitmaps all hold one bit per element, borrowing
    /// this array itself if they already do.
    pub fn to_flat(&self) -> Cow<'_, Flat<Self>> {
        if let Some(flat) = self.as_flat() {
            return Cow::Borrowed(flat);
        }

        let values = if self.values_are_scalar() && self.scalar_value() == Some(None) {
            // Every element is null, and the value of a null element is undetermined, so the
            // repeated bit need not be written out: a zeroed bitmap stands in for it.
            Bitmap::new_zeroed(self.length)
        } else {
            self.values().to_flat().into_owned()
        };

        Cow::Owned(Flat(Self {
            values,
            length: self.length,
            validity: self
                .validity()
                .map(|validity| validity.to_flat().into_owned()),
        }))
    }

    /// Borrows this array as a [`Flat`] one, if every backing bitmap already holds one bit per
    /// element.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: every backing bitmap of a flat array holds one bit per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
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

/// Compares two arrays element-wise; the representation (flat or scalar) is irrelevant.
impl PartialEq for PlBooleanArray {
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
            if let Some(element) = self.scalar_value() {
                return write!(f, "[{:?}; {}]", Element(element), self.length);
            }
        }

        f.debug_list().entries(self.iter().map(Element)).finish()
    }
}

impl PlArray for PlBooleanArray {
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
        PlArrayType::Boolean
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

    #[test]
    fn flat() {
        let arr = PlBooleanArray::from_vec(vec![true, false, true]);

        assert_eq!(arr.len(), 3);
        assert!(arr.is_flat());
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
    fn scalar_scalars_values() {
        let arr = PlBooleanArray::new_scalar(true, 4);

        assert_eq!(arr.len(), 4);
        assert!(arr.values_are_scalar());
        assert_eq!(arr.values().len(), 4);
        assert!(arr.is_scalar());
        assert!(!arr.is_flat());
        assert_eq!(arr.scalar_value(), Some(Some(true)));
        assert_eq!(arr.null_count(), 0);

        for i in 0..arr.len() {
            assert_eq!(arr.get(i), Some(true));
        }
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(true); 4]);
        assert_eq!(arr.values_iter().rev().collect::<Vec<_>>(), [true; 4]);
    }

    #[test]
    fn slicing_a_flat_array_slices_its_bitmaps() {
        let arr: PlBooleanArray = [Some(true), None, Some(false), Some(true)]
            .into_iter()
            .collect();
        let arr = arr.sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.flat_values().unwrap().len(), 2);
        assert_eq!(arr.validity().unwrap().flat_bitmap().unwrap().len(), 2);
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, Some(false)]);
    }

    #[test]
    fn an_array_of_no_elements_keeps_no_bit() {
        // A single slot is scalar for no elements too, but there is no element left to read it, so
        // it is not kept: the array is flat, like every empty array, rather than scalar.
        let arr =
            PlBooleanArray::new_broadcast(Bitmap::new_zeroed(1), 0, Some(Bitmap::new_zeroed(1)));

        assert!(arr.is_empty());
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert!(arr.flat_values().unwrap().is_empty());
        assert!(arr.validity().unwrap().is_empty());

        // The same goes for a mask broadcast over an empty array after the fact.
        let arr = PlBooleanArray::new_empty().with_validity_broadcast(Some(Bitmap::new_zeroed(1)));

        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert!(arr.validity().unwrap().is_empty());
    }
}
