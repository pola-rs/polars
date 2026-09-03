use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::PlBitmapRef;
use crate::broadcast::{
    assert_broadcastable, broadcast_index, is_flat_buffer_len, is_scalar_buffer_len,
    is_valid_buffer_len, scalar_buffer_len,
};
use crate::flat::Flat;

mod builder;
mod flat;
mod iterator;

pub use builder::PlPrimitiveArrayBuilder;
pub use iterator::{PlPrimitiveIter, PlPrimitiveValuesIter};

/// An immutable, cheaply cloneable sequence of `length` optional [`NativeType`] values.
#[derive(Clone)]
pub struct PlPrimitiveArray<T: NativeType> {
    /// Scalar: values.len() == 1
    values: Buffer<T>,
    length: usize,
    /// Scalar: validity.len() == 1
    validity: Option<Bitmap>,
}

impl<T: NativeType> PlPrimitiveArray<T> {
    /// Creates a flat [`PlPrimitiveArray`] out of its internal components.
    ///
    /// # Errors
    /// This function errors if `values` or `validity` does not hold exactly `length` slots.
    pub fn try_new(
        values: Buffer<T>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_flat_buffer_len(values.len(), length),
            ComputeError:
            "values buffer of length {} is not flat for an array of length {}",
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

    /// Creates a flat [`PlPrimitiveArray`] out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(values: Buffer<T>, length: usize, validity: Option<Bitmap>) -> Self {
        Self::try_new(values, length, validity).unwrap()
    }

    /// Creates a flat [`PlPrimitiveArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// `values` and `validity` must each hold exactly `length` slots.
    #[inline]
    pub unsafe fn new_unchecked(
        values: Buffer<T>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
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

    /// Creates a scalar [`PlPrimitiveArray`] of `length` elements out of its internal components.
    ///
    /// # Errors
    /// This function errors if `values` or `validity` does not hold exactly one slot.
    pub fn try_new_broadcast(
        values: Buffer<T>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_scalar_buffer_len(values.len(), length),
            ComputeError:
            "values buffer of length {} is not the single value the {} elements of a broadcast \
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
            values,
            length,
            validity,
        })
    }

    /// Creates a scalar [`PlPrimitiveArray`] of `length` elements out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new_broadcast`] errors.
    #[inline]
    pub fn new_broadcast(values: Buffer<T>, length: usize, validity: Option<Bitmap>) -> Self {
        Self::try_new_broadcast(values, length, validity).unwrap()
    }

    /// Creates a scalar [`PlPrimitiveArray`] of `length` elements out of its internal components
    /// without validating them.
    ///
    /// # Safety
    /// `values` and `validity` must each hold exactly one slot, or none at all if `length` is zero.
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        values: Buffer<T>,
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
            values,
            length,
            validity,
        }
    }

    /// Creates an empty [`PlPrimitiveArray`].
    #[inline]
    pub fn new_empty() -> Self {
        Self {
            values: Buffer::new(),
            length: 0,
            validity: None,
        }
    }

    /// Creates a flat, fully valid [`PlPrimitiveArray`] from `values`.
    #[inline]
    pub fn from_values(values: Buffer<T>) -> Self {
        let length = values.len();
        Self {
            values,
            length,
            validity: None,
        }
    }

    /// Creates a flat, fully valid [`PlPrimitiveArray`] from a [`Vec`].
    #[inline]
    pub fn from_vec(values: Vec<T>) -> Self {
        Self::from_values(Buffer::from(values))
    }

    /// Creates a flat, fully valid [`PlPrimitiveArray`] by copying `values`.
    #[inline]
    pub fn from_slice(values: &[T]) -> Self {
        Self::from_vec(values.to_vec())
    }

    /// Creates a [`PlPrimitiveArray`] of `length` copies of `value`, in `O(1)` memory.
    #[inline]
    pub fn new_scalar(value: T, length: usize) -> Self {
        // There is no element for the value to be shared by when there are no elements at all,
        // which is why an empty array is the one that keeps nothing of the value it repeats.
        let values = if length == 0 {
            Buffer::new()
        } else {
            Buffer::from_owner([value])
        };

        Self {
            values,
            length,
            validity: None,
        }
    }

    /// Creates a [`PlPrimitiveArray`] of `length` nulls, in `O(1)` memory.
    #[inline]
    pub fn new_full_null(length: usize) -> Self {
        Self {
            values: Buffer::zeroed(scalar_buffer_len(length)),
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

    /// The backing values buffer, if it holds one slot per element.
    #[inline]
    pub fn flat_values(&self) -> Option<&Buffer<T>> {
        self.values_are_flat().then_some(&self.values)
    }

    /// The values buffer, if this array holds one slot per element and nothing else shares it.
    #[inline]
    pub fn flat_values_mut(&mut self) -> Option<&mut Buffer<T>> {
        self.values_are_flat().then_some(&mut self.values)
    }

    /// The value every element of this array reads, if the values buffer holds a single slot.
    #[inline]
    pub fn scalar_values(&self) -> Option<T> {
        (self.values_are_scalar() && self.length > 0).then(|| self.values[0])
    }

    /// The validity mask, if any element may be null.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_broadcast_unchecked(validity, self.length) })
    }

    /// Whether the values buffer holds a single value shared by every element.
    #[inline]
    pub fn values_are_scalar(&self) -> bool {
        self.values.len() == 1
    }

    /// Whether the values buffer holds one slot per element.
    #[inline]
    pub fn values_are_flat(&self) -> bool {
        self.values.len() == self.length
    }

    /// Whether the validity mask holds a single value shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether every backing buffer has one slot per element.
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

    /// The single element every element of this array equals, if every backing buffer holds one
    /// slot.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<T>> {
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
    pub fn value(&self, i: usize) -> T {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the value at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> T {
        debug_assert!(i < self.length);
        unsafe {
            *self
                .values
                .get_unchecked(broadcast_index(i, self.values.len()))
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
    pub fn get(&self, i: usize) -> Option<T> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> Option<T> {
        unsafe { self.is_valid_unchecked(i).then(|| self.value_unchecked(i)) }
    }

    /// The number of null elements.
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
    pub fn values_iter(&self) -> PlPrimitiveValuesIter<'_, T> {
        PlPrimitiveValuesIter::new(&self.values, self.length)
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlPrimitiveIter<'_, T> {
        PlPrimitiveIter::new(&self.values, self.validity(), self.length)
    }

    /// Returns an iterator over `length` values, repeating the single value of this array if that
    /// is all it holds, and ignoring validity.
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlPrimitiveValuesIter<'_, T> {
        assert_broadcastable(self.length, length);
        // SAFETY: an array of one element holds a single slot, which is scalar for any length;
        // otherwise `length` is the length the values are already valid for.
        PlPrimitiveValuesIter::new(&self.values, length)
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

        // Scalar buffers are unaffected by slicing — every element reads the same slot — with the
        // one exception of an empty slice, which keeps no element to read it.
        if self.values_are_flat() {
            unsafe {
                self.values
                    .slice_in_place_unchecked(offset..offset + length)
            };
        } else if length == 0 {
            unsafe { self.values.slice_in_place_unchecked(0..0) };
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

    /// Creates a [`PlPrimitiveArray`] of `length` copies of the element at `index`.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        assert!(index < self.length, "index out of bounds");
        unsafe { self.new_from_index_unchecked(index, length) }
    }

    /// Creates a [`PlPrimitiveArray`] of `length` copies of the element at `index`.
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

    /// Returns an equivalent array whose backing buffers all hold one slot per element.
    pub fn to_flat(&self) -> Flat<Self> {
        if self.is_flat() {
            return Flat(self.clone());
        }

        let values = if self.values_are_flat() {
            self.values.clone()
        } else if self.length == 0 {
            Buffer::new()
        } else if self.scalar_value() == Some(None) {
            // Every element is null, and the value of a null element is undetermined, so the
            // repeated value need not be written out: a zeroed buffer stands in for it.
            Buffer::zeroed(self.length)
        } else {
            Buffer::from(vec![self.values[0]; self.length])
        };

        let validity = self.validity().map(|validity| validity.to_flat());

        Flat(Self {
            values,
            length: self.length,
            validity,
        })
    }

    /// Borrows this array as a [`Flat`] one, if every backing buffer already holds one slot per
    /// element.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: every backing buffer of a flat array holds one slot per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }
}

impl<T: NativeType> Default for PlPrimitiveArray<T> {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

impl<T: NativeType> From<Vec<T>> for PlPrimitiveArray<T> {
    #[inline]
    fn from(values: Vec<T>) -> Self {
        Self::from_vec(values)
    }
}

impl<T: NativeType> From<Buffer<T>> for PlPrimitiveArray<T> {
    #[inline]
    fn from(values: Buffer<T>) -> Self {
        Self::from_values(values)
    }
}

impl<T: NativeType> FromIterator<Option<T>> for PlPrimitiveArray<T> {
    fn from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut values = Vec::with_capacity(lower);
        let mut validity = MutableBitmap::with_capacity(lower);

        for item in iter {
            values.push(item.unwrap_or_default());
            validity.push(item.is_some());
        }

        let length = values.len();
        let validity = Bitmap::from(validity);
        let validity = (validity.unset_bits() > 0).then_some(validity);

        Self {
            values: Buffer::from(values),
            length,
            validity,
        }
    }
}

impl<T: NativeType> FromIterator<T> for PlPrimitiveArray<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }
}

impl<'a, T: NativeType> IntoIterator for &'a PlPrimitiveArray<T> {
    type Item = Option<T>;
    type IntoIter = PlPrimitiveIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares two arrays element-wise; the representation (flat or scalar) is irrelevant.
impl<T: NativeType> PartialEq for PlPrimitiveArray<T> {
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

impl<T: NativeType> std::fmt::Debug for PlPrimitiveArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// Renders nulls as `null` instead of `None`.
        struct Element<T>(Option<T>);

        impl<T: std::fmt::Debug> std::fmt::Debug for Element<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match &self.0 {
                    Some(value) => value.fmt(f),
                    None => f.write_str("null"),
                }
            }
        }

        f.write_str("PlPrimitiveArray")?;

        // Never materialize a scalar array: its length is unbounded by its memory use.
        if self.length > 1 {
            if let Some(element) = self.scalar_value() {
                return write!(f, "[{:?}; {}]", Element(element), self.length);
            }
        }

        f.debug_list().entries(self.iter().map(Element)).finish()
    }
}

impl<T: NativeType> PlArray for PlPrimitiveArray<T> {
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
        PlArrayType::Primitive(T::PRIMITIVE)
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
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);

        assert_eq!(arr.len(), 3);
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert_eq!(arr.null_count(), 0);
        assert_eq!(arr.value(1), 2);
        assert_eq!(arr.get(2), Some(3));
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(1), Some(2), Some(3)]);
        assert_eq!(arr.values_iter().collect::<Vec<_>>(), [1, 2, 3]);
    }

    #[test]
    fn scalar_scalars_values() {
        let arr = PlPrimitiveArray::new_scalar(7i32, 4);

        assert_eq!(arr.len(), 4);
        assert!(arr.flat_values().is_none());
        assert_eq!(arr.scalar_values(), Some(7));
        assert!(arr.is_scalar());
        assert!(!arr.is_flat());
        assert_eq!(arr.scalar_value(), Some(Some(7)));
        assert_eq!(arr.null_count(), 0);

        for i in 0..arr.len() {
            assert_eq!(arr.get(i), Some(7));
        }
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(7); 4]);
        assert_eq!(arr.values_iter().rev().collect::<Vec<_>>(), [7; 4]);
    }

    #[test]
    fn slicing_a_flat_array_slices_its_buffers() {
        let arr: PlPrimitiveArray<i32> = [Some(1), None, Some(3), Some(4)].into_iter().collect();
        let arr = arr.sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.flat_values().unwrap().len(), 2);
        assert_eq!(arr.validity().unwrap().len(), 2);
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, Some(3)]);
    }
}
