use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::PlBitmapRef;
use crate::broadcast::{broadcast_index, is_valid_buffer_len};

mod iterator;

pub use iterator::{PlPrimitiveIter, PlPrimitiveValuesIter};

/// An immutable, cheaply cloneable sequence of `length` optional [`NativeType`] values.
///
/// This is the lowest-level array in the Polars vector format. It carries no logical type — only
/// the physical values and their validity.
///
/// The logical length is stored separately from the backing buffers, which lets a *scalar* array —
/// one value repeated `length` times — be represented in `O(1)` memory. Element `i` reads slot
/// [`broadcast_index(i, buf.len())`](crate::broadcast::broadcast_index) of each backing buffer, so
/// both `values` and `validity` are independently either flat (one slot per element) or scalar
/// (a single shared slot). See [`crate::broadcast`] for the full rules.
///
/// # Example
/// ```
/// use polars_array::PlPrimitiveArray;
///
/// let flat = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);
/// assert_eq!(flat.len(), 3);
/// assert_eq!(flat.iter().collect::<Vec<_>>(), [Some(1), Some(2), Some(3)]);
///
/// // A scalar array of a billion elements costs a single `i32` of memory.
/// let scalar = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);
/// assert_eq!(scalar.len(), 1_000_000_000);
/// assert_eq!(scalar.values().len(), 1);
/// assert_eq!(scalar.value(999_999_999), 7);
/// ```
#[derive(Clone)]
pub struct PlPrimitiveArray<T: NativeType> {
    values: Buffer<T>,
    length: usize,
    validity: Option<Bitmap>,
}

impl<T: NativeType> PlPrimitiveArray<T> {
    /// Creates a [`PlPrimitiveArray`] out of its internal components.
    ///
    /// This function is `O(1)`.
    ///
    /// # Errors
    /// This function errors if `values` or `validity` is neither flat (length equal to `length`)
    /// nor scalar (length one).
    pub fn try_new(
        values: Buffer<T>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_valid_buffer_len(values.len(), length),
            ComputeError:
            "values buffer of length {} is neither flat nor scalar for an array of length {}",
            values.len(), length,
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
            length,
            validity,
        })
    }

    /// Creates a [`PlPrimitiveArray`] out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(values: Buffer<T>, length: usize, validity: Option<Bitmap>) -> Self {
        Self::try_new(values, length, validity).unwrap()
    }

    /// Creates a [`PlPrimitiveArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// `values` and `validity` must each be either flat (length equal to `length`) or scalar
    /// (length one).
    #[inline]
    pub unsafe fn new_unchecked(
        values: Buffer<T>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
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
        Self {
            values: Buffer::from_owner([value]),
            length,
            validity: None,
        }
    }

    /// Creates a [`PlPrimitiveArray`] of `length` nulls, in `O(1)` memory.
    #[inline]
    pub fn new_full_null(length: usize) -> Self {
        Self {
            values: Buffer::zeroed(1),
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

    /// The backing values buffer.
    ///
    /// This is *not* guaranteed to have [`Self::len`] elements: it is either flat or scalar.
    /// Index it through [`broadcast_index`](crate::broadcast::broadcast_index), or call
    /// [`Self::to_flat`] first.
    #[inline(always)]
    pub const fn values(&self) -> &Buffer<T> {
        &self.values
    }

    /// The validity mask, if any element may be null.
    ///
    /// The returned [`PlBitmapRef`] has [`Self::len`] bits regardless of whether the backing
    /// bitmap is flat or scalar, so reading validity through it needs no knowledge of which
    /// representation this array is in. Reach for the backing [`Bitmap`] with
    /// [`PlBitmapRef::bitmap`], or materialize a flat one with [`PlBitmapRef::to_flat`].
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_unchecked(validity, self.length) })
    }

    /// Whether the values buffer holds a single value shared by every element.
    ///
    /// This is `false` for a flat array of length one, where the two representations coincide.
    #[inline]
    pub fn values_are_scalar(&self) -> bool {
        self.values.len() != self.length
    }

    /// Whether the validity mask holds a single value shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether every backing buffer has one slot per element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        !self.values_are_scalar() && !self.validity_is_scalar()
    }

    /// Whether this array is entirely stored in the scalar representation, and therefore is a
    /// single logical value repeated [`Self::len`] times in `O(1)` memory.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.values_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The value shared by every element, if the values buffer is a scalar buffer.
    ///
    /// Returns `None` for a flat array and for an empty array. The value of a null element is
    /// undetermined, so this may return a value even when all elements are null.
    #[inline]
    pub fn scalar_value(&self) -> Option<T> {
        (self.values.len() == 1 && self.length > 0).then(|| self.values[0])
    }

    /// Returns the value at `i`.
    ///
    /// The value of a null element is undetermined (it can be anything).
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
    /// The value of a null element is undetermined (it can be anything).
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

    /// Returns an iterator over the values, ignoring validity.
    ///
    /// The values of null elements are undetermined (they can be anything).
    #[inline]
    pub fn values_iter(&self) -> PlPrimitiveValuesIter<'_, T> {
        PlPrimitiveValuesIter::new(&self.values, self.length)
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlPrimitiveIter<'_, T> {
        PlPrimitiveIter::new(&self.values, self.validity(), self.length)
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

        // Scalar buffers are unaffected by slicing: every element reads the same slot.
        if !self.values_are_scalar() {
            unsafe {
                self.values
                    .slice_in_place_unchecked(offset..offset + length)
            };
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

    /// Returns an equivalent array whose backing buffers all hold one slot per element.
    ///
    /// This materializes any scalar buffer and is therefore `O(len)`; it is a no-op clone when
    /// this array [`is_flat`](Self::is_flat).
    pub fn to_flat(&self) -> Self {
        if self.is_flat() {
            return self.clone();
        }

        let values = if !self.values_are_scalar() {
            self.values.clone()
        } else if self.length == 0 {
            Buffer::new()
        } else {
            Buffer::from(vec![self.values[0]; self.length])
        };

        let validity = self.validity().map(|validity| validity.to_flat());

        Self {
            values,
            length: self.length,
            validity,
        }
    }

    /// The single element every element of this array equals, if it is a non-empty scalar array.
    ///
    /// This is what lets equality and formatting avoid walking a scalar array of unbounded length.
    #[inline]
    fn scalar_element(&self) -> Option<Option<T>> {
        (!self.is_empty() && self.is_scalar()).then(|| unsafe { self.get_unchecked(0) })
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
        if let (Some(lhs), Some(rhs)) = (self.scalar_element(), other.scalar_element()) {
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
            if let Some(element) = self.scalar_element() {
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
    fn validity(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    #[inline]
    fn values_are_scalar(&self) -> bool {
        self.values_are_scalar()
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
    fn to_flat_boxed(&self) -> Box<dyn PlArray> {
        Box::new(self.to_flat())
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
        assert_eq!(arr.values().len(), 1);
        assert!(arr.is_scalar());
        assert!(!arr.is_flat());
        assert_eq!(arr.scalar_value(), Some(7));
        assert_eq!(arr.null_count(), 0);

        for i in 0..arr.len() {
            assert_eq!(arr.get(i), Some(7));
        }
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(7); 4]);
        assert_eq!(arr.values_iter().rev().collect::<Vec<_>>(), [7; 4]);
    }

    #[test]
    fn null_scalar() {
        let arr = PlPrimitiveArray::<i32>::new_full_null(3);

        assert_eq!(arr.len(), 3);
        assert!(arr.is_scalar());
        assert_eq!(arr.null_count(), 3);
        assert!(arr.has_nulls());
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, None, None]);
    }

    #[test]
    fn flat_values_with_scalar_validity() {
        let arr =
            PlPrimitiveArray::from_vec(vec![1i32, 2, 3]).with_validity(Some(Bitmap::new_zeroed(1)));

        assert!(arr.validity_is_scalar());
        assert!(!arr.values_are_scalar());
        assert!(!arr.is_flat());
        assert!(!arr.is_scalar());
        assert_eq!(arr.null_count(), 3);
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, None, None]);
    }

    #[test]
    fn validity_hides_the_representation() {
        let scalar = PlPrimitiveArray::<i32>::new_full_null(1_000);
        let validity = scalar.validity().unwrap();

        // The mask covers every element even though it is backed by a single bit.
        assert_eq!(validity.len(), 1_000);
        assert_eq!(validity.bitmap().len(), 1);
        assert!(validity.is_scalar());
        assert_eq!(validity.scalar_value(), Some(false));
        assert!(!validity.get(999));
        assert_eq!(validity.unset_bits(), 1_000);
        assert_eq!(validity.set_bits(), 0);

        // Materializing it yields exactly the mask a flat array would carry.
        assert_eq!(
            validity.to_flat(),
            scalar.to_flat().validity().unwrap().to_flat()
        );

        let flat: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect();
        let validity = flat.validity().unwrap();

        assert_eq!(validity.len(), 3);
        assert!(validity.is_flat());
        assert_eq!(validity.scalar_value(), None);
        assert!(!validity.get(1));
        assert_eq!(validity.unset_bits(), 1);
        assert_eq!(validity.to_flat(), *validity.bitmap());
    }

    #[test]
    fn validity_of_a_fully_valid_array() {
        assert!(
            PlPrimitiveArray::from_vec(vec![1i32, 2])
                .validity()
                .is_none()
        );
        assert!(
            PlPrimitiveArray::new_scalar(7i32, 1_000)
                .validity()
                .is_none()
        );
    }

    #[test]
    fn from_iter_with_nulls() {
        let arr: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect();

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
        assert!(arr.is_valid(0));
        assert!(arr.is_null(1));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(1), None, Some(3)]);
    }

    #[test]
    fn slicing_a_scalar_is_free() {
        let arr = PlPrimitiveArray::new_scalar(7i32, 1_000).sliced(500, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.values().len(), 1);
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(7), Some(7)]);
    }

    #[test]
    fn slicing_a_flat_array_slices_its_buffers() {
        let arr: PlPrimitiveArray<i32> = [Some(1), None, Some(3), Some(4)].into_iter().collect();
        let arr = arr.sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.values().len(), 2);
        assert_eq!(arr.validity().unwrap().len(), 2);
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, Some(3)]);
    }

    #[test]
    fn slicing_keeps_scalar_validity() {
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3])
            .with_validity(Some(Bitmap::new_zeroed(1)))
            .sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.values().len(), 2);
        assert_eq!(arr.validity().unwrap().len(), 2);
        assert_eq!(arr.validity().unwrap().bitmap().len(), 1);
        assert!(arr.validity().unwrap().is_scalar());
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, None]);
    }

    #[test]
    fn to_flat_materializes_scalars() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, 3);
        let flat = scalar.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.values().as_slice(), [7, 7, 7]);
        assert_eq!(flat, scalar);

        let null_scalar = PlPrimitiveArray::<i32>::new_full_null(3);
        let flat = null_scalar.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.validity().unwrap().bitmap().len(), 3);
        assert!(flat.validity().unwrap().is_flat());
        assert_eq!(flat.null_count(), 3);
        assert_eq!(flat, null_scalar);
    }

    #[test]
    fn to_flat_of_empty_scalar() {
        let flat = PlPrimitiveArray::new_scalar(7i32, 0).to_flat();

        assert!(flat.is_flat());
        assert!(flat.is_empty());
        assert_eq!(flat.values().len(), 0);
    }

    #[test]
    fn equality_ignores_representation() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, 3);
        let flat = PlPrimitiveArray::from_vec(vec![7i32, 7, 7]);

        assert_eq!(scalar, flat);
        assert_ne!(scalar, PlPrimitiveArray::new_scalar(7i32, 4));
        assert_ne!(scalar, PlPrimitiveArray::from_vec(vec![7i32, 7, 8]));
        assert_ne!(scalar, PlPrimitiveArray::<i32>::new_full_null(3));
    }

    #[test]
    fn equality_of_scalars_does_not_walk_elements() {
        // Element-by-element comparison of a billion elements would not finish; the fast path must
        // hit.
        let arr = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);

        assert_eq!(arr, arr.clone());
        assert_ne!(arr, PlPrimitiveArray::new_scalar(8i32, 1_000_000_000));
        assert_ne!(arr, PlPrimitiveArray::<i32>::new_full_null(1_000_000_000));
        assert_eq!(
            PlPrimitiveArray::<i32>::new_full_null(1_000_000_000),
            PlPrimitiveArray::<i32>::new_full_null(1_000_000_000),
        );
    }

    #[test]
    fn empty() {
        let arr = PlPrimitiveArray::<i32>::new_empty();

        assert!(arr.is_empty());
        assert!(arr.is_flat());
        assert_eq!(arr.null_count(), 0);
        assert_eq!(arr.scalar_value(), None);
        assert_eq!(arr.iter().next(), None);
    }

    #[test]
    fn try_new_rejects_mismatched_buffers() {
        assert!(PlPrimitiveArray::try_new(Buffer::from(vec![1i32, 2]), 3, None).is_err());
        assert!(
            PlPrimitiveArray::try_new(Buffer::from(vec![1i32]), 3, Some(Bitmap::new_zeroed(2)))
                .is_err()
        );
        assert!(
            PlPrimitiveArray::try_new(Buffer::from(vec![1i32]), 3, Some(Bitmap::new_zeroed(3)))
                .is_ok()
        );
    }

    #[test]
    fn iterators_are_exact_sized() {
        let arr = PlPrimitiveArray::new_scalar(7i32, 5);

        assert_eq!(arr.iter().len(), 5);
        assert_eq!(arr.values_iter().len(), 5);
        assert_eq!(arr.iter().size_hint(), (5, Some(5)));
    }

    #[test]
    fn debug_does_not_materialize_scalars() {
        let arr = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);
        assert_eq!(format!("{arr:?}"), "PlPrimitiveArray[7; 1000000000]");

        let arr: PlPrimitiveArray<i32> = [Some(1), None].into_iter().collect();
        assert_eq!(format!("{arr:?}"), "PlPrimitiveArray[1, null]");
    }
}
