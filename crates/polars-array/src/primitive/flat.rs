//! What a [`PlPrimitiveArray`] gains from being known to be [`Flat`].

use arrow::bitmap::Bitmap;
use arrow::bitmap::utils::{BitmapIter, ZipValidity};
use arrow::types::NativeType;
use polars_buffer::Buffer;

use super::PlPrimitiveArray;
use crate::flat::Flat;

/// The methods a [`PlPrimitiveArray`] gains from having one slot per element in every backing
/// buffer.
///
/// These are the counterparts of the methods on
/// [`PrimitiveArray`](arrow::array::PrimitiveArray), whose values buffer *is* its elements: they
/// hand out the backing buffers as they are and read them without a
/// [`broadcast_index`](crate::broadcast::broadcast_index). Each shadows the broadcast-aware method
/// of the same name on [`PlPrimitiveArray`], which remains reachable through the deref.
impl<T: NativeType> Flat<PlPrimitiveArray<T>> {
    /// The backing values buffer, holding exactly [`len`](PlPrimitiveArray::len) slots.
    ///
    /// Unlike [`PlPrimitiveArray::values`], this is guaranteed to hold one slot per element: slot
    /// `i` is element `i`. The values of null elements are undetermined (they can be anything).
    #[inline(always)]
    pub const fn values(&self) -> &Buffer<T> {
        &self.0.values
    }

    /// The values as a slice of exactly [`len`](PlPrimitiveArray::len) elements.
    ///
    /// The values of null elements are undetermined (they can be anything).
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.0.values.as_slice()
    }

    /// The validity mask, if any element may be null, as an ordinary [`Bitmap`] of exactly
    /// [`len`](PlPrimitiveArray::len) bits.
    ///
    /// Unlike [`PlPrimitiveArray::validity`], this needs no [`PlBitmapRef`](crate::PlBitmapRef) to
    /// hide a scalar bit: bit `i` is element `i`.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.0.validity.as_ref()
    }

    /// Returns the value at `i`.
    ///
    /// The value of a null element is undetermined (it can be anything).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value(&self, i: usize) -> T {
        assert!(i < self.0.length, "index out of bounds");
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
        debug_assert!(i < self.0.length);
        unsafe { *self.0.values.get_unchecked(i) }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        assert!(i < self.0.length, "index out of bounds");
        unsafe { self.is_valid_unchecked(i) }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        debug_assert!(i < self.0.length);
        // SAFETY: the mask has one bit per element, so `i` is in bounds of it too.
        self.validity()
            .is_none_or(|validity| unsafe { validity.get_bit_unchecked(i) })
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
        assert!(i < self.0.length, "index out of bounds");
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

    /// Returns an iterator over the values, ignoring validity.
    ///
    /// This walks the values buffer directly, so — unlike
    /// [`PlPrimitiveArray::values_iter`] — it is an ordinary [`slice::Iter`](std::slice::Iter) and
    /// yields references. The values of null elements are undetermined (they can be anything).
    #[inline]
    pub fn values_iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns an iterator over the optional elements.
    ///
    /// This zips the two backing buffers directly, so — unlike [`PlPrimitiveArray::iter`], which
    /// yields `Option<T>` — it mirrors [`PrimitiveArray::iter`](arrow::array::PrimitiveArray::iter)
    /// and yields `Option<&T>`.
    #[inline]
    pub fn iter(&self) -> ZipValidity<&T, std::slice::Iter<'_, T>, BitmapIter<'_>> {
        ZipValidity::new_with_validity(self.values_iter(), self.validity())
    }

    /// Consumes this array into its backing buffers, which both hold one slot per element.
    ///
    /// The length is not part of the result: it is the length of the values buffer.
    #[inline]
    pub fn into_inner(self) -> (Buffer<T>, Option<Bitmap>) {
        let PlPrimitiveArray {
            values,
            length: _,
            validity,
        } = self.0;

        (values, validity)
    }
}

impl<'a, T: NativeType> IntoIterator for &'a Flat<PlPrimitiveArray<T>> {
    type Item = Option<&'a T>;
    type IntoIter = ZipValidity<&'a T, std::slice::Iter<'a, T>, BitmapIter<'a>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares an array of unknown representation against a flat one; see
/// [`PartialEq<PlPrimitiveArray<T>> for Flat<PlPrimitiveArray<T>>`](Flat).
impl<T: NativeType> PartialEq<Flat<PlPrimitiveArray<T>>> for PlPrimitiveArray<T> {
    #[inline]
    fn eq(&self, other: &Flat<PlPrimitiveArray<T>>) -> bool {
        *self == other.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(flat.validity().unwrap().len(), 3);
        assert_eq!(flat.null_count(), 3);
        assert_eq!(flat, null_scalar);
    }

    #[test]
    fn to_flat_of_a_flat_array_only_clones() {
        let arr: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect();
        let flat = arr.to_flat();

        assert_eq!(flat, arr);
        assert!(
            flat.values().is_same_buffer(arr.values()),
            "the values buffer must be shared, not materialized again",
        );
    }

    #[test]
    fn to_flat_of_empty_scalar() {
        let flat = PlPrimitiveArray::new_scalar(7i32, 0).to_flat();

        assert!(flat.is_flat());
        assert!(flat.is_empty());
        assert_eq!(flat.values().len(), 0);
    }

    #[test]
    fn buffers_are_handed_out_as_they_are() {
        let flat = PlPrimitiveArray::<i32>::new_full_null(3).to_flat();

        assert_eq!(flat.values().len(), 3);
        assert_eq!(flat.as_slice().len(), 3);
        assert_eq!(flat.validity().unwrap().len(), 3);
        assert_eq!(flat.validity().unwrap().unset_bits(), 3);
    }

    #[test]
    fn elements_are_read_without_a_broadcast() {
        let flat: Flat<PlPrimitiveArray<i32>> = [Some(1), None, Some(3)]
            .into_iter()
            .collect::<PlPrimitiveArray<i32>>()
            .to_flat();

        assert_eq!(flat.value(0), 1);
        assert_eq!(flat.get(0), Some(1));
        assert!(flat.is_valid(0));
        assert!(flat.is_null(1));
        assert_eq!(flat.get(1), None);
        assert_eq!(flat.get(2), Some(3));

        assert_eq!(unsafe { flat.value_unchecked(2) }, 3);
        assert_eq!(unsafe { flat.get_unchecked(1) }, None);
        assert!(unsafe { flat.is_null_unchecked(1) });
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn value_panics_out_of_bounds() {
        let _ = PlPrimitiveArray::new_scalar(7i32, 3).to_flat().value(3);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn get_panics_out_of_bounds() {
        let _ = PlPrimitiveArray::new_scalar(7i32, 3).to_flat().get(3);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn is_valid_panics_out_of_bounds() {
        let _ = PlPrimitiveArray::new_scalar(7i32, 3).to_flat().is_valid(3);
    }

    #[test]
    fn iterators_walk_the_buffers() {
        let flat = PlPrimitiveArray::new_scalar(7i32, 3).to_flat();

        assert_eq!(flat.values_iter().copied().collect::<Vec<_>>(), [7, 7, 7]);
        assert_eq!(flat.iter().collect::<Vec<_>>(), [Some(&7); 3]);
        assert_eq!((&flat).into_iter().collect::<Vec<_>>(), [Some(&7); 3]);

        let flat: Flat<PlPrimitiveArray<i32>> = [Some(1), None, Some(3)]
            .into_iter()
            .collect::<PlPrimitiveArray<i32>>()
            .to_flat();

        assert_eq!(flat.values_iter().len(), 3);
        assert_eq!(flat.iter().collect::<Vec<_>>(), [Some(&1), None, Some(&3)]);
    }

    #[test]
    fn into_inner_gives_up_the_length() {
        let (values, validity) = PlPrimitiveArray::<i32>::new_full_null(3)
            .to_flat()
            .into_inner();

        assert_eq!(values.len(), 3);
        assert_eq!(validity.unwrap().len(), 3);

        let (values, validity) = PlPrimitiveArray::from_vec(vec![1i32, 2])
            .to_flat()
            .into_inner();

        assert_eq!(values.as_slice(), [1, 2]);
        assert!(validity.is_none());
    }

    #[test]
    fn equality_ignores_representation() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, 3);
        let flat = scalar.to_flat();

        assert_eq!(flat, scalar);
        assert_eq!(scalar, flat);
        assert_eq!(flat, PlPrimitiveArray::from_vec(vec![7i32, 7, 7]).to_flat());
        assert_ne!(flat, PlPrimitiveArray::new_scalar(7i32, 4));
        assert_ne!(PlPrimitiveArray::<i32>::new_full_null(3), flat);
    }
}
