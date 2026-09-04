//! What a [`PlPrimitiveArray`] gains from being known to be [`Flat`].

use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_buffer::Buffer;

use super::{PlPrimitiveArray, PlPrimitiveIter};
use crate::flat::Flat;

/// The methods a [`PlPrimitiveArray`] gains from having one slot per element in every backing
/// buffer.
impl<T: NativeType> Flat<PlPrimitiveArray<T>> {
    /// The backing values buffer, holding exactly [`len`](PlPrimitiveArray::len) slots.
    #[inline(always)]
    pub const fn values(&self) -> &Buffer<T> {
        &self.0.values
    }

    /// The values as a slice of exactly [`len`](PlPrimitiveArray::len) elements.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.0.values.as_slice()
    }

    /// The validity mask, if any element may be null, as an ordinary [`Bitmap`] of exactly
    /// [`len`](PlPrimitiveArray::len) bits.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.0.validity.as_ref()
    }

    /// Returns the value at `i`.
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
    #[inline]
    pub fn values_iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns an iterator over the optional elements.
    ///
    /// Knowing the array is flat buys nothing here, so this is the array's own iterator. Arrow's
    /// `ZipValidity`, which this used to return, resolves its representation once per step rather
    /// than once per walk and leaves [`Iterator::fold`] to the default; either of those stops the
    /// loop from vectorizing.
    #[inline]
    pub fn iter(&self) -> PlPrimitiveIter<'_, T> {
        self.0.iter()
    }

    /// The backing values buffer as a mutable slice, if no other array shares it.
    #[inline]
    pub fn values_mut(&mut self) -> Option<&mut [T]> {
        self.0.values.get_mut_slice()
    }

    /// Takes the validity mask out, leaving every element valid.
    #[inline]
    pub fn take_validity(&mut self) -> Option<Bitmap> {
        self.0.validity.take()
    }

    /// Reinterprets the values buffer as one of `U`, keeping the validity mask.
    ///
    /// # Panics
    /// Panics unless `T` and `U` have the same size and alignment.
    pub fn transmute<U: NativeType>(self) -> Flat<PlPrimitiveArray<U>> {
        let (values, validity) = self.into_inner();
        let values = values
            .try_transmute::<U>()
            .expect("values buffer cannot be reinterpreted");
        let length = values.len();

        // SAFETY: the buffers held one slot per element, and reinterpreting the values buffer as
        // a type of the same size leaves it one slot per element too.
        unsafe { Flat::new(PlPrimitiveArray::new_unchecked(values, length, validity)) }
    }

    /// Fills every element with `value`, leaving the validity mask as it is.
    pub fn fill_with(self, value: T) -> PlPrimitiveArray<T> {
        let length = self.len();
        let (_, validity) = self.into_inner();

        PlPrimitiveArray::new_scalar(value, length).with_validity(validity)
    }

    /// Consumes this array into its backing buffers, which both hold one slot per element.
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
    type Item = Option<T>;
    type IntoIter = PlPrimitiveIter<'a, T>;

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
    fn values_are_written_over_when_the_buffer_is_not_shared() {
        let mut flat = PlPrimitiveArray::from_vec(vec![1i32, 2, 3])
            .to_flat()
            .into_owned();
        flat.values_mut().expect("the buffer is not shared")[0] = 7;
        assert_eq!(flat.as_slice(), [7, 2, 3]);

        // A second reference to the buffer leaves the caller to allocate one of its own.
        let shared = flat.clone();
        assert!(flat.values_mut().is_none());
        drop(shared);
    }

    #[test]
    fn values_are_reinterpreted_in_place() {
        let flat = PlPrimitiveArray::from_vec(vec![1u32, 2, 3])
            .to_flat()
            .into_owned();
        let values = flat.values().as_ptr();

        let transmuted = flat.transmute::<i32>();

        assert_eq!(transmuted.as_slice(), [1i32, 2, 3]);
        assert_eq!(
            transmuted.values().as_ptr().cast::<u32>(),
            values,
            "the values buffer must be reinterpreted, not copied",
        );
    }

    #[test]
    fn elements_are_the_ones_the_array_itself_yields() {
        let arr: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect();
        let flat = arr.as_flat().expect("the array is flat");

        let expected = [Some(1), None, Some(3)];
        assert_eq!(flat.iter().collect::<Vec<_>>(), expected);
        assert_eq!(flat.into_iter().collect::<Vec<_>>(), expected);
        assert_eq!(flat.values_iter().copied().collect::<Vec<_>>(), [1, 0, 3]);
    }

    #[test]
    fn as_flat_borrows_an_already_flat_array() {
        let arr: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect();
        let flat = arr.as_flat().expect("the array is flat");

        assert_eq!(flat.as_slice(), [1, 0, 3]);
        assert_eq!(*flat, arr);
        assert!(
            flat.values().is_same_buffer(arr.flat_values().unwrap()),
            "the values buffer must be borrowed, not materialized again",
        );

        // Neither a scalar buffer nor a scalar validity mask can be borrowed as flat.
        assert!(PlPrimitiveArray::new_scalar(7i32, 3).as_flat().is_none());
        assert!(
            PlPrimitiveArray::from_vec(vec![1i32, 2, 3])
                .with_validity_broadcast(Some(Bitmap::new_zeroed(1)))
                .as_flat()
                .is_none()
        );

        // A scalar array of unbounded length is still `O(1)` to reject.
        assert!(
            PlPrimitiveArray::<i32>::new_full_null(1_000_000_000)
                .as_flat()
                .is_none()
        );
    }
}
