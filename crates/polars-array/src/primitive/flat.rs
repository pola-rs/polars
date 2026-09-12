//! What a [`PlPrimitiveArray`] gains from being known to be [`Flat`].

use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_buffer::Buffer;

use super::{PlPrimitiveArray, PlPrimitiveIter};
use crate::bitmap::PlBitmap;
use crate::flat::Flat;

/// The methods a [`PlPrimitiveArray`] gains from holding one slot per element everywhere.
impl<T: NativeType> Flat<PlPrimitiveArray<T>> {
    /// The backing values buffer, holding exactly [`len`](PlPrimitiveArray::len) slots.
    #[inline(always)]
    pub const fn values(&self) -> &Buffer<T> {
        &self.as_array().values
    }

    /// The values as a slice of exactly [`len`](PlPrimitiveArray::len) elements.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.as_array().values.as_slice()
    }

    /// Returns the value at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> T {
        debug_assert!(i < self.as_array().length);
        unsafe { *self.as_array().values.get_unchecked(i) }
    }

    /// Returns an iterator over the values, ignoring validity.
    #[inline]
    pub fn values_iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlPrimitiveIter<'_, T> {
        self.as_array().iter()
    }

    /// The backing values buffer as a mutable slice, if no other array shares it.
    #[inline]
    pub fn values_mut(&mut self) -> Option<&mut [T]> {
        // SAFETY: writing over the values leaves the buffer as many slots as it was, so the array
        // is still flat when the borrow ends.
        unsafe { self.as_array_mut() }.values.get_mut_slice()
    }

    /// Takes the validity mask out, leaving every element valid.
    #[inline]
    pub fn take_validity(&mut self) -> Option<Bitmap> {
        // SAFETY: dropping the mask leaves every element valid, which is as flat as a mask of one
        // bit per element.
        unsafe { self.as_array_mut() }.validity.take()
    }

    /// Reinterprets the values buffer as one of `U`, keeping the validity mask.
    pub fn transmute<U: NativeType>(self) -> Flat<PlPrimitiveArray<U>> {
        let (values, validity) = self.into_inner();
        let validity = validity.map(PlBitmap::from_bitmap);
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
        let validity = validity.map(PlBitmap::from_bitmap);

        PlPrimitiveArray::new_scalar(value, length).with_validity(validity)
    }

    /// Consumes this array into its backing buffers, which both hold one slot per element.
    #[inline]
    pub fn into_inner(self) -> (Buffer<T>, Option<Bitmap>) {
        let PlPrimitiveArray {
            values,
            length: _,
            validity,
        } = self.into_array();

        (values, validity)
    }
}

crate::impl_flat_methods!([T: NativeType] PlPrimitiveArray<T>, T);

crate::impl_into_iterator!([T: NativeType] Flat<PlPrimitiveArray<T>>, PlPrimitiveIter<'a, T>);

/// Compares an array of unknown representation against a flat one.
impl<T: NativeType> PartialEq<Flat<PlPrimitiveArray<T>>> for PlPrimitiveArray<T> {
    #[inline]
    fn eq(&self, other: &Flat<PlPrimitiveArray<T>>) -> bool {
        *self == *other.as_array()
    }
}
