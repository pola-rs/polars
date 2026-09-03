//! The wrapper that marks an array as being in the flat representation.

use std::ops::Deref;

use arrow::bitmap::Bitmap;

use crate::array::PlArray;

/// An array whose backing buffers all hold one slot per element.
#[repr(transparent)]
pub struct Flat<T>(pub(crate) T);

impl<T: PlArray> Flat<T> {
    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Panics
    /// Panics if `offset + length > self.len()`.
    pub fn slice(&mut self, offset: usize, length: usize) {
        self.0.slice(offset, length);
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        unsafe { self.0.slice_unchecked(offset, length) };
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

    /// Replaces the validity mask with a flat one.
    ///
    /// # Panics
    /// Panics if `validity` does not have exactly [`len`](PlArray::len) bits.
    pub fn set_validity(&mut self, validity: Option<Bitmap>) {
        // The flat requirement is the one `PlArray::set_validity` itself imposes.
        self.0.set_validity(validity);
    }

    /// Returns this array with its validity mask replaced by a flat one.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::set_validity`] panics.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Returns this array with its validity mask dropped, making every element valid.
    #[must_use]
    pub fn without_validity(mut self) -> Self {
        self.0.set_validity(None);
        self
    }
}

impl<T> Flat<T> {
    /// Wraps `array` as a flat one.
    ///
    /// # Safety
    /// Every backing buffer of `array` must hold one slot per element.
    #[inline(always)]
    pub const unsafe fn new(array: T) -> Self {
        Self(array)
    }

    /// Borrows `array` as a flat one.
    ///
    /// # Safety
    /// As [`Flat::new`].
    #[inline(always)]
    pub const unsafe fn new_ref(array: &T) -> &Self {
        // SAFETY: `Flat` is `repr(transparent)` over the array it wraps, which the caller
        // guarantees is flat.
        unsafe { &*(std::ptr::from_ref(array).cast::<Self>()) }
    }

    /// The array itself, which is in the flat representation.
    #[inline(always)]
    pub const fn as_array(&self) -> &T {
        &self.0
    }

    /// Borrows `array` as a flat one, mutably.
    ///
    /// # Safety
    /// As [`Flat::new`], and the array must still be flat when the borrow ends.
    #[inline(always)]
    pub const unsafe fn new_mut(array: &mut T) -> &mut Self {
        // SAFETY: `Flat` is `repr(transparent)` over the array it wraps, which the caller
        // guarantees is flat.
        unsafe { &mut *(std::ptr::from_mut(array).cast::<Self>()) }
    }

    /// The array itself, mutably.
    ///
    /// # Safety
    /// The array must still be flat when the borrow ends: every backing buffer must hold one slot
    /// per element.
    #[inline(always)]
    pub const unsafe fn as_array_mut(&mut self) -> &mut T {
        &mut self.0
    }

    /// Unwraps the array, giving up the guarantee that it is flat.
    #[inline(always)]
    pub fn into_array(self) -> T {
        self.0
    }
}

impl<T> Deref for Flat<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> AsRef<T> for Flat<T> {
    #[inline(always)]
    fn as_ref(&self) -> &T {
        &self.0
    }
}

impl<T: Clone> Clone for Flat<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Default> Default for Flat<T> {
    /// An empty array is flat: it has no element for a buffer to be scalar over.
    #[inline]
    fn default() -> Self {
        Self(T::default())
    }
}

/// Compares two arrays element-wise, exactly like comparing the arrays themselves.
impl<T: PartialEq> PartialEq for Flat<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Eq> Eq for Flat<T> {}

/// Compares this array against one of unknown representation; being flat is not part of a value.
impl<T: PartialEq> PartialEq<T> for Flat<T> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.0 == *other
    }
}

/// Formats the array itself: the wrapper is a representation guarantee, not part of the value.
impl<T: std::fmt::Debug> std::fmt::Debug for Flat<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PlBooleanArray, PlPrimitiveArray};

    #[test]
    fn slicing_stays_flat() {
        let flat = PlPrimitiveArray::new_scalar(7i32, 5)
            .to_flat()
            .into_owned()
            .sliced(1, 2);

        assert!(flat.is_flat());
        assert_eq!(flat.values().as_slice(), [7, 7]);

        // Slicing away every element leaves an empty flat array.
        let flat = PlBooleanArray::new_full_null(5)
            .to_flat()
            .into_owned()
            .sliced(2, 0);

        assert!(flat.is_flat());
        assert!(flat.is_empty());
    }

    #[test]
    fn validity_must_stay_flat() {
        let flat = PlPrimitiveArray::from_vec(vec![1i32, 2, 3])
            .to_flat()
            .into_owned();

        let with_nulls = flat.with_validity(Some(Bitmap::from_iter([true, false, true])));
        assert!(with_nulls.is_flat());
        assert_eq!(with_nulls.null_count(), 1);
        assert!(with_nulls.without_validity().validity().is_none());
    }

    #[test]
    fn a_flat_array_is_the_array_it_wraps() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, 3);
        let flat = scalar.to_flat();

        // Equality and formatting see through the wrapper.
        assert_eq!(*flat, scalar);
        assert_eq!(flat, flat.clone());
        assert_eq!(format!("{flat:?}"), format!("{:?}", flat.as_array()));

        // The deref reaches everything that is not specialized.
        assert_eq!(flat.len(), 3);
        assert!(!flat.has_nulls());
        assert_eq!(flat.into_owned().into_array(), scalar);
    }
}
