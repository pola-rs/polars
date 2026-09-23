//! The wrapper that marks an array as being in the flat representation.

use std::ops::Deref;

use crate::array::PlArray;
use crate::bitmap::PlBitmap;
use crate::no_nulls::NoNulls;

/// An array whose backing buffers all hold one slot per element.
#[repr(transparent)]
pub struct Flat<T>(T);

impl<T: PlArray> Flat<T> {
    /// Slices this array in place to `length` elements starting at `offset`.
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
    #[must_use]
    pub fn sliced(&self, offset: usize, length: usize) -> Self
    where
        T: Clone,
    {
        let mut sliced = self.clone();
        sliced.slice(offset, length);
        sliced
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    #[must_use]
    pub unsafe fn sliced_unchecked(&self, offset: usize, length: usize) -> Self
    where
        T: Clone,
    {
        let mut sliced = self.clone();
        unsafe { sliced.slice_unchecked(offset, length) };
        sliced
    }

    /// Replaces the validity mask with a flat one.
    pub fn set_validity(&mut self, validity: Option<PlBitmap>) {
        assert!(
            validity.as_ref().is_none_or(PlBitmap::is_flat),
            "a flat array takes a flat validity mask, not one that repeats a single bit",
        );
        self.0.set_validity(validity);
    }

    /// Returns this array with its validity mask replaced by a flat one.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<PlBitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Returns this array with its validity mask dropped, making every element valid.
    #[must_use]
    pub fn without_validity(mut self) -> Self {
        self.0.set_validity(None);
        self
    }

    /// Borrows this array as one with no null elements, or `None` if any element is null.
    #[inline]
    pub fn as_no_nulls(&self) -> Option<&NoNulls<Self>> {
        // SAFETY: no element is null, as just counted.
        (self.0.null_count() == 0).then(|| unsafe { NoNulls::new_ref(self) })
    }

    /// Wraps this array as one with no null elements, or hands it back if any element is null.
    pub fn try_into_no_nulls(self) -> Result<NoNulls<Self>, Self> {
        match self.0.null_count() {
            // SAFETY: no element is null, as just counted.
            0 => Ok(unsafe { NoNulls::new(self) }),
            _ => Err(self),
        }
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
    /// The array must still be flat when the borrow ends: one slot per element everywhere.
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
