//! The wrapper that marks an array as having no null elements.

use std::ops::Deref;

use crate::array::PlArray;
use crate::flat::Flat;

/// An array none of whose elements is null.
///
/// This is the witness a kernel takes when its answer is only defined for valid elements — the
/// counterpart of [`Flat`] on the validity axis. It is deliberately *not* a representation: an
/// array with no nulls may carry a validity mask that happens to be all-set, in either
/// representation, and wrapping it here says nothing about that mask beyond it having no zero bit.
///
/// Like [`Flat`], it derefs to the array it wraps and has no `DerefMut`: a mutation could set a
/// bit and leave the witness lying. `NoNulls<Flat<A>>` composes the two, and reaches the inherent
/// methods of both through the deref chain.
#[repr(transparent)]
pub struct NoNulls<T>(T);

impl<T> NoNulls<T> {
    /// Wraps `array` as one with no null elements.
    ///
    /// # Safety
    /// No element of `array` may be null.
    #[inline(always)]
    pub const unsafe fn new(array: T) -> Self {
        Self(array)
    }

    /// Borrows `array` as one with no null elements.
    ///
    /// # Safety
    /// As [`NoNulls::new`].
    #[inline(always)]
    pub const unsafe fn new_ref(array: &T) -> &Self {
        // SAFETY: `NoNulls` is `repr(transparent)` over the array it wraps, which the caller
        // guarantees has no null element.
        unsafe { &*(std::ptr::from_ref(array).cast::<Self>()) }
    }

    /// The array itself, which has no null element.
    #[inline(always)]
    pub const fn as_array(&self) -> &T {
        &self.0
    }

    /// Unwraps the array, giving up the guarantee that it has no null element.
    #[inline(always)]
    pub fn into_array(self) -> T {
        self.0
    }
}

impl<T: PlArray> NoNulls<T> {
    /// Wraps `array` as one with no null elements, or returns it back if any element is null.
    pub fn try_new(array: T) -> Result<Self, T> {
        match array.null_count() {
            // SAFETY: no element is null, as just counted.
            0 => Ok(unsafe { Self::new(array) }),
            _ => Err(array),
        }
    }

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
}

impl<T> NoNulls<Flat<T>> {
    /// The array itself, in the flat representation and with no null element.
    #[inline(always)]
    pub const fn as_flat(&self) -> &Flat<T> {
        &self.0
    }
}

impl<T> Deref for NoNulls<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> AsRef<T> for NoNulls<T> {
    #[inline(always)]
    fn as_ref(&self) -> &T {
        &self.0
    }
}

impl<T: Clone> Clone for NoNulls<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Default> Default for NoNulls<T> {
    /// An empty array has no null element: it has no element at all.
    #[inline]
    fn default() -> Self {
        Self(T::default())
    }
}

/// Compares two arrays element-wise, exactly like comparing the arrays themselves.
impl<T: PartialEq> PartialEq for NoNulls<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Eq> Eq for NoNulls<T> {}

/// Compares this array against one that may hold nulls; the witness is not part of a value.
impl<T: PartialEq> PartialEq<T> for NoNulls<T> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.0 == *other
    }
}

/// Formats the array itself: the wrapper is a guarantee about it, not part of the value.
impl<T: std::fmt::Debug> std::fmt::Debug for NoNulls<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use crate::bitmap::PlBitmap;
    use crate::static_array::StaticArray;
    use crate::PlPrimitiveArray;

    #[test]
    fn an_array_without_a_mask_has_no_nulls() {
        let array = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);
        assert!(array.as_no_nulls().is_some());
    }

    #[test]
    fn an_all_set_mask_still_has_no_nulls() {
        let array = PlPrimitiveArray::from_vec(vec![1i32, 2, 3])
            .with_validity(Some(PlBitmap::from_bitmap(Bitmap::new_with_value(
                true, 3,
            ))));
        assert_eq!(array.null_count(), 0);
        assert!(array.as_no_nulls().is_some());
    }

    #[test]
    fn a_single_null_denies_the_witness() {
        let array = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]).with_validity(Some(
            PlBitmap::from_bitmap(Bitmap::from_iter([true, false, true])),
        ));
        assert!(array.as_no_nulls().is_none());
    }

    #[test]
    fn a_scalar_array_can_be_witnessed_without_being_flattened() {
        let array = PlPrimitiveArray::new_scalar(7i64, 1024);
        let no_nulls = array.as_no_nulls().unwrap();
        assert!(!no_nulls.is_flat());
        assert_eq!(no_nulls.len(), 1024);
    }
}
