use crate::array::Array;

/// Utility trait to slice concrete arrow arrays whilst keeping their
/// concrete type. E.g. don't return `Box<dyn Array>`.
pub trait SlicedArray {
    /// Slices this [`Array`].
    /// # Implementation
    /// This operation is `O(1)` over `len`.
    /// # Panic
    /// This function panics iff `offset + length > self.len()`.
    fn slice_typed(&self, offset: usize, length: usize) -> Self
    where
        Self: Sized;

    /// Slices the [`Array`].
    /// # Implementation
    /// This operation is `O(1)`.
    ///
    /// # Safety
    /// The caller must ensure that `offset + length <= self.len()`
    unsafe fn slice_typed_unchecked(&self, offset: usize, length: usize) -> Self
    where
        Self: Sized;
}

impl<T: Array + Clone> SlicedArray for T {
    fn slice_typed(&self, offset: usize, length: usize) -> Self {
        let mut arr = self.clone();
        arr.slice(offset, length);
        arr
    }

    unsafe fn slice_typed_unchecked(&self, offset: usize, length: usize) -> Self {
        let mut arr = self.clone();
        arr.slice_unchecked(offset, length);
        arr
    }
}
