use std::marker::PhantomData;

use arrow::array::*;

use crate::prelude::*;
use crate::utils::index_to_chunked_index;

pub struct Chunks<'a, T> {
    chunks: &'a [ArrayRef],
    phantom: PhantomData<T>,
}

impl<'a, T> Chunks<'a, T> {
    #[inline]
    fn new(chunks: &'a [ArrayRef]) -> Self {
        Chunks {
            chunks,
            phantom: PhantomData,
        }
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&'a T> {
        self.chunks.get(index).map(|arr| {
            let arr = &**arr;
            unsafe { &*(arr as *const dyn Array as *const T) }
        })
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &'a T {
        let arr = self.chunks.get_unchecked(index);
        let arr = &**arr;
        &*(arr as *const dyn Array as *const T)
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }
}

impl<T: PolarsDowncastType> ChunkedArray<T> {
    #[inline]
    pub fn downcast_iter(&self) -> impl Iterator<Item = &T::Array> + DoubleEndedIterator {
        // Safety:
        // Array type should be correct as specified in PolarsDowncastType
        self.chunks.iter().map(|arr| {
            let arr = &**arr;
            unsafe { &*(arr as *const dyn Array as *const T::Array) }
        })
    }

    #[inline]
    pub fn downcast_chunks(&self) -> Chunks<'_, T::Array> {
        Chunks::new(&self.chunks)
    }

    /// Get the index of the chunk and the index of the value in that chunk.
    #[inline]
    pub(crate) fn index_to_chunked_index(&self, index: usize) -> (usize, usize) {
        if self.chunks.len() == 1 {
            return (0, index);
        }
        index_to_chunked_index(self.downcast_iter().map(|arr| arr.len()), index)
    }
}

#[doc(hidden)]
impl<T> ChunkedArray<T>
where
    T: PolarsDowncastType + PolarsNumericType,
{
    /// # Safety
    /// The caller must ensure:
    ///     * the length remains correct.
    ///     * the flags (sorted, etc) remain correct.
    pub unsafe fn downcast_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut T::Array> + DoubleEndedIterator {
        // it's not used atm except for numeric types, hence the bound
        self.chunks.iter_mut().map(|arr| {
            // Safety:
            // This should be the array type in PolarsNumericType
            let arr = &mut **arr;
            &mut *(arr as *mut dyn Array as *mut T::Array)
        })
    }
}
