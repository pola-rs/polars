use std::marker::PhantomData;

use arrow::array::*;

use crate::prelude::*;
use crate::utils::index_to_chunked_index;

pub struct Chunks<'a, T> {
    chunks: &'a [ArrayRef],
    phantom: PhantomData<T>,
}

impl<'a, T> Chunks<'a, T> {
    fn new(chunks: &'a [ArrayRef]) -> Self {
        Chunks {
            chunks,
            phantom: PhantomData,
        }
    }

    pub fn get(&self, index: usize) -> Option<&'a T> {
        self.chunks.get(index).map(|arr| {
            let arr = &**arr;
            unsafe { &*(arr as *const dyn Array as *const T) }
        })
    }

    pub unsafe fn get_unchecked(&self, index: usize) -> &'a T {
        let arr = self.chunks.get_unchecked(index);
        let arr = &**arr;
        &*(arr as *const dyn Array as *const T)
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn last(&self) -> Option<&'a T> {
        self.chunks.last().map(|arr| {
            let arr = &**arr;
            unsafe { &*(arr as *const dyn Array as *const T) }
        })
    }
}

#[doc(hidden)]
impl<T: PolarsDataType> ChunkedArray<T>
where
    Self: HasUnderlyingArray,
{
    pub fn downcast_iter(
        &self,
    ) -> impl Iterator<Item = &<Self as HasUnderlyingArray>::ArrayT> + DoubleEndedIterator {
        self.chunks.iter().map(|arr| {
            // SAFETY: HasUnderlyingArray guarantees this is correct.
            let arr = &**arr;
            unsafe { &*(arr as *const dyn Array as *const <Self as HasUnderlyingArray>::ArrayT) }
        })
    }

    /// # Safety
    /// The caller must ensure:
    ///     * the length remains correct.
    ///     * the flags (sorted, etc) remain correct.
    pub unsafe fn downcast_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut <Self as HasUnderlyingArray>::ArrayT> + DoubleEndedIterator {
        self.chunks.iter_mut().map(|arr| {
            // SAFETY: HasUnderlyingArray guarantees this is correct.
            let arr = &mut **arr;
            &mut *(arr as *mut dyn Array as *mut <Self as HasUnderlyingArray>::ArrayT)
        })
    }

    pub fn downcast_chunks(&self) -> Chunks<'_, <Self as HasUnderlyingArray>::ArrayT> {
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
