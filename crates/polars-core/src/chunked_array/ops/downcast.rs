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

    #[inline]
    pub fn last(&self) -> Option<&'a T> {
        self.chunks.last().map(|arr| {
            let arr = &**arr;
            unsafe { &*(arr as *const dyn Array as *const T) }
        })
    }
}

#[doc(hidden)]
impl<T: PolarsDataType> ChunkedArray<T> {
    #[inline]
    pub fn downcast_into_iter(mut self) -> impl DoubleEndedIterator<Item = T::Array> {
        let chunks = std::mem::take(&mut self.chunks);
        chunks.into_iter().map(|arr| {
            // SAFETY: T::Array guarantees this is correct.
            let ptr = Box::into_raw(arr).cast::<T::Array>();
            unsafe { *Box::from_raw(ptr) }
        })
    }

    #[inline]
    pub fn downcast_iter(&self) -> impl DoubleEndedIterator<Item = &T::Array> {
        self.chunks.iter().map(|arr| {
            // SAFETY: T::Array guarantees this is correct.
            let arr = &**arr;
            unsafe { &*(arr as *const dyn Array as *const T::Array) }
        })
    }

    /// # Safety
    /// The caller must ensure:
    ///     * the length remains correct.
    ///     * the flags (sorted, etc) remain correct.
    #[inline]
    pub unsafe fn downcast_iter_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut T::Array> {
        self.chunks.iter_mut().map(|arr| {
            // SAFETY: T::Array guarantees this is correct.
            let arr = &mut **arr;
            &mut *(arr as *mut dyn Array as *mut T::Array)
        })
    }

    #[inline]
    pub fn downcast_chunks(&self) -> Chunks<'_, T::Array> {
        Chunks::new(&self.chunks)
    }

    #[inline]
    pub fn downcast_get(&self, idx: usize) -> Option<&T::Array> {
        let arr = self.chunks.get(idx)?;
        // SAFETY: T::Array guarantees this is correct.
        let arr = &**arr;
        unsafe { Some(&*(arr as *const dyn Array as *const T::Array)) }
    }

    #[inline]
    /// # Safety
    /// It is up to the caller to ensure the chunk idx is in-bounds
    pub unsafe fn downcast_get_unchecked(&self, idx: usize) -> &T::Array {
        let arr = self.chunks.get_unchecked(idx);
        // SAFETY: T::Array guarantees this is correct.
        let arr = &**arr;
        unsafe { &*(arr as *const dyn Array as *const T::Array) }
    }

    /// Get the index of the chunk and the index of the value in that chunk.
    #[inline]
    pub(crate) fn index_to_chunked_index(&self, index: usize) -> (usize, usize) {
        if self.chunks.len() == 1 {
            // SAFETY: chunks.len() == 1 guarantees this is correct.
            let len = unsafe { self.chunks.get_unchecked(0).len() };
            return if index < len {
                (0, index)
            } else {
                (1, index - len)
            };
        }
        index_to_chunked_index(self.downcast_iter().map(|arr| arr.len()), index)
    }
}
