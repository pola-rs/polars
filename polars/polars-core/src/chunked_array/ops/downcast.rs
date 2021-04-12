#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectArray;
use crate::prelude::*;
use arrow::array::{
    Array, ArrayRef, BooleanArray, LargeListArray, LargeStringArray, PrimitiveArray,
};
use std::marker::PhantomData;

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
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    pub fn downcast_iter(&self) -> impl Iterator<Item = &PrimitiveArray<T>> + DoubleEndedIterator {
        self.chunks.iter().map(|arr| {
            let arr = &**arr;
            unsafe { &*(arr as *const dyn Array as *const PrimitiveArray<T>) }
        })
    }
    pub fn downcast_chunks(&self) -> Chunks<'_, PrimitiveArray<T>> {
        Chunks::new(&self.chunks)
    }
}

impl BooleanChunked {
    pub fn downcast_iter(&self) -> impl Iterator<Item = &BooleanArray> + DoubleEndedIterator {
        self.chunks.iter().map(|arr| {
            let arr = &**arr;
            unsafe { &*(arr as *const dyn Array as *const BooleanArray) }
        })
    }
    pub fn downcast_chunks(&self) -> Chunks<'_, BooleanArray> {
        Chunks::new(&self.chunks)
    }
}

impl Utf8Chunked {
    pub fn downcast_iter(&self) -> impl Iterator<Item = &LargeStringArray> + DoubleEndedIterator {
        self.chunks.iter().map(|arr| {
            let arr = &**arr;
            unsafe { &*(arr as *const dyn Array as *const LargeStringArray) }
        })
    }
    pub fn downcast_chunks(&self) -> Chunks<'_, LargeStringArray> {
        Chunks::new(&self.chunks)
    }
}

impl ListChunked {
    pub fn downcast_iter(&self) -> impl Iterator<Item = &LargeListArray> + DoubleEndedIterator {
        self.chunks.iter().map(|arr| {
            let arr = &**arr;
            unsafe { &*(arr as *const dyn Array as *const LargeListArray) }
        })
    }
    pub fn downcast_chunks(&self) -> Chunks<'_, LargeListArray> {
        Chunks::new(&self.chunks)
    }
}

#[cfg(feature = "object")]
impl<T> ObjectChunked<T>
where
    T: 'static + std::fmt::Debug + Clone + Send + Sync + Default,
{
    pub fn downcast_iter(&self) -> impl Iterator<Item = &ObjectArray<T>> + DoubleEndedIterator {
        self.chunks.iter().map(|arr| {
            let arr = &**arr;
            unsafe { &*(arr as *const dyn Array as *const ObjectArray<T>) }
        })
    }
    pub fn downcast_chunks(&self) -> Chunks<'_, ObjectArray<T>> {
        Chunks::new(&self.chunks)
    }
}
