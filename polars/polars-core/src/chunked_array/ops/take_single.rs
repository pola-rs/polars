#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectArray;
use crate::prelude::*;
use arrow::array::{
    Array, ArrayRef, BooleanArray, LargeListArray, LargeStringArray, PrimitiveArray,
};
use std::convert::TryFrom;
use std::ops::Deref;
use std::sync::Arc;
use unsafe_unwrap::UnsafeUnwrap;

macro_rules! impl_take_random_get {
    ($self:ident, $index:ident, $array_type:ty) => {{
        let (chunk_idx, idx) = $self.index_to_chunked_index($index);
        // Safety:
        // bounds are checked above
        let arr = $self.chunks.get_unchecked(chunk_idx);

        // Safety:
        // caller should give right array type
        let arr = &*(arr as *const ArrayRef as *const Arc<$array_type>);

        // Safety:
        // index should be in bounds
        if arr.is_valid(idx) {
            Some(arr.value_unchecked(idx))
        } else {
            None
        }
    }};
}

macro_rules! impl_take_random_get_unchecked {
    ($self:ident, $index:ident, $array_type:ty) => {{
        let (chunk_idx, idx) = $self.index_to_chunked_index($index);
        let arr = {
            let arr = $self.chunks.get_unchecked(chunk_idx);
            &*(arr as *const ArrayRef as *const Arc<$array_type>)
        };
        arr.value_unchecked(idx)
    }};
}

impl<T> TakeRandom for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        unsafe { impl_take_random_get!(self, index, PrimitiveArray<T>) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        impl_take_random_get_unchecked!(self, index, PrimitiveArray<T>)
    }
}

impl<'a, T> TakeRandom for &'a ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        (*self).get(index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        (*self).get_unchecked(index)
    }
}

impl TakeRandom for BooleanChunked {
    type Item = bool;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        // Safety:
        // Out of bounds is checked and downcast is of correct type
        unsafe { impl_take_random_get!(self, index, BooleanArray) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        impl_take_random_get_unchecked!(self, index, BooleanArray)
    }
}

impl TakeRandom for CategoricalChunked {
    type Item = u32;

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.deref().get(index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        self.deref().get_unchecked(index)
    }
}

impl<'a> TakeRandom for &'a Utf8Chunked {
    type Item = &'a str;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        // Safety:
        // Out of bounds is checked and downcast is of correct type
        unsafe { impl_take_random_get!(self, index, LargeStringArray) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        let (chunk_idx, idx) = self.index_to_chunked_index(index);
        let arr = {
            let arr = self.chunks.get_unchecked(chunk_idx);
            &*(arr as *const ArrayRef as *const Arc<LargeStringArray>)
        };
        arr.value_unchecked(idx)
    }
}

// extra trait such that it also works without extra reference.
// Autoref will insert the reference and
impl<'a> TakeRandomUtf8 for &'a Utf8Chunked {
    type Item = &'a str;

    #[inline]
    fn get(self, index: usize) -> Option<Self::Item> {
        // Safety:
        // Out of bounds is checkedn and downcast is of correct type
        unsafe { impl_take_random_get!(self, index, LargeStringArray) }
    }

    #[inline]
    unsafe fn get_unchecked(self, index: usize) -> Self::Item {
        let (chunk_idx, idx) = self.index_to_chunked_index(index);
        let arr = {
            let arr = self.chunks.get_unchecked(chunk_idx);
            &*(arr as *const ArrayRef as *const Arc<LargeStringArray>)
        };
        arr.value_unchecked(idx)
    }
}

#[cfg(feature = "object")]
impl<'a, T: PolarsObject> TakeRandom for &'a ObjectChunked<T> {
    type Item = &'a T;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        // Safety:
        // Out of bounds is checked and downcast is of correct type
        unsafe { impl_take_random_get!(self, index, ObjectArray<T>) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        impl_take_random_get_unchecked!(self, index, ObjectArray<T>)
    }
}

impl TakeRandom for ListChunked {
    type Item = Series;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        // Safety:
        // Out of bounds is checked and downcast is of correct type
        let opt_arr = unsafe { impl_take_random_get!(self, index, LargeListArray) };
        opt_arr.map(|arr| {
            let s = Series::try_from((self.name(), arr));
            s.unwrap()
        })
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        let (chunk_idx, idx) = self.index_to_chunked_index(index);
        let arr = {
            let arr = self.chunks.get_unchecked(chunk_idx);
            &*(arr as *const ArrayRef as *const Arc<LargeListArray>)
        };
        let arr = arr.value_unchecked(idx);
        let s = Series::try_from((self.name(), arr));
        s.unsafe_unwrap()
    }
}
