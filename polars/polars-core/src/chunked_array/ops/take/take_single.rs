use std::convert::TryFrom;
use std::process::exit;

use arrow::array::*;
use polars_arrow::is_valid::IsValid;

#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectArray;
use crate::prelude::*;

macro_rules! impl_take_random_get {
    ($self:ident, $index:ident, $array_type:ty) => {{
        if $index >= $self.len() {
            dbg!($self, $index);
            exit(1);
        }
        assert!($index < $self.len());
        let (chunk_idx, idx) = $self.index_to_chunked_index($index);
        // Safety:
        // bounds are checked above
        let arr = $self.chunks.get_unchecked(chunk_idx);

        // Safety:
        // caller should give right array type
        let arr = &*(arr as *const ArrayRef as *const Box<$array_type>);

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
        debug_assert!(chunk_idx < $self.chunks.len());
        // Safety:
        // bounds are checked above
        let arr = $self.chunks.get_unchecked(chunk_idx);

        // Safety:
        // caller should give right array type
        let arr = &*(&**arr as *const dyn Array as *const $array_type);

        // Safety:
        // index should be in bounds
        debug_assert!(idx < arr.len());
        if arr.is_valid_unchecked(idx) {
            Some(arr.value_unchecked(idx))
        } else {
            None
        }
    }};
}

impl<T> TakeRandom for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        unsafe { impl_take_random_get!(self, index, PrimitiveArray<T::Native>) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        impl_take_random_get_unchecked!(self, index, PrimitiveArray<T::Native>)
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
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
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
}

impl<'a> TakeRandom for &'a Utf8Chunked {
    type Item = &'a str;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        // Safety:
        // Out of bounds is checked and downcast is of correct type
        unsafe { impl_take_random_get!(self, index, LargeStringArray) }
    }
}

#[cfg(feature = "dtype-binary")]
impl<'a> TakeRandom for &'a BinaryChunked {
    type Item = &'a [u8];

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        // Safety:
        // Out of bounds is checked and downcast is of correct type
        unsafe { impl_take_random_get!(self, index, LargeBinaryArray) }
    }
}

// extra trait such that it also works without extra reference.
// Autoref will insert the reference and
impl<'a> TakeRandomUtf8 for &'a Utf8Chunked {
    type Item = &'a str;

    #[inline]
    fn get(self, index: usize) -> Option<Self::Item> {
        // Safety:
        // Out of bounds is checked and downcast is of correct type
        unsafe { impl_take_random_get!(self, index, LargeStringArray) }
    }

    #[inline]
    unsafe fn get_unchecked(self, index: usize) -> Option<Self::Item> {
        impl_take_random_get_unchecked!(self, index, LargeStringArray)
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
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
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
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        let opt_arr = impl_take_random_get_unchecked!(self, index, LargeListArray);
        opt_arr.map(|arr| {
            let s = Series::try_from((self.name(), arr));
            s.unwrap()
        })
    }
}
