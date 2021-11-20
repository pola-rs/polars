#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectArray;
use crate::prelude::*;
use arrow::array::*;
use polars_arrow::is_valid::IsValid;
use std::convert::TryFrom;
#[cfg(feature = "dtype-categorical")]
use std::ops::Deref;
use std::sync::Arc;

macro_rules! impl_take_random_get {
    ($self:ident, $index:ident, $array_type:ty) => {{
        assert!($index < $self.len());
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
        // Safety:
        // bounds are checked above
        let arr = $self.chunks.get_unchecked(chunk_idx);

        // Safety:
        // caller should give right array type
        let arr = &*(arr as *const ArrayRef as *const Arc<$array_type>);

        // Safety:
        // index should be in bounds
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

#[cfg(feature = "dtype-categorical")]
impl TakeRandom for CategoricalChunked {
    type Item = u32;

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.deref().get(index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[should_panic]
    fn test_oob() {
        let data: Series = [1.0, 2.0, 3.0].iter().collect();
        let data = data.f64().unwrap();
        let matches = data.equal(5.0);
        let matches_indexes = matches.arg_true();
        matches_indexes.get(0);
    }
}
