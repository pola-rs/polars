//! Traits to provide fast Random access to ChunkedArrays data.
//! This prevents downcasting every iteration.
//! IntoTakeRandom provides structs that implement the TakeRandom trait.
//! There are several structs that implement the fastest path for random access.
//!
use crate::chunked_array::builder::{
    get_list_builder, PrimitiveChunkedBuilder, Utf8ChunkedBuilder,
};
use crate::prelude::*;
use arrow::array::{
    Array, ArrayRef, BooleanArray, ListArray, PrimitiveArray, PrimitiveArrayOps, StringArray,
};
use std::sync::Arc;

macro_rules! impl_take_random_get {
    ($self:ident, $index:ident, $array_type:ty) => {{
        let (chunk_idx, idx) = $self.index_to_chunked_index($index);
        let arr = unsafe {
            let arr = $self.chunks.get_unchecked(chunk_idx);
            &*(arr as *const ArrayRef as *const Arc<$array_type>)
        };
        if arr.is_valid(idx) {
            Some(arr.value(idx))
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
        arr.value(idx)
    }};
}

impl<T> TakeRandom for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    type Item = T::Native;

    fn get(&self, index: usize) -> Option<Self::Item> {
        impl_take_random_get!(self, index, PrimitiveArray<T>)
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        impl_take_random_get_unchecked!(self, index, PrimitiveArray<T>)
    }
}

impl<'a, T> TakeRandom for &'a ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    type Item = T::Native;

    fn get(&self, index: usize) -> Option<Self::Item> {
        (*self).get(index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        (*self).get_unchecked(index)
    }
}

impl<'a> TakeRandom for &'a Utf8Chunked {
    type Item = &'a str;

    fn get(&self, index: usize) -> Option<Self::Item> {
        impl_take_random_get!(self, index, StringArray)
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        impl_take_random_get_unchecked!(self, index, StringArray)
    }
}

// extra trait such that it also works without extra reference.
// Autoref will insert the refererence and
impl<'a> TakeRandomUtf8 for &'a Utf8Chunked {
    type Item = &'a str;

    fn get(self, index: usize) -> Option<Self::Item> {
        impl_take_random_get!(self, index, StringArray)
    }

    unsafe fn get_unchecked(self, index: usize) -> Self::Item {
        impl_take_random_get_unchecked!(self, index, StringArray)
    }
}

impl TakeRandom for ListChunked {
    type Item = Series;

    fn get(&self, index: usize) -> Option<Self::Item> {
        let opt_arr = impl_take_random_get!(self, index, ListArray);
        opt_arr.map(|arr| (self.name(), arr).into())
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        let arr = impl_take_random_get_unchecked!(self, index, ListArray);
        (self.name(), arr).into()
    }
}

macro_rules! impl_take {
    ($self:ident, $indices:ident, $capacity:ident, $builder:ident) => {{
        let capacity = $capacity.unwrap_or($indices.size_hint().0);
        let mut builder = $builder::new($self.name(), capacity);

        let taker = $self.take_rand();
        for idx in $indices {
            match taker.get(idx) {
                Some(v) => builder.append_value(v),
                None => builder.append_null(),
            }
        }
        builder.finish()
    }};
}

macro_rules! impl_take_opt {
    ($self:ident, $indices:ident, $capacity:ident, $builder:ident) => {{
        let capacity = $capacity.unwrap_or($indices.size_hint().0);
        let mut builder = $builder::new($self.name(), capacity);
        let taker = $self.take_rand();

        for opt_idx in $indices {
            match opt_idx {
                Some(idx) => match taker.get(idx) {
                    Some(v) => builder.append_value(v),
                    None => builder.append_null(),
                },
                None => builder.append_null(),
            };
        }
        builder.finish()
    }};
}

macro_rules! impl_take_opt_unchecked {
    ($self:ident, $indices:ident, $capacity:ident, $builder:ident) => {{
        let capacity = $capacity.unwrap_or($indices.size_hint().0);
        let mut builder = $builder::new($self.name(), capacity);
        let taker = $self.take_rand();

        for opt_idx in $indices {
            match opt_idx {
                Some(idx) => {
                    let v = taker.get_unchecked(idx);
                    builder.append_value(v);
                }
                None => builder.append_null(),
            };
        }
        builder.finish()
    }};
}

macro_rules! impl_take_unchecked {
    ($self:ident, $indices:ident, $capacity:ident, $builder:ident) => {{
        let capacity = $capacity.unwrap_or($indices.size_hint().0);
        let mut builder = $builder::new($self.name(), capacity);

        let taker = $self.take_rand();
        for idx in $indices {
            let v = taker.get_unchecked(idx);
            builder.append_value(v);
        }
        builder.finish()
    }};
}

impl<T> ChunkTake for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn take(&self, indices: impl Iterator<Item = usize>, capacity: Option<usize>) -> Self {
        impl_take!(self, indices, capacity, PrimitiveChunkedBuilder)
    }

    unsafe fn take_unchecked(
        &self,
        indices: impl Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Self {
        impl_take_unchecked!(self, indices, capacity, PrimitiveChunkedBuilder)
    }

    fn take_opt(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        impl_take_opt!(self, indices, capacity, PrimitiveChunkedBuilder)
    }

    unsafe fn take_opt_unchecked(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        impl_take_opt_unchecked!(self, indices, capacity, PrimitiveChunkedBuilder)
    }
}

impl ChunkTake for BooleanChunked {
    fn take(&self, indices: impl Iterator<Item = usize>, capacity: Option<usize>) -> Self
    where
        Self: std::marker::Sized,
    {
        impl_take!(self, indices, capacity, PrimitiveChunkedBuilder)
    }

    unsafe fn take_unchecked(
        &self,
        indices: impl Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Self {
        impl_take_unchecked!(self, indices, capacity, PrimitiveChunkedBuilder)
    }

    fn take_opt(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        impl_take_opt!(self, indices, capacity, PrimitiveChunkedBuilder)
    }

    unsafe fn take_opt_unchecked(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        impl_take_opt_unchecked!(self, indices, capacity, PrimitiveChunkedBuilder)
    }
}

impl ChunkTake for Utf8Chunked {
    fn take(&self, indices: impl Iterator<Item = usize>, capacity: Option<usize>) -> Self
    where
        Self: std::marker::Sized,
    {
        impl_take!(self, indices, capacity, Utf8ChunkedBuilder)
    }

    unsafe fn take_unchecked(
        &self,
        indices: impl Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Self {
        impl_take_unchecked!(self, indices, capacity, Utf8ChunkedBuilder)
    }

    fn take_opt(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self
    where
        Self: std::marker::Sized,
    {
        impl_take_opt!(self, indices, capacity, Utf8ChunkedBuilder)
    }

    unsafe fn take_opt_unchecked(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        impl_take_opt_unchecked!(self, indices, capacity, Utf8ChunkedBuilder)
    }
}

impl ChunkTake for ListChunked {
    fn take(&self, indices: impl Iterator<Item = usize>, capacity: Option<usize>) -> Self {
        let capacity = capacity.unwrap_or(indices.size_hint().0);

        match self.dtype() {
            ArrowDataType::List(dt) => {
                let mut builder = get_list_builder(&**dt, capacity, self.name());
                let taker = self.take_rand();

                for idx in indices {
                    builder.append_opt_series(&taker.get(idx));
                }
                builder.finish()
            }
            _ => unimplemented!(),
        }
    }

    unsafe fn take_unchecked(
        &self,
        indices: impl Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Self {
        let capacity = capacity.unwrap_or(indices.size_hint().0);
        match self.dtype() {
            ArrowDataType::List(dt) => {
                let mut builder = get_list_builder(&**dt, capacity, self.name());
                let taker = self.take_rand();
                for idx in indices {
                    let v = taker.get_unchecked(idx);
                    builder.append_opt_series(&Some(v));
                }
                builder.finish()
            }
            _ => unimplemented!(),
        }
    }

    fn take_opt(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        let capacity = capacity.unwrap_or(indices.size_hint().0);

        match self.dtype() {
            ArrowDataType::List(dt) => {
                let mut builder = get_list_builder(&**dt, capacity, self.name());

                let taker = self.take_rand();

                for opt_idx in indices {
                    match opt_idx {
                        Some(idx) => {
                            let opt_s = taker.get(idx);
                            builder.append_opt_series(&opt_s)
                        }
                        None => builder.append_opt_series(&None),
                    };
                }
                builder.finish()
            }
            _ => unimplemented!(),
        }
    }

    unsafe fn take_opt_unchecked(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        let capacity = capacity.unwrap_or(indices.size_hint().0);

        match self.dtype() {
            ArrowDataType::List(dt) => {
                let mut builder = get_list_builder(&**dt, capacity, self.name());
                let taker = self.take_rand();

                for opt_idx in indices {
                    match opt_idx {
                        Some(idx) => {
                            let s = taker.get_unchecked(idx);
                            builder.append_opt_series(&Some(s))
                        }
                        None => builder.append_opt_series(&None),
                    };
                }
                builder.finish()
            }
            _ => unimplemented!(),
        }
    }
}

pub trait AsTakeIndex {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a>;

    fn as_opt_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = Option<usize>> + 'a> {
        unimplemented!()
    }

    fn take_index_len(&self) -> usize;
}

impl AsTakeIndex for &UInt32Chunked {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> {
        match self.cont_slice() {
            Ok(slice) => Box::new(slice.iter().map(|&val| val as usize)),
            Err(_) => Box::new(
                self.into_iter()
                    .filter_map(|opt_val| opt_val.map(|val| val as usize)),
            ),
        }
    }
    fn as_opt_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = Option<usize>> + 'a> {
        Box::new(
            self.into_iter()
                .map(|opt_val| opt_val.map(|val| val as usize)),
        )
    }
    fn take_index_len(&self) -> usize {
        self.len()
    }
}

impl<T> AsTakeIndex for T
where
    T: AsRef<[usize]>,
{
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> {
        Box::new(self.as_ref().iter().copied())
    }
    fn take_index_len(&self) -> usize {
        self.as_ref().len()
    }
}

impl AsTakeIndex for [u32] {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> {
        Box::new(self.iter().map(|&v| v as usize))
    }
    fn take_index_len(&self) -> usize {
        self.len()
    }
}

/// Create a type that implements a faster `TakeRandom`.
pub trait IntoTakeRandom<'a> {
    type Item;
    type TakeRandom;
    /// Create a type that implements `TakeRandom`.
    fn take_rand(&self) -> Self::TakeRandom;
}

/// Choose the Struct for multiple chunks or the struct for a single chunk.
macro_rules! many_or_single {
    ($self:ident, $StructSingle:ident, $StructMany:ident) => {{
        let chunks = $self.downcast_chunks();
        if chunks.len() == 1 {
            Box::new($StructSingle { arr: chunks[0] })
        } else {
            Box::new($StructMany { ca: $self, chunks })
        }
    }};
}

pub enum NumTakeRandomDispatch<'a, T>
where
    T: PolarsNumericType,
    T::Native: Copy,
{
    Cont(NumTakeRandomCont<'a, T::Native>),
    Single(NumTakeRandomSingleChunk<'a, T>),
    Many(NumTakeRandomChunked<'a, T>),
}

impl<'a, T> TakeRandom for NumTakeRandomDispatch<'a, T>
where
    T: PolarsNumericType,
    T::Native: Copy,
{
    type Item = T::Native;

    fn get(&self, index: usize) -> Option<Self::Item> {
        use NumTakeRandomDispatch::*;
        match self {
            Cont(a) => a.get(index),
            Single(a) => a.get(index),
            Many(a) => a.get(index),
        }
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        use NumTakeRandomDispatch::*;
        match self {
            Cont(a) => a.get_unchecked(index),
            Single(a) => a.get_unchecked(index),
            Many(a) => a.get_unchecked(index),
        }
    }
}

impl<'a, T> IntoTakeRandom<'a> for &'a ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;
    type TakeRandom = NumTakeRandomDispatch<'a, T>;

    fn take_rand(&self) -> Self::TakeRandom {
        match self.cont_slice() {
            Ok(slice) => NumTakeRandomDispatch::Cont(NumTakeRandomCont { slice }),
            _ => {
                let chunks = self.downcast_chunks();
                if chunks.len() == 1 {
                    NumTakeRandomDispatch::Single(NumTakeRandomSingleChunk { arr: chunks[0] })
                } else {
                    NumTakeRandomDispatch::Many(NumTakeRandomChunked { ca: self, chunks })
                }
            }
        }
    }
}

impl<'a> IntoTakeRandom<'a> for &'a Utf8Chunked {
    type Item = &'a str;
    type TakeRandom = Box<dyn TakeRandom<Item = Self::Item> + 'a>;

    fn take_rand(&self) -> Self::TakeRandom {
        many_or_single!(self, Utf8TakeRandomSingleChunk, Utf8TakeRandom)
    }
}

impl<'a> IntoTakeRandom<'a> for &'a BooleanChunked {
    type Item = bool;
    type TakeRandom = Box<dyn TakeRandom<Item = Self::Item> + 'a>;

    fn take_rand(&self) -> Self::TakeRandom {
        many_or_single!(self, BoolTakeRandomSingleChunk, BoolTakeRandom)
    }
}

impl<'a> IntoTakeRandom<'a> for &'a ListChunked {
    type Item = Series;
    type TakeRandom = Box<dyn TakeRandom<Item = Self::Item> + 'a>;

    fn take_rand(&self) -> Self::TakeRandom {
        let chunks = self.downcast_chunks();
        if chunks.len() == 1 {
            Box::new(ListTakeRandomSingleChunk {
                arr: chunks[0],
                name: self.name(),
            })
        } else {
            Box::new(ListTakeRandom { ca: self, chunks })
        }
    }
}

pub struct NumTakeRandomChunked<'a, T>
where
    T: PolarsNumericType,
{
    ca: &'a ChunkedArray<T>,
    chunks: Vec<&'a PrimitiveArray<T>>,
}

macro_rules! take_random_get {
    ($self:ident, $index:ident) => {{
        let (chunk_idx, arr_idx) = $self.ca.index_to_chunked_index($index);
        let arr = $self.chunks.get(chunk_idx);
        match arr {
            Some(arr) => {
                if arr.is_null(arr_idx) {
                    None
                } else {
                    Some(arr.value(arr_idx))
                }
            }
            None => None,
        }
    }};
}

macro_rules! take_random_get_unchecked {
    ($self:ident, $index:ident) => {{
        let (chunk_idx, arr_idx) = $self.ca.index_to_chunked_index($index);
        $self.chunks.get_unchecked(chunk_idx).value(arr_idx)
    }};
}

macro_rules! take_random_get_single {
    ($self:ident, $index:ident) => {{
        if $self.arr.is_null($index) {
            None
        } else {
            Some($self.arr.value($index))
        }
    }};
}

impl<'a, T> TakeRandom for NumTakeRandomChunked<'a, T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;

    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get!(self, index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        take_random_get_unchecked!(self, index)
    }
}

pub struct NumTakeRandomCont<'a, T> {
    slice: &'a [T],
}

impl<'a, T> TakeRandom for NumTakeRandomCont<'a, T>
where
    T: Copy,
{
    type Item = T;

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.slice.get(index).copied()
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        *self.slice.get_unchecked(index)
    }
}

pub struct NumTakeRandomSingleChunk<'a, T>
where
    T: PolarsNumericType,
{
    arr: &'a PrimitiveArray<T>,
}

impl<'a, T> TakeRandom for NumTakeRandomSingleChunk<'a, T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;

    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get_single!(self, index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        self.arr.value(index)
    }
}

pub struct Utf8TakeRandom<'a> {
    ca: &'a Utf8Chunked,
    chunks: Vec<&'a StringArray>,
}

impl<'a> TakeRandom for Utf8TakeRandom<'a> {
    type Item = &'a str;

    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get!(self, index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        take_random_get_unchecked!(self, index)
    }
}

pub struct Utf8TakeRandomSingleChunk<'a> {
    arr: &'a StringArray,
}

impl<'a> TakeRandom for Utf8TakeRandomSingleChunk<'a> {
    type Item = &'a str;

    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get_single!(self, index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        self.arr.value(index)
    }
}

pub struct BoolTakeRandom<'a> {
    ca: &'a BooleanChunked,
    chunks: Vec<&'a BooleanArray>,
}

impl<'a> TakeRandom for BoolTakeRandom<'a> {
    type Item = bool;

    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get!(self, index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        take_random_get_unchecked!(self, index)
    }
}

pub struct BoolTakeRandomSingleChunk<'a> {
    arr: &'a BooleanArray,
}

impl<'a> TakeRandom for BoolTakeRandomSingleChunk<'a> {
    type Item = bool;

    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get_single!(self, index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        self.arr.value(index)
    }
}
pub struct ListTakeRandom<'a> {
    ca: &'a ListChunked,
    chunks: Vec<&'a ListArray>,
}

impl<'a> TakeRandom for ListTakeRandom<'a> {
    type Item = Series;

    fn get(&self, index: usize) -> Option<Self::Item> {
        let v = take_random_get!(self, index);
        v.map(|v| (self.ca.name(), v).into())
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        let v = take_random_get_unchecked!(self, index);
        (self.ca.name(), v).into()
    }
}

pub struct ListTakeRandomSingleChunk<'a> {
    arr: &'a ListArray,
    name: &'a str,
}

impl<'a> TakeRandom for ListTakeRandomSingleChunk<'a> {
    type Item = Series;

    fn get(&self, index: usize) -> Option<Self::Item> {
        let v = take_random_get_single!(self, index);
        v.map(|v| (self.name, v).into())
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        (self.name, self.arr.value(index)).into()
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_take_random() {
        let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
        assert_eq!(ca.get(0), Some(1));
        assert_eq!(ca.get(1), Some(2));
        assert_eq!(ca.get(2), Some(3));

        let ca = Utf8Chunked::new_from_slice("a", &["a", "b", "c"]);
        assert_eq!(ca.get(0), Some("a"));
        assert_eq!(ca.get(1), Some("b"));
        assert_eq!(ca.get(2), Some("c"));
    }
}
