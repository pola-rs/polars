//! Traits to provide fast Random access to ChunkedArrays data.
//! This prevents downcasting every iteration.
//! IntoTakeRandom provides structs that implement the TakeRandom trait.
//! There are several structs that implement the fastest path for random access.
//!
use crate::chunked_array::builder::{PrimitiveChunkedBuilder, Utf8ChunkedBuilder};
use crate::prelude::*;
use arrow::array::{Array, BooleanArray, PrimitiveArray, StringArray};

pub trait Take {
    /// Take values from ChunkedArray by index.
    fn take(&self, indices: impl Iterator<Item = usize>, capacity: Option<usize>) -> Result<Self>
    where
        Self: std::marker::Sized;

    /// Take values from ChunkedArray by Option<index>.
    fn take_opt(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Result<Self>
    where
        Self: std::marker::Sized;
}

macro_rules! impl_take {
    ($self:ident, $indices:ident, $capacity:ident, $builder:ident) => {{
        let capacity = $capacity.unwrap_or($indices.size_hint().0);
        let mut builder = $builder::new($self.name(), capacity);

        let taker = $self.take_rand();
        for idx in $indices {
            match taker.get(idx) {
                Some(v) => builder.append_value(v)?,
                None => builder.append_null()?,
            }
        }
        Ok(builder.finish())
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
                    Some(v) => builder.append_value(v)?,
                    None => builder.append_null()?,
                },
                None => builder.append_null()?,
            };
        }
        Ok(builder.finish())
    }};
}

impl<T> Take for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn take(&self, indices: impl Iterator<Item = usize>, capacity: Option<usize>) -> Result<Self> {
        impl_take!(self, indices, capacity, PrimitiveChunkedBuilder)
    }

    fn take_opt(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Result<Self> {
        impl_take_opt!(self, indices, capacity, PrimitiveChunkedBuilder)
    }
}

impl Take for BooleanChunked {
    fn take(&self, indices: impl Iterator<Item = usize>, capacity: Option<usize>) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        impl_take!(self, indices, capacity, PrimitiveChunkedBuilder)
    }

    fn take_opt(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Result<Self> {
        impl_take_opt!(self, indices, capacity, PrimitiveChunkedBuilder)
    }
}

impl Take for Utf8Chunked {
    fn take(&self, indices: impl Iterator<Item = usize>, capacity: Option<usize>) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        impl_take!(self, indices, capacity, Utf8ChunkedBuilder)
    }

    fn take_opt(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        impl_take_opt!(self, indices, capacity, Utf8ChunkedBuilder)
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
            Ok(slice) => Box::new(slice.into_iter().map(|&val| val as usize)),
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

impl AsTakeIndex for [usize] {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> {
        Box::new(self.iter().filter_map(|&v| Some(v)))
    }
    fn take_index_len(&self) -> usize {
        self.len()
    }
}

impl AsTakeIndex for Vec<usize> {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> {
        Box::new(self.iter().copied())
    }
    fn take_index_len(&self) -> usize {
        self.len()
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

pub trait IntoTakeRandom<'a> {
    type Item;
    fn take_rand(&self) -> Box<dyn TakeRandom<Item = Self::Item> + 'a>;
}

/// Choose the Struct for multiple chunks or the struct for a single chunk.
macro_rules! many_or_single {
    ($self:ident, $StructSingle:ident, $StructMany:ident) => {{
        let chunks = $self.downcast_chunks();
        if chunks.len() == 1 {
            Box::new($StructSingle { arr: chunks[0] })
        } else {
            Box::new($StructMany {
                ca: $self,
                chunks: chunks,
            })
        }
    }};
}

impl<'a, T> IntoTakeRandom<'a> for &'a ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;

    fn take_rand(&self) -> Box<dyn TakeRandom<Item = Self::Item> + 'a> {
        match self.cont_slice() {
            Ok(slice) => Box::new(NumTakeRandomCont { slice }),
            _ => many_or_single!(self, NumTakeRandomSingleChunk, NumTakeRandomChunked),
        }
    }
}

impl<'a> IntoTakeRandom<'a> for &'a Utf8Chunked {
    type Item = &'a str;

    fn take_rand(&self) -> Box<dyn TakeRandom<Item = Self::Item> + 'a> {
        many_or_single!(self, Utf8TakeRandomSingleChunk, Utf8TakeRandom)
    }
}

impl<'a> IntoTakeRandom<'a> for &'a BooleanChunked {
    type Item = bool;

    fn take_rand(&self) -> Box<dyn TakeRandom<Item = Self::Item> + 'a> {
        many_or_single!(self, BoolTakeRandomSingleChunk, BoolTakeRandom)
    }
}

pub trait TakeRandom {
    type Item;
    fn get(&self, index: usize) -> Option<Self::Item>;

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item;
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
        self.slice.get(index).map(|v| *v)
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
