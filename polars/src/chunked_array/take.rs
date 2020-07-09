use crate::chunked_array::builder::{PrimitiveChunkedBuilder, Utf8ChunkedBuilder};
use crate::prelude::*;
use arrow::array::{Array, BooleanArray, PrimitiveArray, StringArray};

pub trait Take {
    fn take(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Result<Self>
    where
        Self: std::marker::Sized;
}

macro_rules! impl_take_builder {
    ($self:ident, $indices:ident, $builder:ident, $chunks:ident) => {{
        for opt_idx in $indices {
            match opt_idx {
                Some(idx) => {
                    let (chunk_idx, i) = $self.index_to_chunked_index(idx);
                    let arr = unsafe { $chunks.get_unchecked(chunk_idx) };
                    $builder.append_value(arr.value(i))?
                }
                None => $builder.append_null()?,
            }
        }
        Ok($builder.finish())
    }};
}

impl<T> Take for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn take(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Result<Self> {
        let capacity = capacity.unwrap_or(1024);
        let mut builder = PrimitiveChunkedBuilder::new(self.name(), capacity);

        let chunks = self.downcast_chunks();
        if let Ok(slice) = self.cont_slice() {
            for opt_idx in indices {
                match opt_idx {
                    Some(idx) => builder.append_value(slice[idx])?,
                    None => builder.append_null()?,
                };
            }
            Ok(builder.finish())
        } else {
            impl_take_builder!(self, indices, builder, chunks)
        }
    }
}

impl Take for BooleanChunked {
    fn take(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Result<Self> {
        let capacity = capacity.unwrap_or(1024);
        let mut builder = PrimitiveChunkedBuilder::new(self.name(), capacity);
        let chunks = self.downcast_chunks();
        impl_take_builder!(self, indices, builder, chunks)
    }
}

impl Take for Utf8Chunked {
    fn take(
        &self,
        indices: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        let capacity = capacity.unwrap_or(1024);
        let mut builder = Utf8ChunkedBuilder::new(self.name(), capacity);
        let chunks = self.downcast_chunks();
        impl_take_builder!(self, indices, builder, chunks)
    }
}

pub trait TakeIndex {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = Option<usize>> + 'a>;

    fn take_index_len(&self) -> usize;
}

impl TakeIndex for UInt32Chunked {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = Option<usize>> + 'a> {
        Box::new(
            self.into_iter()
                .map(|opt_val| opt_val.map(|val| val as usize)),
        )
    }
    fn take_index_len(&self) -> usize {
        self.len()
    }
}

impl TakeIndex for [usize] {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = Option<usize>> + 'a> {
        Box::new(self.iter().map(|&v| Some(v)))
    }
    fn take_index_len(&self) -> usize {
        self.len()
    }
}

impl TakeIndex for Vec<usize> {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = Option<usize>> + 'a> {
        Box::new(self.iter().map(|&v| Some(v)))
    }
    fn take_index_len(&self) -> usize {
        self.len()
    }
}

impl TakeIndex for [u32] {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = Option<usize>> + 'a> {
        Box::new(self.iter().map(|&v| Some(v as usize)))
    }
    fn take_index_len(&self) -> usize {
        self.len()
    }
}

// Traits to provide fast Random access to ChunkedArrays data.
// This prevents downcasting every iteration.
// IntoTakeRandom provides structs that implement the TakeRandom trait.
// There are several structs that implement the fastest path for random access.

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
