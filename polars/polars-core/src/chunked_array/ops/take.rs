//! Traits to provide fast Random access to ChunkedArrays data.
//! This prevents downcasting every iteration.
//! IntoTakeRandom provides structs that implement the TakeRandom trait.
//! There are several structs that implement the fastest path for random access.
//!
use crate::chunked_array::kernels::take::{
    take_bool_iter, take_bool_iter_unchecked, take_bool_opt_iter_unchecked, take_no_null_bool_iter,
    take_no_null_bool_iter_unchecked, take_no_null_bool_opt_iter_unchecked, take_no_null_primitive,
    take_no_null_primitive_iter, take_no_null_primitive_iter_unchecked,
    take_no_null_primitive_opt_iter_unchecked, take_no_null_utf8_iter,
    take_no_null_utf8_iter_unchecked, take_no_null_utf8_opt_iter_unchecked, take_primitive_iter,
    take_primitive_iter_n_chunks, take_primitive_iter_unchecked, take_primitive_opt_iter_n_chunks,
    take_primitive_opt_iter_unchecked, take_utf8, take_utf8_iter, take_utf8_iter_unchecked,
    take_utf8_opt_iter_unchecked,
};
use crate::chunked_array::ops::downcast::Chunks;
use crate::prelude::*;
use crate::utils::NoNull;
use arrow::array::{
    Array, ArrayRef, BooleanArray, LargeListArray, LargeStringArray, PrimitiveArray,
};
use arrow::compute::kernels::take::take;
use std::convert::TryFrom;
use std::ops::Deref;
use unsafe_unwrap::UnsafeUnwrap;

macro_rules! take_iter_n_chunks {
    ($ca:expr, $indices:expr) => {{
        let taker = $ca.take_rand();
        $indices.into_iter().map(|idx| taker.get(idx)).collect()
    }};
}

macro_rules! take_opt_iter_n_chunks {
    ($ca:expr, $indices:expr) => {{
        let taker = $ca.take_rand();
        $indices
            .into_iter()
            .map(|opt_idx| opt_idx.and_then(|idx| taker.get(idx)))
            .collect()
    }};
}

impl<T> ChunkTake for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => take_no_null_primitive(chunks.next().unwrap(), array) as ArrayRef,
                    (_, 1) => take(chunks.next().unwrap(), array, None).unwrap(),
                    _ => {
                        return if array.null_count() == 0 {
                            let iter = array.values().iter().map(|i| *i as usize);
                            let mut ca = take_primitive_iter_n_chunks(self, iter);
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| idx as usize));
                            let mut ca = take_primitive_opt_iter_n_chunks(self, iter);
                            ca.rename(self.name());
                            ca
                        }
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => take_no_null_primitive_iter_unchecked(chunks.next().unwrap(), iter)
                        as ArrayRef,
                    (_, 1) => {
                        take_primitive_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    _ => {
                        let mut ca = take_primitive_iter_n_chunks(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::IterNulls(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => {
                        take_no_null_primitive_opt_iter_unchecked(chunks.next().unwrap(), iter)
                            as ArrayRef
                    }
                    (_, 1) => {
                        take_primitive_opt_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    _ => {
                        let mut ca = take_primitive_opt_iter_n_chunks(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
        }
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take(chunks.next().unwrap(), array, None).unwrap(),
                    _ => {
                        let iter = array
                            .into_iter()
                            .filter_map(|opt_idx| opt_idx.map(|idx| idx as usize));

                        let mut ca = take_primitive_iter_n_chunks(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => take_no_null_primitive_iter(chunks.next().unwrap(), iter) as ArrayRef,
                    (_, 1) => take_primitive_iter(chunks.next().unwrap(), iter) as ArrayRef,
                    _ => {
                        let mut ca = take_primitive_iter_n_chunks(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::IterNulls(_) => {
                panic!("not supported in take, only supported in take_unchecked for the join operation")
            }
        }
    }
}

impl ChunkTake for BooleanChunked {
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take(chunks.next().unwrap(), array, None).unwrap(),
                    _ => {
                        return if array.null_count() == 0 {
                            let iter = array.values().iter().map(|i| *i as usize);
                            let mut ca: BooleanChunked = take_iter_n_chunks!(self, iter);
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| idx as usize));
                            let mut ca: BooleanChunked = take_opt_iter_n_chunks!(self, iter);
                            ca.rename(self.name());
                            ca
                        }
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => {
                        take_no_null_bool_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    (_, 1) => take_bool_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef,
                    _ => {
                        let mut ca: BooleanChunked = take_iter_n_chunks!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::IterNulls(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => take_no_null_bool_opt_iter_unchecked(chunks.next().unwrap(), iter)
                        as ArrayRef,
                    (_, 1) => {
                        take_bool_opt_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    _ => {
                        let mut ca: BooleanChunked = take_opt_iter_n_chunks!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
        }
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take(chunks.next().unwrap(), array, None).unwrap(),
                    _ => {
                        let iter = array.values().iter().map(|i| *i as usize);
                        let mut ca: BooleanChunked = take_iter_n_chunks!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => take_no_null_bool_iter(chunks.next().unwrap(), iter) as ArrayRef,
                    (_, 1) => take_bool_iter(chunks.next().unwrap(), iter) as ArrayRef,
                    _ => {
                        let mut ca: BooleanChunked = take_iter_n_chunks!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::IterNulls(_) => {
                panic!("not supported in take, only supported in take_unchecked for the join operation")
            }
        }
    }
}

impl ChunkTake for Utf8Chunked {
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take_utf8(chunks.next().unwrap(), array) as ArrayRef,
                    _ => {
                        return if array.null_count() == 0 {
                            let iter = array.values().iter().map(|i| *i as usize);
                            let mut ca: Utf8Chunked = take_iter_n_chunks!(self, iter);
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| idx as usize));
                            let mut ca: Utf8Chunked = take_opt_iter_n_chunks!(self, iter);
                            ca.rename(self.name());
                            ca
                        }
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => {
                        take_no_null_utf8_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    (_, 1) => take_utf8_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef,
                    _ => {
                        let mut ca: Utf8Chunked = take_iter_n_chunks!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::IterNulls(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => take_no_null_utf8_opt_iter_unchecked(chunks.next().unwrap(), iter)
                        as ArrayRef,
                    (_, 1) => {
                        take_utf8_opt_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    _ => {
                        let mut ca: Utf8Chunked = take_opt_iter_n_chunks!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
        }
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take(chunks.next().unwrap(), array, None).unwrap() as ArrayRef,
                    _ => {
                        let iter = array.values().iter().map(|i| *i as usize);
                        let mut ca: Utf8Chunked = take_iter_n_chunks!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => take_no_null_utf8_iter(chunks.next().unwrap(), iter) as ArrayRef,
                    (_, 1) => take_utf8_iter(chunks.next().unwrap(), iter) as ArrayRef,
                    _ => {
                        let mut ca: Utf8Chunked = take_iter_n_chunks!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::IterNulls(_) => {
                panic!("not supported in take, only supported in take_unchecked for the join operation")
            }
        }
    }
}

impl ChunkTake for ListChunked {
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        self.take(indices)
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take(chunks.next().unwrap(), array, None).unwrap() as ArrayRef,
                    _ => {
                        let ca = self.rechunk();
                        return ca.take(indices);
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let idx_ca = iter
                    .map(|idx| idx as u32)
                    .collect::<NoNull<UInt32Chunked>>()
                    .into_inner();
                self.take((&idx_ca).into())
            }
            TakeIdx::IterNulls(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let idx_ca = iter
                    .map(|opt_idx| opt_idx.map(|idx| idx as u32))
                    .collect::<UInt32Chunked>();
                self.take((&idx_ca).into())
            }
        }
    }
}

impl ChunkTake for CategoricalChunked {
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let ca: CategoricalChunked = self.deref().take_unchecked(indices).into();
        ca.set_state(self)
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let ca: CategoricalChunked = self.deref().take(indices).into();
        ca.set_state(self)
    }
}

#[cfg(feature = "object")]
impl<T> ChunkTake for ObjectChunked<T> {
    unsafe fn take_unchecked<I, INulls>(&self, _indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        unimplemented!()
    }

    fn take<I, INulls>(&self, _indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        unimplemented!()
    }
}

pub trait AsTakeIndex {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a>;

    fn as_opt_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = Option<usize>> + 'a> {
        unimplemented!()
    }

    fn take_index_len(&self) -> usize;
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
        if $self.chunks.len() == 1 {
            Box::new($StructSingle {
                arr: $self.downcast_iter().next().unwrap(),
            })
        } else {
            Box::new($StructMany {
                ca: $self,
                chunks: $self.downcast_chunks(),
            })
        }
    }};
}

impl<'a, T> IntoTakeRandom<'a> for &'a ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;
    type TakeRandom = Box<dyn TakeRandom<Item = Self::Item> + 'a>;

    fn take_rand(&self) -> Self::TakeRandom {
        match self.cont_slice() {
            Ok(slice) => Box::new(NumTakeRandomCont { slice }),
            _ => {
                let mut chunks = self.downcast_iter();
                if self.chunks.len() == 1 {
                    Box::new(NumTakeRandomSingleChunk {
                        arr: chunks.next().unwrap(),
                    })
                } else {
                    Box::new(NumTakeRandomChunked {
                        ca: self,
                        chunks: chunks.collect(),
                    })
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
        let mut chunks = self.downcast_iter();
        if self.chunks.len() == 1 {
            Box::new(ListTakeRandomSingleChunk {
                arr: chunks.next().unwrap(),
                name: self.name(),
            })
        } else {
            Box::new(ListTakeRandom {
                ca: self,
                chunks: chunks.collect(),
            })
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
                    // SAFETY:
                    // bounds checked above
                    unsafe { Some(arr.value_unchecked(arr_idx)) }
                }
            }
            None => None,
        }
    }};
}

macro_rules! take_random_get_unchecked {
    ($self:ident, $index:ident) => {{
        let (chunk_idx, arr_idx) = $self.ca.index_to_chunked_index($index);
        $self
            .chunks
            .get_unchecked(chunk_idx)
            .value_unchecked(arr_idx)
    }};
}

macro_rules! take_random_get_single {
    ($self:ident, $index:ident) => {{
        if $self.arr.is_null($index) {
            None
        } else {
            // Safety:
            // bound checked above
            unsafe { Some($self.arr.value_unchecked($index)) }
        }
    }};
}

impl<'a, T> TakeRandom for NumTakeRandomChunked<'a, T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get!(self, index)
    }

    #[inline]
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

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        self.slice.get(index).copied()
    }

    #[inline]
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

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get_single!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        self.arr.value_unchecked(index)
    }
}

pub struct Utf8TakeRandom<'a> {
    ca: &'a Utf8Chunked,
    chunks: Chunks<'a, LargeStringArray>,
}

impl<'a> TakeRandom for Utf8TakeRandom<'a> {
    type Item = &'a str;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        let (chunk_idx, arr_idx) = self.ca.index_to_chunked_index(index);
        self.chunks
            .get_unchecked(chunk_idx)
            .value_unchecked(arr_idx)
    }
}

pub struct Utf8TakeRandomSingleChunk<'a> {
    arr: &'a LargeStringArray,
}

impl<'a> TakeRandom for Utf8TakeRandomSingleChunk<'a> {
    type Item = &'a str;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get_single!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        self.arr.value_unchecked(index)
    }
}

pub struct BoolTakeRandom<'a> {
    ca: &'a BooleanChunked,
    chunks: Chunks<'a, BooleanArray>,
}

impl<'a> TakeRandom for BoolTakeRandom<'a> {
    type Item = bool;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        take_random_get_unchecked!(self, index)
    }
}

pub struct BoolTakeRandomSingleChunk<'a> {
    arr: &'a BooleanArray,
}

impl<'a> TakeRandom for BoolTakeRandomSingleChunk<'a> {
    type Item = bool;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get_single!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        self.arr.value(index)
    }
}
pub struct ListTakeRandom<'a> {
    ca: &'a ListChunked,
    chunks: Vec<&'a LargeListArray>,
}

impl<'a> TakeRandom for ListTakeRandom<'a> {
    type Item = Series;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        let v = take_random_get!(self, index);
        v.map(|v| {
            let s = Series::try_from((self.ca.name(), v));
            s.unwrap()
        })
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        let v = take_random_get_unchecked!(self, index);
        let s = Series::try_from((self.ca.name(), v));
        s.unwrap()
    }
}

pub struct ListTakeRandomSingleChunk<'a> {
    arr: &'a LargeListArray,
    name: &'a str,
}

impl<'a> TakeRandom for ListTakeRandomSingleChunk<'a> {
    type Item = Series;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        let v = take_random_get_single!(self, index);
        v.map(|v| {
            let s = Series::try_from((self.name, v));
            s.unwrap()
        })
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        let s = Series::try_from((self.name, self.arr.value_unchecked(index)));
        s.unsafe_unwrap()
    }
}

impl<T> ChunkTakeEvery<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn take_every(&self, n: usize) -> ChunkedArray<T> {
        if self.null_count() == 0 {
            let a: NoNull<_> = self.into_no_null_iter().step_by(n).collect();
            a.into_inner()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<BooleanType> for BooleanChunked {
    fn take_every(&self, n: usize) -> BooleanChunked {
        if self.null_count() == 0 {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<Utf8Type> for Utf8Chunked {
    fn take_every(&self, n: usize) -> Utf8Chunked {
        if self.null_count() == 0 {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<ListType> for ListChunked {
    fn take_every(&self, n: usize) -> ListChunked {
        if self.null_count() == 0 {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<CategoricalType> for CategoricalChunked {
    fn take_every(&self, n: usize) -> CategoricalChunked {
        let mut ca = if self.null_count() == 0 {
            let ca: NoNull<UInt32Chunked> = self.into_no_null_iter().step_by(n).collect();
            ca.into_inner()
        } else {
            self.into_iter().step_by(n).collect()
        };
        ca.categorical_map = self.categorical_map.clone();
        ca.cast().unwrap()
    }
}
#[cfg(feature = "object")]
impl<T> ChunkTakeEvery<ObjectType<T>> for ObjectChunked<T> {
    fn take_every(&self, _n: usize) -> ObjectChunked<T> {
        todo!()
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
