#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectArray;
use crate::prelude::downcast::Chunks;
use crate::prelude::*;
use arrow::array::{Array, BooleanArray, ListArray, PrimitiveArray, Utf8Array};
use std::convert::TryFrom;
use unsafe_unwrap::UnsafeUnwrap;

macro_rules! take_random_get {
    ($self:ident, $index:ident) => {{
        let (chunk_idx, arr_idx) =
            crate::utils::index_to_chunked_index($self.chunk_lens.iter().copied(), $index as u32);
        let arr = $self.chunks.get(chunk_idx as usize);
        match arr {
            Some(arr) => {
                if arr.is_null(arr_idx as usize) {
                    None
                } else {
                    // SAFETY:
                    // bounds checked above
                    unsafe { Some(arr.value_unchecked(arr_idx as usize)) }
                }
            }
            None => None,
        }
    }};
}

macro_rules! take_random_get_unchecked {
    ($self:ident, $index:ident) => {{
        let (chunk_idx, arr_idx) =
            crate::utils::index_to_chunked_index($self.chunk_lens.iter().copied(), $index as u32);
        $self
            .chunks
            .get_unchecked(chunk_idx as usize)
            .value_unchecked(arr_idx as usize)
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

/// Create a type that implements a faster `TakeRandom`.
pub trait IntoTakeRandom<'a> {
    type Item;
    type TakeRandom;
    /// Create a type that implements `TakeRandom`.
    fn take_rand(&self) -> Self::TakeRandom;
}

pub enum TakeRandBranch3<N, S, M> {
    SingleNoNull(N),
    Single(S),
    Multi(M),
}

impl<N, S, M, I> TakeRandom for TakeRandBranch3<N, S, M>
where
    N: TakeRandom<Item = I>,
    S: TakeRandom<Item = I>,
    M: TakeRandom<Item = I>,
{
    type Item = I;

    fn get(&self, index: usize) -> Option<Self::Item> {
        match self {
            Self::SingleNoNull(s) => s.get(index),
            Self::Single(s) => s.get(index),
            Self::Multi(m) => m.get(index),
        }
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        match self {
            Self::SingleNoNull(s) => s.get_unchecked(index),
            Self::Single(s) => s.get_unchecked(index),
            Self::Multi(m) => m.get_unchecked(index),
        }
    }
}

pub enum TakeRandBranch2<S, M> {
    Single(S),
    Multi(M),
}

impl<S, M, I> TakeRandom for TakeRandBranch2<S, M>
where
    S: TakeRandom<Item = I>,
    M: TakeRandom<Item = I>,
{
    type Item = I;

    fn get(&self, index: usize) -> Option<Self::Item> {
        match self {
            Self::Single(s) => s.get(index),
            Self::Multi(m) => m.get(index),
        }
    }

    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        match self {
            Self::Single(s) => s.get_unchecked(index),
            Self::Multi(m) => m.get_unchecked(index),
        }
    }
}

impl<'a, T> IntoTakeRandom<'a> for &'a ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;
    type TakeRandom = TakeRandBranch3<
        NumTakeRandomCont<'a, T::Native>,
        NumTakeRandomSingleChunk<'a, T>,
        NumTakeRandomChunked<'a, T>,
    >;

    #[inline]
    fn take_rand(&self) -> Self::TakeRandom {
        let mut chunks = self.downcast_iter();

        if self.chunks.len() == 1 {
            let arr = chunks.next().unwrap();

            if self.null_count() == 0 {
                let t = NumTakeRandomCont {
                    slice: arr.values(),
                };
                TakeRandBranch3::SingleNoNull(t)
            } else {
                let t = NumTakeRandomSingleChunk { arr };
                TakeRandBranch3::Single(t)
            }
        } else {
            let t = NumTakeRandomChunked {
                chunks: chunks.collect(),
                chunk_lens: self.chunks.iter().map(|a| a.len() as u32).collect(),
            };
            TakeRandBranch3::Multi(t)
        }
    }
}

pub struct Utf8TakeRandom<'a> {
    chunks: Chunks<'a, ListArray<i64>>,
    chunk_lens: Vec<u32>,
}

impl<'a> TakeRandom for Utf8TakeRandom<'a> {
    type Item = &'a str;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        take_random_get_unchecked!(self, index)
    }
}

pub struct Utf8TakeRandomSingleChunk<'a> {
    arr: &'a Utf8Array<i64>,
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

impl<'a> IntoTakeRandom<'a> for &'a Utf8Chunked {
    type Item = &'a str;
    type TakeRandom = TakeRandBranch2<Utf8TakeRandomSingleChunk<'a>, Utf8TakeRandom<'a>>;

    fn take_rand(&self) -> Self::TakeRandom {
        match self.chunks.len() {
            1 => {
                let arr = self.downcast_iter().next().unwrap();
                let t = Utf8TakeRandomSingleChunk { arr };
                TakeRandBranch2::Single(t)
            }
            _ => {
                let chunks = self.downcast_chunks();
                let t = Utf8TakeRandom {
                    chunks,
                    chunk_lens: self.chunks.iter().map(|a| a.len() as u32).collect(),
                };
                TakeRandBranch2::Multi(t)
            }
        }
    }
}

impl<'a> IntoTakeRandom<'a> for &'a BooleanChunked {
    type Item = bool;
    type TakeRandom = TakeRandBranch2<BoolTakeRandomSingleChunk<'a>, BoolTakeRandom<'a>>;

    fn take_rand(&self) -> Self::TakeRandom {
        match self.chunks.len() {
            1 => {
                let arr = self.downcast_iter().next().unwrap();
                let t = BoolTakeRandomSingleChunk { arr };
                TakeRandBranch2::Single(t)
            }
            _ => {
                let chunks = self.downcast_chunks();
                let t = BoolTakeRandom {
                    chunks,
                    chunk_lens: self.chunks.iter().map(|a| a.len() as u32).collect(),
                };
                TakeRandBranch2::Multi(t)
            }
        }
    }
}

impl<'a> IntoTakeRandom<'a> for &'a ListChunked {
    type Item = Series;
    type TakeRandom = TakeRandBranch2<ListTakeRandomSingleChunk<'a>, ListTakeRandom<'a>>;

    fn take_rand(&self) -> Self::TakeRandom {
        let mut chunks = self.downcast_iter();
        if self.chunks.len() == 1 {
            let t = ListTakeRandomSingleChunk {
                arr: chunks.next().unwrap(),
                name: self.name(),
            };
            TakeRandBranch2::Single(t)
        } else {
            let t = ListTakeRandom {
                ca: self,
                chunks: chunks.collect(),
                chunk_lens: self.chunks.iter().map(|a| a.len() as u32).collect(),
            };
            TakeRandBranch2::Multi(t)
        }
    }
}

pub struct NumTakeRandomChunked<'a, T>
where
    T: PolarsNumericType,
{
    chunks: Vec<&'a PrimitiveArray<T>>,
    chunk_lens: Vec<u32>,
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

pub struct BoolTakeRandom<'a> {
    chunks: Chunks<'a, BooleanArray>,
    chunk_lens: Vec<u32>,
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
    chunks: Vec<&'a ListArray<i64>>,
    chunk_lens: Vec<u32>,
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
        s.unsafe_unwrap()
    }
}

pub struct ListTakeRandomSingleChunk<'a> {
    arr: &'a ListArray<i64>,
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

#[cfg(feature = "object")]
pub struct ObjectTakeRandom<'a, T: PolarsObject> {
    chunks: Vec<&'a ObjectArray<T>>,
    chunk_lens: Vec<u32>,
}

#[cfg(feature = "object")]
impl<'a, T: PolarsObject> TakeRandom for ObjectTakeRandom<'a, T> {
    type Item = &'a T;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        take_random_get_unchecked!(self, index)
    }
}

#[cfg(feature = "object")]
pub struct ObjectTakeRandomSingleChunk<'a, T: PolarsObject> {
    arr: &'a ObjectArray<T>,
}

#[cfg(feature = "object")]
impl<'a, T: PolarsObject> TakeRandom for ObjectTakeRandomSingleChunk<'a, T> {
    type Item = &'a T;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get_single!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Item {
        self.arr.value(index)
    }
}

#[cfg(feature = "object")]
impl<'a, T: PolarsObject> IntoTakeRandom<'a> for &'a ObjectChunked<T> {
    type Item = &'a T;
    type TakeRandom = TakeRandBranch2<ObjectTakeRandomSingleChunk<'a, T>, ObjectTakeRandom<'a, T>>;

    fn take_rand(&self) -> Self::TakeRandom {
        let mut chunks = self.downcast_iter();
        if self.chunks.len() == 1 {
            let t = ObjectTakeRandomSingleChunk {
                arr: chunks.next().unwrap(),
            };
            TakeRandBranch2::Single(t)
        } else {
            let t = ObjectTakeRandom {
                chunks: chunks.collect(),
                chunk_lens: self.chunks.iter().map(|a| a.len() as u32).collect(),
            };
            TakeRandBranch2::Multi(t)
        }
    }
}
