#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectArray;
use crate::prelude::downcast::Chunks;
use crate::prelude::*;
use crate::utils::arrow::array::LargeStringArray;
use arrow::array::{Array, BooleanArray, LargeListArray, PrimitiveArray};
use polars_arrow::is_valid::*;
use std::convert::TryFrom;

macro_rules! take_random_get {
    ($self:ident, $index:ident) => {{
        let (chunk_idx, arr_idx) =
            crate::utils::index_to_chunked_index($self.chunk_lens.iter().copied(), $index as u32);

        // Safety:
        // bounds are checked above
        let arr = unsafe { $self.chunks.get_unchecked(chunk_idx as usize) };

        if arr.is_null(arr_idx as usize) {
            None
        } else {
            // SAFETY:
            // bounds checked above
            unsafe { Some(arr.value_unchecked(arr_idx as usize)) }
        }
    }};
}

macro_rules! take_random_get_unchecked {
    ($self:ident, $index:ident) => {{
        let (chunk_idx, arr_idx) =
            crate::utils::index_to_chunked_index($self.chunk_lens.iter().copied(), $index as u32);

        // Safety:
        // bounds are checked above
        let arr = $self.chunks.get_unchecked(chunk_idx as usize);

        if arr.is_null_unchecked(arr_idx as usize) {
            None
        } else {
            // SAFETY:
            // bounds checked above
            Some(arr.value_unchecked(arr_idx as usize))
        }
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

    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
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

    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
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
    pub(crate) chunks: Chunks<'a, LargeStringArray>,
    pub(crate) chunk_lens: Vec<u32>,
}

impl<'a> TakeRandom for Utf8TakeRandom<'a> {
    type Item = &'a str;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        take_random_get_unchecked!(self, index)
    }
}

pub struct Utf8TakeRandomSingleChunk<'a> {
    pub(crate) arr: &'a LargeStringArray,
}

impl<'a> TakeRandom for Utf8TakeRandomSingleChunk<'a> {
    type Item = &'a str;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get_single!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        if self.arr.is_valid_unchecked(index) {
            Some(self.arr.value_unchecked(index))
        } else {
            None
        }
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
    pub(crate) chunks: Vec<&'a PrimitiveArray<T>>,
    pub(crate) chunk_lens: Vec<u32>,
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
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        take_random_get_unchecked!(self, index)
    }
}

pub struct NumTakeRandomCont<'a, T> {
    pub(crate) slice: &'a [T],
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
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        Some(*self.slice.get_unchecked(index))
    }
}

pub struct NumTakeRandomSingleChunk<'a, T>
where
    T: PolarsNumericType,
{
    pub(crate) arr: &'a PrimitiveArray<T>,
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
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        if self.arr.is_valid_unchecked(index) {
            Some(self.arr.value_unchecked(index))
        } else {
            None
        }
    }
}

pub struct BoolTakeRandom<'a> {
    pub(crate) chunks: Chunks<'a, BooleanArray>,
    pub(crate) chunk_lens: Vec<u32>,
}

impl<'a> TakeRandom for BoolTakeRandom<'a> {
    type Item = bool;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        take_random_get_unchecked!(self, index)
    }
}

pub struct BoolTakeRandomSingleChunk<'a> {
    pub(crate) arr: &'a BooleanArray,
}

impl<'a> TakeRandom for BoolTakeRandomSingleChunk<'a> {
    type Item = bool;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get_single!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        if self.arr.is_valid_unchecked(index) {
            Some(self.arr.value_unchecked(index))
        } else {
            None
        }
    }
}

pub struct ListTakeRandom<'a> {
    ca: &'a ListChunked,
    chunks: Vec<&'a LargeListArray>,
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
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        let v = take_random_get_unchecked!(self, index);
        v.map(|v| {
            let s = Series::try_from((self.ca.name(), v));
            s.unwrap()
        })
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
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        if self.arr.is_valid_unchecked(index) {
            let v = self.arr.value_unchecked(index);
            let s = Series::try_from((self.name, v));
            s.ok()
        } else {
            None
        }
    }
}

#[cfg(feature = "object")]
pub struct ObjectTakeRandom<'a, T: PolarsObject> {
    pub(crate) chunks: Chunks<'a, ObjectArray<T>>,
    pub(crate) chunk_lens: Vec<u32>,
}

#[cfg(feature = "object")]
impl<'a, T: PolarsObject> TakeRandom for ObjectTakeRandom<'a, T> {
    type Item = &'a T;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        take_random_get_unchecked!(self, index)
    }
}

#[cfg(feature = "object")]
pub struct ObjectTakeRandomSingleChunk<'a, T: PolarsObject> {
    pub(crate) arr: &'a ObjectArray<T>,
}

#[cfg(feature = "object")]
impl<'a, T: PolarsObject> TakeRandom for ObjectTakeRandomSingleChunk<'a, T> {
    type Item = &'a T;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        take_random_get_single!(self, index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        if self.arr.is_valid_unchecked(index) {
            Some(self.arr.value_unchecked(index))
        } else {
            None
        }
    }
}

#[cfg(feature = "object")]
impl<'a, T: PolarsObject> IntoTakeRandom<'a> for &'a ObjectChunked<T> {
    type Item = &'a T;
    type TakeRandom = TakeRandBranch2<ObjectTakeRandomSingleChunk<'a, T>, ObjectTakeRandom<'a, T>>;

    fn take_rand(&self) -> Self::TakeRandom {
        let chunks = self.downcast_chunks();
        if self.chunks.len() == 1 {
            let t = ObjectTakeRandomSingleChunk {
                arr: chunks.get(0).unwrap(),
            };
            TakeRandBranch2::Single(t)
        } else {
            let t = ObjectTakeRandom {
                chunks,
                chunk_lens: self.chunks.iter().map(|a| a.len() as u32).collect(),
            };
            TakeRandBranch2::Multi(t)
        }
    }
}
