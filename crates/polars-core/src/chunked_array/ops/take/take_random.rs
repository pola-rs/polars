use arrow::array::Array;

use crate::prelude::*;
use crate::utils::index_to_chunked_index;

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

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        match self {
            Self::SingleNoNull(s) => s.get(index),
            Self::Single(s) => s.get(index),
            Self::Multi(m) => m.get(index),
        }
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        match self {
            Self::SingleNoNull(s) => s.get_unchecked(index),
            Self::Single(s) => s.get_unchecked(index),
            Self::Multi(m) => m.get_unchecked(index),
        }
    }

    fn last(&self) -> Option<Self::Item> {
        match self {
            Self::SingleNoNull(s) => s.last(),
            Self::Single(s) => s.last(),
            Self::Multi(m) => m.last(),
        }
    }
}

#[allow(clippy::type_complexity)]
impl<'a, T> IntoTakeRandom<'a> for &'a ChunkedArray<T>
where
    T: PolarsDataType,
{
    type Item = T::Physical<'a>;
    type TakeRandom = TakeRandBranch3<
        NumTakeRandomCont<'a, T>,
        NumTakeRandomSingleChunk<'a, T>,
        NumTakeRandomChunked<'a, T>,
    >;

    #[inline]
    fn take_rand(&self) -> Self::TakeRandom {
        let mut chunks = self.downcast_iter();

        if self.chunks.len() == 1 {
            let arr = chunks.next().unwrap();

            if !self.has_validity() {
                let t = NumTakeRandomCont { arr };
                TakeRandBranch3::SingleNoNull(t)
            } else {
                let t = NumTakeRandomSingleChunk { arr };
                TakeRandBranch3::Single(t)
            }
        } else {
            let t = NumTakeRandomChunked {
                chunks: chunks.collect(),
                chunk_lens: self.chunks.iter().map(|a| a.len() as IdxSize).collect(),
            };
            TakeRandBranch3::Multi(t)
        }
    }
}

pub struct NumTakeRandomChunked<'a, T>
where
    T: PolarsDataType,
{
    pub(crate) chunks: Vec<&'a T::Array>,
    pub(crate) chunk_lens: Vec<IdxSize>,
}

impl<'a, T> TakeRandom for NumTakeRandomChunked<'a, T>
where
    T: PolarsDataType,
{
    type Item = T::Physical<'a>;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        let (chunk_idx, arr_idx) =
            index_to_chunked_index(self.chunk_lens.iter().copied(), index as IdxSize);

        let arr = self.chunks.get(chunk_idx as usize)?;

        // SAFETY: if index_to_chunked_index returns a valid chunk_idx, we know
        // that arr_idx < arr.len().
        unsafe { arr.get_unchecked(arr_idx as usize) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        let (chunk_idx, arr_idx) =
            index_to_chunked_index(self.chunk_lens.iter().copied(), index as IdxSize);

        unsafe {
            // SAFETY: up to the caller to make sure the index is valid.
            self.chunks
                .get_unchecked(chunk_idx as usize)
                .get_unchecked(arr_idx as usize)
        }
    }

    fn last(&self) -> Option<Self::Item> {
        self.chunks
            .last()
            .and_then(|arr| arr.get(arr.len().saturating_sub(1)))
    }
}

pub struct NumTakeRandomCont<'a, T: PolarsDataType> {
    pub(crate) arr: &'a T::Array,
}

impl<'a, T> TakeRandom for NumTakeRandomCont<'a, T>
where
    T: PolarsDataType,
{
    type Item = T::Physical<'a>;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        Some(self.arr.value(index))
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        Some(self.arr.value_unchecked(index))
    }

    fn last(&self) -> Option<Self::Item> {
        self.arr.last()
    }
}

pub struct NumTakeRandomSingleChunk<'a, T: PolarsDataType> {
    pub(crate) arr: &'a T::Array,
}

impl<'a, T> TakeRandom for NumTakeRandomSingleChunk<'a, T>
where
    T: PolarsDataType,
{
    type Item = T::Physical<'a>;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        self.arr.get(index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> Option<Self::Item> {
        self.arr.get_unchecked(index)
    }

    fn last(&self) -> Option<Self::Item> {
        self.arr.last()
    }
}
