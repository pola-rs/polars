//!
//! Used to speed up PartialEq and PartialOrd of elements within an array
//!

use super::take_random::{
    BoolTakeRandom, BoolTakeRandomSingleChunk, NumTakeRandomChunked, NumTakeRandomCont,
    NumTakeRandomSingleChunk, Utf8TakeRandom, Utf8TakeRandomSingleChunk,
};
use crate::prelude::*;
use std::cmp::PartialEq;

pub trait PartialEqInner: Send + Sync {
    /// Safety:
    /// Does not do any bound checks
    unsafe fn eq_element_unchecked(&self, idx_a: usize, idx_b: usize) -> bool;
}

macro_rules! impl_traits {
    ($struct:ty) => {
        impl PartialEqInner for $struct {
            #[inline]
            unsafe fn eq_element_unchecked(&self, idx_a: usize, idx_b: usize) -> bool {
                self.get(idx_a) == self.get(idx_b)
            }
        }
    };
    ($struct:ty, $T:tt) => {
        impl<$T> PartialEqInner for $struct
        where
            $T: PolarsNumericType + Sync,
            $T::Native: Copy + PartialEq,
        {
            #[inline]
            unsafe fn eq_element_unchecked(&self, idx_a: usize, idx_b: usize) -> bool {
                self.get(idx_a) == self.get(idx_b)
            }
        }
    };
}

impl_traits!(Utf8TakeRandom<'_>);
impl_traits!(Utf8TakeRandomSingleChunk<'_>);
impl_traits!(BoolTakeRandom<'_>);
impl_traits!(BoolTakeRandomSingleChunk<'_>);
impl_traits!(NumTakeRandomSingleChunk<'_, T>, T);
impl_traits!(NumTakeRandomChunked<'_, T>, T);

impl<T> PartialEqInner for NumTakeRandomCont<'_, T>
where
    T: Copy + PartialEq + Sync,
{
    unsafe fn eq_element_unchecked(&self, idx_a: usize, idx_b: usize) -> bool {
        // no nulls so we can do unchecked
        self.get_unchecked(idx_a) == self.get_unchecked(idx_b)
    }
}

/// Create a type that implements PartialEqInner
pub(crate) trait IntoPartialEqInner<'a> {
    /// Create a type that implements `TakeRandom`.
    fn into_partial_eq_inner(self) -> Box<dyn PartialEqInner + 'a>;
}

/// We use a trait object because we want to call this from Series and cannot use a typed enum.
impl<'a, T> IntoPartialEqInner<'a> for &'a ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: PartialEq,
{
    fn into_partial_eq_inner(self) -> Box<dyn PartialEqInner + 'a> {
        let mut chunks = self.downcast_iter();

        if self.chunks.len() == 1 {
            let arr = chunks.next().unwrap();

            if self.null_count() == 0 {
                let t = NumTakeRandomCont {
                    slice: arr.values(),
                };
                Box::new(t)
            } else {
                let t = NumTakeRandomSingleChunk::<'_, T> { arr };
                Box::new(t)
            }
        } else {
            let t = NumTakeRandomChunked::<'_, T> {
                chunks: chunks.collect(),
                chunk_lens: self.chunks.iter().map(|a| a.len() as u32).collect(),
            };
            Box::new(t)
        }
    }
}

impl<'a> IntoPartialEqInner<'a> for &'a Utf8Chunked {
    fn into_partial_eq_inner(self) -> Box<dyn PartialEqInner + 'a> {
        match self.chunks.len() {
            1 => {
                let arr = self.downcast_iter().next().unwrap();
                let t = Utf8TakeRandomSingleChunk { arr };
                Box::new(t)
            }
            _ => {
                let chunks = self.downcast_chunks();
                let t = Utf8TakeRandom {
                    chunks,
                    chunk_lens: self.chunks.iter().map(|a| a.len() as u32).collect(),
                };
                Box::new(t)
            }
        }
    }
}

impl<'a> IntoPartialEqInner<'a> for &'a BooleanChunked {
    fn into_partial_eq_inner(self) -> Box<dyn PartialEqInner + 'a> {
        match self.chunks.len() {
            1 => {
                let arr = self.downcast_iter().next().unwrap();
                let t = BoolTakeRandomSingleChunk { arr };
                Box::new(t)
            }
            _ => {
                let chunks = self.downcast_chunks();
                let t = BoolTakeRandom {
                    chunks,
                    chunk_lens: self.chunks.iter().map(|a| a.len() as u32).collect(),
                };
                Box::new(t)
            }
        }
    }
}

impl<'a> IntoPartialEqInner<'a> for &'a ListChunked {
    fn into_partial_eq_inner(self) -> Box<dyn PartialEqInner> {
        unimplemented!()
    }
}

impl<'a> IntoPartialEqInner<'a> for &'a CategoricalChunked {
    fn into_partial_eq_inner(self) -> Box<dyn PartialEqInner> {
        unimplemented!()
    }
}
