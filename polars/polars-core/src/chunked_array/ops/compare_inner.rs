//!
//! Used to speed up PartialEq and PartialOrd of elements within an array
//!

use crate::chunked_array::ops::take::take_random::{
    BoolTakeRandom, BoolTakeRandomSingleChunk, NumTakeRandomChunked, NumTakeRandomCont,
    NumTakeRandomSingleChunk, Utf8TakeRandom, Utf8TakeRandomSingleChunk,
};
#[cfg(feature = "object")]
use crate::chunked_array::ops::take::take_random::{ObjectTakeRandom, ObjectTakeRandomSingleChunk};
use crate::prelude::*;
use std::cmp::{Ordering, PartialEq};

pub trait PartialEqInner: Send + Sync {
    /// Safety:
    /// Does not do any bound checks
    unsafe fn eq_element_unchecked(&self, idx_a: usize, idx_b: usize) -> bool;
}

pub trait PartialOrdInner: Send + Sync {
    /// Safety:
    /// Does not do any bound checks
    unsafe fn cmp_element_unchecked(&self, idx_a: usize, idx_b: usize) -> Ordering;
}

macro_rules! impl_traits {
    ($struct:ty) => {
        impl PartialEqInner for $struct {
            #[inline]
            unsafe fn eq_element_unchecked(&self, idx_a: usize, idx_b: usize) -> bool {
                self.get(idx_a) == self.get(idx_b)
            }
        }
        impl PartialOrdInner for $struct {
            #[inline]
            unsafe fn cmp_element_unchecked(&self, idx_a: usize, idx_b: usize) -> Ordering {
                let a = self.get(idx_a);
                let b = self.get(idx_b);
                a.partial_cmp(&b).unwrap_or_else(|| fallback(a))
            }
        }
    };
    ($struct:ty, $T:tt) => {
        impl<$T> PartialEqInner for $struct
        where
            $T: PolarsNumericType + Sync,
        {
            #[inline]
            unsafe fn eq_element_unchecked(&self, idx_a: usize, idx_b: usize) -> bool {
                self.get(idx_a) == self.get(idx_b)
            }
        }

        impl<$T> PartialOrdInner for $struct
        where
            $T: PolarsNumericType + Sync,
        {
            #[inline]
            unsafe fn cmp_element_unchecked(&self, idx_a: usize, idx_b: usize) -> Ordering {
                // nulls so we can not do unchecked
                let a = self.get(idx_a);
                let b = self.get(idx_b);
                a.partial_cmp(&b).unwrap_or_else(|| fallback(a))
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

#[cfg(feature = "dtype-categorical")]
impl<'a> IntoPartialEqInner<'a> for &'a CategoricalChunked {
    fn into_partial_eq_inner(self) -> Box<dyn PartialEqInner> {
        unimplemented!()
    }
}

// Partial ordering implementations

fn fallback<T: PartialEq>(a: T) -> Ordering {
    // nan != nan
    // this is a simple way to check if it is nan
    // without convincing the compiler we deal with floats
    #[allow(clippy::eq_op)]
    if a != a {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

impl<T> PartialOrdInner for NumTakeRandomCont<'_, T>
where
    T: Copy + PartialOrd + Sync,
{
    unsafe fn cmp_element_unchecked(&self, idx_a: usize, idx_b: usize) -> Ordering {
        // no nulls so we can do unchecked
        let a = self.get_unchecked(idx_a);
        let b = self.get_unchecked(idx_b);
        a.partial_cmp(&b).unwrap_or_else(|| fallback(a))
    }
}
/// Create a type that implements PartialOrdInner
pub(crate) trait IntoPartialOrdInner<'a> {
    /// Create a type that implements `TakeRandom`.
    fn into_partial_ord_inner(self) -> Box<dyn PartialOrdInner + 'a>;
}
/// We use a trait object because we want to call this from Series and cannot use a typed enum.
impl<'a, T> IntoPartialOrdInner<'a> for &'a ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn into_partial_ord_inner(self) -> Box<dyn PartialOrdInner + 'a> {
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

impl<'a> IntoPartialOrdInner<'a> for &'a Utf8Chunked {
    fn into_partial_ord_inner(self) -> Box<dyn PartialOrdInner + 'a> {
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

impl<'a> IntoPartialOrdInner<'a> for &'a BooleanChunked {
    fn into_partial_ord_inner(self) -> Box<dyn PartialOrdInner + 'a> {
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

impl<'a> IntoPartialOrdInner<'a> for &'a ListChunked {
    fn into_partial_ord_inner(self) -> Box<dyn PartialOrdInner> {
        unimplemented!()
    }
}

#[cfg(feature = "dtype-categorical")]
impl<'a> IntoPartialOrdInner<'a> for &'a CategoricalChunked {
    fn into_partial_ord_inner(self) -> Box<dyn PartialOrdInner> {
        unimplemented!()
    }
}

#[cfg(feature = "object")]
impl<'a, T> PartialEqInner for ObjectTakeRandom<'a, T>
where
    T: PolarsObject,
{
    #[inline]
    unsafe fn eq_element_unchecked(&self, idx_a: usize, idx_b: usize) -> bool {
        self.get(idx_a) == self.get(idx_b)
    }
}

#[cfg(feature = "object")]
impl<'a, T> PartialEqInner for ObjectTakeRandomSingleChunk<'a, T>
where
    T: PolarsObject,
{
    #[inline]
    unsafe fn eq_element_unchecked(&self, idx_a: usize, idx_b: usize) -> bool {
        self.get(idx_a) == self.get(idx_b)
    }
}

#[cfg(feature = "object")]
impl<'a, T: PolarsObject> IntoPartialEqInner<'a> for &'a ObjectChunked<T> {
    fn into_partial_eq_inner(self) -> Box<dyn PartialEqInner + 'a> {
        match self.chunks.len() {
            1 => {
                let arr = self.downcast_iter().next().unwrap();
                let t = ObjectTakeRandomSingleChunk { arr };
                Box::new(t)
            }
            _ => {
                let chunks = self.downcast_chunks();
                let t = ObjectTakeRandom {
                    chunks,
                    chunk_lens: self.chunks.iter().map(|a| a.len() as u32).collect(),
                };
                Box::new(t)
            }
        }
    }
}
