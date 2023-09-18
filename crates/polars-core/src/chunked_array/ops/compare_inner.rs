//! Used to speed up PartialEq and PartialOrd of elements within an array

use std::cmp::{Ordering, PartialEq};

use crate::chunked_array::ops::take::take_random::{
    TakeRandomArray, TakeRandomArrayValues, TakeRandomChunked,
};
use crate::prelude::*;

pub trait PartialEqInner: Send + Sync {
    /// # Safety
    /// Does not do any bound checks.
    unsafe fn eq_element_unchecked(&self, idx_a: usize, idx_b: usize) -> bool;
}

pub trait PartialOrdInner: Send + Sync {
    /// # Safety
    /// Does not do any bound checks.
    unsafe fn cmp_element_unchecked(&self, idx_a: usize, idx_b: usize) -> Ordering;
}

impl<T> PartialEqInner for T
where
    T: TakeRandom + Send + Sync,
    T::Item: PartialEq,
{
    #[inline]
    unsafe fn eq_element_unchecked(&self, idx_a: usize, idx_b: usize) -> bool {
        self.get_unchecked(idx_a) == self.get_unchecked(idx_b)
    }
}

/// Create a type that implements PartialEqInner.
pub(crate) trait IntoPartialEqInner<'a> {
    /// Create a type that implements `TakeRandom`.
    fn into_partial_eq_inner(self) -> Box<dyn PartialEqInner + 'a>;
}

/// We use a trait object because we want to call this from Series and cannot use a typed enum.
impl<'a, T> IntoPartialEqInner<'a> for &'a ChunkedArray<T>
where
    T: PolarsDataType,
    T::Physical<'a>: PartialEq,
{
    fn into_partial_eq_inner(self) -> Box<dyn PartialEqInner + 'a> {
        let mut chunks = self.downcast_iter();

        if self.chunks.len() == 1 {
            let arr = chunks.next().unwrap();

            if !self.has_validity() {
                Box::new(TakeRandomArrayValues::<T> { arr })
            } else {
                Box::new(TakeRandomArray::<T> { arr })
            }
        } else {
            let t = TakeRandomChunked::<T> {
                chunks: chunks.collect(),
                chunk_lens: self.chunks.iter().map(|a| a.len() as IdxSize).collect(),
            };
            Box::new(t)
        }
    }
}

// Partial ordering implementations.
#[inline]
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

impl<T> PartialOrdInner for T
where
    T: TakeRandom + Send + Sync,
    T::Item: PartialOrd,
{
    #[inline]
    unsafe fn cmp_element_unchecked(&self, idx_a: usize, idx_b: usize) -> Ordering {
        let a = self.get_unchecked(idx_a);
        let b = self.get_unchecked(idx_b);
        a.partial_cmp(&b).unwrap_or_else(|| fallback(a))
    }
}

/// Create a type that implements PartialOrdInner.
pub(crate) trait IntoPartialOrdInner<'a> {
    /// Create a type that implements `TakeRandom`.
    fn into_partial_ord_inner(self) -> Box<dyn PartialOrdInner + 'a>;
}

/// We use a trait object because we want to call this from Series and cannot use a typed enum.
impl<'a, T> IntoPartialOrdInner<'a> for &'a ChunkedArray<T>
where
    T: PolarsDataType,
    T::Physical<'a>: PartialOrd,
{
    fn into_partial_ord_inner(self) -> Box<dyn PartialOrdInner + 'a> {
        let mut chunks = self.downcast_iter();

        if self.chunks.len() == 1 {
            let arr = chunks.next().unwrap();

            if !self.has_validity() {
                Box::new(TakeRandomArrayValues::<T> { arr })
            } else {
                Box::new(TakeRandomArray::<T> { arr })
            }
        } else {
            let t = TakeRandomChunked::<T> {
                chunks: chunks.collect(),
                chunk_lens: self.chunks.iter().map(|a| a.len() as IdxSize).collect(),
            };
            Box::new(t)
        }
    }
}

#[cfg(feature = "dtype-categorical")]
impl<'a> IntoPartialOrdInner<'a> for &'a CategoricalChunked {
    fn into_partial_ord_inner(self) -> Box<dyn PartialOrdInner + 'a> {
        match &**self.get_rev_map() {
            RevMapping::Local(_) => Box::new(CategoricalTakeRandomLocal::new(self)),
            RevMapping::Global(_, _, _) => Box::new(CategoricalTakeRandomGlobal::new(self)),
        }
    }
}
