//! Traits that indicate the allowed arguments in a ChunkedArray::take operation.
use crate::frame::groupby::GroupsProxyIter;
use crate::prelude::*;

// Utility traits
pub trait TakeIterator: Iterator<Item = usize> + TrustedLen {
    fn check_bounds(&self, bound: usize) -> PolarsResult<()>;
    // a sort of clone
    fn boxed_clone(&self) -> Box<dyn TakeIterator + '_>;
}
pub trait TakeIteratorNulls: Iterator<Item = Option<usize>> + TrustedLen {
    fn check_bounds(&self, bound: usize) -> PolarsResult<()>;

    fn boxed_clone(&self) -> Box<dyn TakeIteratorNulls + '_>;
}

unsafe impl TrustedLen for &mut dyn TakeIterator {}
unsafe impl TrustedLen for &mut dyn TakeIteratorNulls {}
unsafe impl TrustedLen for GroupsProxyIter<'_> {}

// Implement for the ref as well
impl TakeIterator for &mut dyn TakeIterator {
    fn check_bounds(&self, bound: usize) -> PolarsResult<()> {
        (**self).check_bounds(bound)
    }

    fn boxed_clone(&self) -> Box<dyn TakeIterator + '_> {
        (**self).boxed_clone()
    }
}
impl TakeIteratorNulls for &mut dyn TakeIteratorNulls {
    fn check_bounds(&self, bound: usize) -> PolarsResult<()> {
        (**self).check_bounds(bound)
    }

    fn boxed_clone(&self) -> Box<dyn TakeIteratorNulls + '_> {
        (**self).boxed_clone()
    }
}

// Clonable iterators may implement the traits above
impl<I> TakeIterator for I
where
    I: Iterator<Item = usize> + Clone + Sized + TrustedLen,
{
    fn check_bounds(&self, bound: usize) -> PolarsResult<()> {
        // clone so that the iterator can be used again.
        let iter = self.clone();
        let mut inbounds = true;

        for i in iter {
            if i >= bound {
                // we will not break here as that prevents SIMD
                inbounds = false;
            }
        }
        if inbounds {
            Ok(())
        } else {
            Err(PolarsError::ComputeError(
                "Take indices are out of bounds.".into(),
            ))
        }
    }

    fn boxed_clone(&self) -> Box<dyn TakeIterator + '_> {
        Box::new(self.clone())
    }
}
impl<I> TakeIteratorNulls for I
where
    I: Iterator<Item = Option<usize>> + Clone + Sized + TrustedLen,
{
    fn check_bounds(&self, bound: usize) -> PolarsResult<()> {
        // clone so that the iterator can be used again.
        let iter = self.clone();
        let mut inbounds = true;

        for i in iter.flatten() {
            if i >= bound {
                // we will not break here as that prevents SIMD
                inbounds = false;
            }
        }
        if inbounds {
            Ok(())
        } else {
            Err(PolarsError::ComputeError(
                "Take indices are out of bounds.".into(),
            ))
        }
    }

    fn boxed_clone(&self) -> Box<dyn TakeIteratorNulls + '_> {
        Box::new(self.clone())
    }
}

/// One of the three arguments allowed in unchecked_take
pub enum TakeIdx<'a, I, INulls>
where
    I: TakeIterator,
    INulls: TakeIteratorNulls,
{
    Array(&'a IdxArr),
    Iter(I),
    // will return a null where None
    IterNulls(INulls),
}

impl<'a, I, INulls> TakeIdx<'a, I, INulls>
where
    I: TakeIterator,
    INulls: TakeIteratorNulls,
{
    pub(crate) fn check_bounds(&self, bound: usize) -> PolarsResult<()> {
        match self {
            TakeIdx::Iter(i) => i.check_bounds(bound),
            TakeIdx::IterNulls(i) => i.check_bounds(bound),
            TakeIdx::Array(arr) => {
                let values = arr.values().as_slice();
                let mut inbounds = true;
                let len = bound as IdxSize;
                if arr.null_count() == 0 {
                    for &i in values {
                        // we will not break here as that prevents SIMD
                        if i >= len {
                            inbounds = false;
                        }
                    }
                } else {
                    for opt_v in *arr {
                        match opt_v {
                            Some(&v) if v >= len => {
                                inbounds = false;
                            }
                            _ => {}
                        }
                    }
                }
                if inbounds {
                    Ok(())
                } else {
                    Err(PolarsError::ComputeError(
                        "Take indices are out of bounds.".into(),
                    ))
                }
            }
        }
    }
}

/// Dummy type, we need to instantiate all generic types, so we fill one with a dummy.
pub type Dummy<T> = std::iter::Once<T>;

// Below the conversions from
// * UInt32Chunked
// * Iterator<Item=usize>
// * Iterator<Item=Option<usize>>
//
// To the checked and unchecked TakeIdx enums

// Unchecked conversions

/// Conversion from UInt32Chunked to Unchecked TakeIdx
impl<'a> From<&'a IdxCa> for TakeIdx<'a, Dummy<usize>, Dummy<Option<usize>>> {
    fn from(ca: &'a IdxCa) -> Self {
        if ca.chunks.len() == 1 {
            TakeIdx::Array(ca.downcast_iter().next().unwrap())
        } else {
            panic!("implementation error, should be transformed to an iterator by the caller")
        }
    }
}

/// Conversion from Iterator<Item=usize> to Unchecked TakeIdx
impl<'a, I> From<I> for TakeIdx<'a, I, Dummy<Option<usize>>>
where
    I: TakeIterator,
{
    fn from(iter: I) -> Self {
        TakeIdx::Iter(iter)
    }
}

/// Conversion from Iterator<Item=Option<usize>> to Unchecked TakeIdx
impl<'a, I> From<I> for TakeIdx<'a, Dummy<usize>, I>
where
    I: TakeIteratorNulls,
{
    fn from(iter: I) -> Self {
        TakeIdx::IterNulls(iter)
    }
}

#[inline]
fn to_usize(idx: &IdxSize) -> usize {
    *idx as usize
}

/// Conversion from `&[IdxSize]` to Unchecked TakeIdx
impl<'a> From<&'a [IdxSize]>
    for TakeIdx<
        'a,
        std::iter::Map<std::slice::Iter<'a, IdxSize>, fn(&IdxSize) -> usize>,
        Dummy<Option<usize>>,
    >
{
    fn from(slice: &'a [IdxSize]) -> Self {
        TakeIdx::Iter(slice.iter().map(to_usize))
    }
}

/// Conversion from `&[IdxSize]` to Unchecked TakeIdx
impl<'a> From<&'a Vec<IdxSize>>
    for TakeIdx<
        'a,
        std::iter::Map<std::slice::Iter<'a, IdxSize>, fn(&IdxSize) -> usize>,
        Dummy<Option<usize>>,
    >
{
    fn from(slice: &'a Vec<IdxSize>) -> Self {
        (&**slice).into()
    }
}
