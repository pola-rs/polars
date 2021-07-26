//! Traits that indicate the allowed arguments in a ChunkedArray::take operation.
use crate::prelude::*;
use arrow::array::{Array, UInt32Array};

// Utility traits
pub trait TakeIterator: Iterator<Item = usize> {
    fn check_bounds(&self, bound: usize) -> Result<()>;
}
pub trait TakeIteratorNulls: Iterator<Item = Option<usize>> {
    fn check_bounds(&self, bound: usize) -> Result<()>;
}

// Implement for the ref as well
impl TakeIterator for &mut dyn TakeIterator {
    fn check_bounds(&self, bound: usize) -> Result<()> {
        (**self).check_bounds(bound)
    }
}
impl TakeIteratorNulls for &mut dyn TakeIteratorNulls {
    fn check_bounds(&self, bound: usize) -> Result<()> {
        (**self).check_bounds(bound)
    }
}

// Clonable iterators may implement the traits above
impl<I> TakeIterator for I
where
    I: Iterator<Item = usize> + Clone + Sized,
{
    fn check_bounds(&self, bound: usize) -> Result<()> {
        // clone so that the iterator can be used again.
        let iter = self.clone();
        let mut inbounds = true;

        for i in iter {
            if i >= bound {
                inbounds = false;
                break;
            }
        }
        if inbounds {
            Ok(())
        } else {
            Err(PolarsError::OutOfBounds(
                "take indices are out of bounds".into(),
            ))
        }
    }
}
impl<I> TakeIteratorNulls for I
where
    I: Iterator<Item = Option<usize>> + Clone + Sized,
{
    fn check_bounds(&self, bound: usize) -> Result<()> {
        // clone so that the iterator can be used again.
        let iter = self.clone();
        let mut inbounds = true;

        for i in iter.flatten() {
            if i >= bound {
                inbounds = false;
                break;
            }
        }
        if inbounds {
            Ok(())
        } else {
            Err(PolarsError::OutOfBounds(
                "take indices are out of bounds".into(),
            ))
        }
    }
}

/// One of the three arguments allowed in unchecked_take
pub enum TakeIdx<'a, I, INulls>
where
    I: TakeIterator,
    INulls: TakeIteratorNulls,
{
    Array(&'a UInt32Array),
    Iter(I),
    // will return a null where None
    IterNulls(INulls),
}

impl<'a, I, INulls> TakeIdx<'a, I, INulls>
where
    I: TakeIterator,
    INulls: TakeIteratorNulls,
{
    pub(crate) fn check_bounds(&self, bound: usize) -> Result<()> {
        match self {
            TakeIdx::Iter(i) => i.check_bounds(bound),
            TakeIdx::IterNulls(i) => i.check_bounds(bound),
            TakeIdx::Array(arr) => {
                let mut inbounds = true;
                let len = bound as u32;
                if arr.null_count() == 0 {
                    for &i in arr.values().as_slice() {
                        if i >= len {
                            inbounds = false;
                            break;
                        }
                    }
                } else {
                    for opt_v in *arr {
                        match opt_v {
                            Some(&v) if v >= len => {
                                inbounds = false;
                                break;
                            }
                            _ => {}
                        }
                    }
                }
                if inbounds {
                    Ok(())
                } else {
                    Err(PolarsError::OutOfBounds(
                        "take indices are out of bounds".into(),
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
impl<'a> From<&'a UInt32Chunked> for TakeIdx<'a, Dummy<usize>, Dummy<Option<usize>>> {
    fn from(ca: &'a UInt32Chunked) -> Self {
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
