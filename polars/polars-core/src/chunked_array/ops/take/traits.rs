//! Traits that indicate the allowed arguments in a ChunkedArray::take operation.
use crate::prelude::*;
use arrow::array::UInt32Array;

// Utility traits
pub trait TakeIterator: Iterator<Item = usize> {}
pub trait TakeIteratorNulls: Iterator<Item = Option<usize>> {}

// Implement for the ref as well
impl TakeIterator for &mut dyn TakeIterator {}
impl TakeIteratorNulls for &mut dyn TakeIteratorNulls {}

// Clonable iterators may implement the traits above
impl<I> TakeIterator for I where I: Iterator<Item = usize> + Clone + Sized {}
impl<I> TakeIteratorNulls for I where I: Iterator<Item = Option<usize>> + Clone + Sized {}

/// One of the three arguments allowed in unchecked_take
pub enum TakeIdxUnchecked<'a, I, INulls>
where
    I: TakeIterator,
    INulls: TakeIteratorNulls,
{
    Array(&'a UInt32Array),
    Iter(I),
    // will return a null where None
    IterNulls(INulls),
}

/// One of the two arguments allowed in safe take
/// This one differs from unchecked because we don't allow a take with an optional iterator
pub enum TakeIdx<'a, I>
where
    I: TakeIterator,
{
    Array(&'a UInt32Array),
    Iter(I),
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
impl<'a> From<&'a UInt32Chunked> for TakeIdxUnchecked<'a, Dummy<usize>, Dummy<Option<usize>>> {
    fn from(ca: &'a UInt32Chunked) -> Self {
        if ca.chunks.len() == 1 {
            TakeIdxUnchecked::Array(ca.downcast_iter().next().unwrap())
        } else {
            panic!("implementation error, should be transformed to an iterator by the caller")
        }
    }
}

/// Conversion from Iterator<Item=usize> to Unchecked TakeIdx
impl<'a, I> From<I> for TakeIdxUnchecked<'a, I, Dummy<Option<usize>>>
where
    I: TakeIterator,
{
    fn from(iter: I) -> Self {
        TakeIdxUnchecked::Iter(iter)
    }
}

/// Conversion from Iterator<Item=Option<usize>> to Unchecked TakeIdx
impl<'a, I> From<I> for TakeIdxUnchecked<'a, Dummy<usize>, I>
where
    I: TakeIteratorNulls,
{
    fn from(iter: I) -> Self {
        TakeIdxUnchecked::IterNulls(iter)
    }
}

// Checked conversions

/// Conversion from UInt32Chunked to TakeIdx
impl<'a> From<&'a UInt32Chunked> for TakeIdx<'a, Dummy<usize>> {
    fn from(ca: &'a UInt32Chunked) -> Self {
        if ca.chunks.len() == 1 {
            TakeIdx::Array(ca.downcast_iter().next().unwrap())
        } else {
            panic!("implementation error, should be transformed to an iterator by the caller")
        }
    }
}

/// Conversion from Iterator<Item=usize> to TakeIdx
impl<'a, I> From<I> for TakeIdx<'a, I>
where
    I: TakeIterator,
{
    fn from(iter: I) -> Self {
        TakeIdx::Iter(iter)
    }
}
