//! Traits that indicate the allowed arguments in a ChunkedArray::take operation.
use crate::prelude::*;
use arrow::array::UInt32Array;

// Utility traits
pub trait TakeIterator: Iterator<Item = usize> {
    fn shallow_clone<'a>(&'a self) -> Box<dyn TakeIterator + 'a>;
}
pub trait TakeIteratorNulls: Iterator<Item = Option<usize>> {
    fn shallow_clone<'a>(&'a self) -> Box<dyn TakeIteratorNulls + 'a>;
}

// Implement for the ref as well
impl TakeIterator for &mut dyn TakeIterator {
    fn shallow_clone<'a>(&'a self) -> Box<dyn TakeIterator + 'a> {
        (**self).shallow_clone()
    }
}
impl TakeIteratorNulls for &mut dyn TakeIteratorNulls {
    fn shallow_clone<'a>(&'a self) -> Box<dyn TakeIteratorNulls + 'a> {
        (**self).shallow_clone()
    }
}

// Clonable iterators may implement the traits above
impl<I> TakeIterator for I
where
    I: Iterator<Item = usize> + Clone + Sized,
{
    fn shallow_clone<'a>(&'a self) -> Box<dyn TakeIterator + 'a> {
        Box::new(self.clone())
    }
}
impl<I> TakeIteratorNulls for I
where
    I: Iterator<Item = Option<usize>> + Clone + Sized,
{
    fn shallow_clone<'a>(&'a self) -> Box<dyn TakeIteratorNulls + 'a> {
        Box::new(self.clone())
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
