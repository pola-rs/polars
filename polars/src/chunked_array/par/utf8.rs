use crate::prelude::*;
use rayon::iter::plumbing::*;
use rayon::iter::plumbing::{Consumer, ProducerCallback};
use rayon::prelude::*;
use std::{marker::PhantomData, mem, ops::Range};

#[derive(Debug, Clone)]
pub struct Utf8IntoIter<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> IntoParallelIterator for &'a Utf8Chunked {
    type Iter = Utf8IntoIter<'a>;
    type Item = Option<&'a str>;

    fn into_par_iter(self) -> Self::Iter {
        Utf8IntoIter { ca: self }
    }
}
impl<'a> ParallelIterator for Utf8IntoIter<'a> {
    type Item = Option<&'a str>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.ca.len())
    }
}
impl<'a> IndexedParallelIterator for Utf8IntoIter<'a> {
    fn len(&self) -> usize {
        self.ca.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(Utf8Producer {
            ca: self.ca.clone(),
            phantom: &PhantomData,
        })
    }
}

struct Utf8Producer<'a> {
    ca: Utf8Chunked,
    phantom: &'a PhantomData<()>,
}

impl<'a> Producer for Utf8Producer<'a> {
    type Item = Option<&'a str>;
    type IntoIter = Utf8Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let iter = (0..self.ca.len()).into_iter();
        Utf8Iter {
            ca: self.ca,
            phantom: &PhantomData,
            iter,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let left = self.ca.slice(0, index).unwrap();
        let right = self.ca.slice(index, self.ca.len() - index).unwrap();
        debug_assert!(right.len() + left.len() == self.ca.len());
        (
            Utf8Producer {
                ca: left,
                phantom: &PhantomData,
            },
            Utf8Producer {
                ca: right,
                phantom: &PhantomData,
            },
        )
    }
}

struct Utf8Iter<'a> {
    ca: Utf8Chunked,
    phantom: &'a PhantomData<()>,
    iter: Range<usize>,
}

impl<'a> Iterator for Utf8Iter<'a> {
    type Item = Option<&'a str>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|idx| unsafe {
            mem::transmute::<Option<&'_ str>, Option<&'a str>>(self.ca.get(idx))
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.ca.len();
        (len, Some(len))
    }
}

impl<'a> DoubleEndedIterator for Utf8Iter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|idx| unsafe {
            mem::transmute::<Option<&'_ str>, Option<&'a str>>(self.ca.get(idx))
        })
    }
}

impl<'a> ExactSizeIterator for Utf8Iter<'a> {}

/// No null Iterators

#[derive(Debug, Clone)]
pub struct Utf8IntoIterNoNull<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> IntoParallelIterator for NoNull<&'a Utf8Chunked> {
    type Iter = Utf8IntoIterNoNull<'a>;
    type Item = &'a str;

    fn into_par_iter(self) -> Self::Iter {
        Utf8IntoIterNoNull { ca: self.0 }
    }
}
impl<'a> ParallelIterator for Utf8IntoIterNoNull<'a> {
    type Item = &'a str;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.ca.len())
    }
}
impl<'a> IndexedParallelIterator for Utf8IntoIterNoNull<'a> {
    fn len(&self) -> usize {
        self.ca.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(Utf8ProducerNoNull {
            ca: self.ca.clone(),
            phantom: &PhantomData,
        })
    }
}

struct Utf8ProducerNoNull<'a> {
    ca: Utf8Chunked,
    phantom: &'a PhantomData<()>,
}

impl<'a> Producer for Utf8ProducerNoNull<'a> {
    type Item = &'a str;
    type IntoIter = Utf8IterNoNull<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let iter = (0..self.ca.len()).into_iter();
        Utf8IterNoNull {
            ca: self.ca,
            phantom: &PhantomData,
            iter,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let left = self.ca.slice(0, index).unwrap();
        let right = self.ca.slice(index, self.ca.len() - index).unwrap();
        debug_assert!(right.len() + left.len() == self.ca.len());
        (
            Utf8ProducerNoNull {
                ca: left,
                phantom: &PhantomData,
            },
            Utf8ProducerNoNull {
                ca: right,
                phantom: &PhantomData,
            },
        )
    }
}

struct Utf8IterNoNull<'a> {
    ca: Utf8Chunked,
    phantom: &'a PhantomData<()>,
    iter: Range<usize>,
}

impl<'a> Iterator for Utf8IterNoNull<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|idx| unsafe { mem::transmute::<&'_ str, &'a str>(self.ca.get_unchecked(idx)) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.ca.len();
        (len, Some(len))
    }
}

impl<'a> DoubleEndedIterator for Utf8IterNoNull<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .next_back()
            .map(|idx| unsafe { mem::transmute::<&'_ str, &'a str>(self.ca.get_unchecked(idx)) })
    }
}

impl<'a> ExactSizeIterator for Utf8IterNoNull<'a> {}
