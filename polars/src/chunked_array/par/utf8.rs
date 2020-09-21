use crate::prelude::*;
use arrow::array::StringArray;
use rayon::iter::plumbing::*;
use rayon::iter::plumbing::{Consumer, ProducerCallback};
use rayon::prelude::*;
use std::{mem, ops::Range};

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
            ca: &self.ca,
            offset: 0,
            len: self.ca.len(),
        })
    }
}

struct Utf8Producer<'a> {
    ca: &'a Utf8Chunked,
    offset: usize,
    len: usize,
}

impl<'a> Producer for Utf8Producer<'a> {
    type Item = Option<&'a str>;
    type IntoIter = Utf8Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        // TODO: slice and create normal iterator?
        let iter = (0..self.len).into_iter();
        Utf8Iter { ca: self.ca, iter }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        (
            Utf8Producer {
                ca: self.ca,
                offset: self.offset,
                len: index + 1,
            },
            Utf8Producer {
                ca: self.ca,
                offset: self.offset + index,
                len: self.len - index,
            },
        )
    }
}

struct Utf8Iter<'a> {
    ca: &'a Utf8Chunked,
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
pub struct Utf8IntoIterCont<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> IntoParallelIterator for NoNull<&'a Utf8Chunked> {
    type Iter = Utf8IntoIterCont<'a>;
    type Item = &'a str;

    fn into_par_iter(self) -> Self::Iter {
        Utf8IntoIterCont { ca: self.0 }
    }
}
impl<'a> ParallelIterator for Utf8IntoIterCont<'a> {
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
impl<'a> IndexedParallelIterator for Utf8IntoIterCont<'a> {
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
        callback.callback(Utf8ProducerCont {
            arr: self.ca.downcast_chunks()[0],
            offset: 0,
            len: self.ca.len(),
        })
    }
}

struct Utf8ProducerCont<'a> {
    arr: &'a StringArray,
    offset: usize,
    len: usize,
}

impl<'a> Producer for Utf8ProducerCont<'a> {
    type Item = &'a str;
    type IntoIter = Utf8IterCont<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let iter = (0..self.len).into_iter();
        Utf8IterCont {
            arr: self.arr,
            iter,
            offset: self.offset,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        (
            Utf8ProducerCont {
                arr: self.arr,
                offset: self.offset,
                len: index + 1,
            },
            Utf8ProducerCont {
                arr: self.arr,
                offset: self.offset + index,
                len: self.len - index,
            },
        )
    }
}

struct Utf8IterCont<'a> {
    arr: &'a StringArray,
    iter: Range<usize>,
    offset: usize,
}

impl<'a> Iterator for Utf8IterCont<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|idx| unsafe {
            mem::transmute::<&'_ str, &'a str>(self.arr.value(idx + self.offset))
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a> DoubleEndedIterator for Utf8IterCont<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|idx| unsafe {
            mem::transmute::<&'_ str, &'a str>(self.arr.value(idx + self.offset))
        })
    }
}

impl<'a> ExactSizeIterator for Utf8IterCont<'a> {}
