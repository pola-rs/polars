use crate::prelude::*;
use crate::chunked_array::iterator::{
    Utf8IterSingleChunk,
    Utf8IterSingleChunkNullCheck,
    Utf8IterManyChunk,
    Utf8IterManyChunkNullCheck,
};
use arrow::array::{
    StringArray,
    Array
};
use rayon::iter::plumbing::*;
use rayon::iter::plumbing::{Consumer, ProducerCallback};
use rayon::prelude::*;
use std::{mem, ops::Range};

/// Parallel Iterator for chunked arrays with just one chunk.
/// It does NOT perform null check, then, it is appropriated
/// for chunks whose contents are never null.
/// 
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct Utf8ParIterSingleChunk<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> IntoParallelIterator for &'a Utf8Chunked {
    type Iter = Utf8ParIterSingleChunk<'a>;
    type Item = Option<&'a str>;

    fn into_par_iter(self) -> Self::Iter {
        Utf8ParIterSingleChunk { ca: self }
    }
}

impl<'a> ParallelIterator for Utf8ParIterSingleChunk<'a> {
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

impl<'a> IndexedParallelIterator for Utf8ParIterSingleChunk<'a> {
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
        callback.callback(Utf8ProducerSingleChunk {
            ca: &self.ca,
            offset: 0,
            len: self.ca.len(),
        })
    }
}

struct Utf8ProducerSingleChunk<'a> {
    ca: &'a Utf8Chunked,
    offset: usize,
    len: usize,
}

impl<'a> Producer for Utf8ProducerSingleChunk<'a> {
    type Item = Option<&'a str>;
    type IntoIter = Utf8IterSingleChunk<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let chunks = self.ca.downcast_chunks();
        let current_array = chunks[0];
        let idx_left = self.offset;
        let idx_right = self.offset + self.len;

        Utf8IterSingleChunk {
            current_array,
            idx_left,
            idx_right,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        (
            Utf8ProducerSingleChunk {
                ca: self.ca,
                offset: self.offset,
                len: index,
            },
            Utf8ProducerSingleChunk {
                ca: self.ca,
                offset: self.offset + index,
                len: self.len - index,
            },
        )
    }
}


/// Parallel Iterator for chunked arrays with just one chunk.
/// It DOES perform null check, then, it is appropriated
/// for chunks whose contents can be null.
/// 
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct Utf8ParIterSingleChunkNullCheck<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> ParallelIterator for Utf8ParIterSingleChunkNullCheck<'a> {
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

impl<'a> IndexedParallelIterator for Utf8ParIterSingleChunkNullCheck<'a> {
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
        callback.callback(Utf8ProducerSingleChunkNullCheck {
            ca: &self.ca,
            offset: 0,
            len: self.ca.len(),
        })
    }
}

struct Utf8ProducerSingleChunkNullCheck<'a> {
    ca: &'a Utf8Chunked,
    offset: usize,
    len: usize,
}

impl<'a> Producer for Utf8ProducerSingleChunkNullCheck<'a> {
    type Item = Option<&'a str>;
    type IntoIter = Utf8IterSingleChunkNullCheck<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let chunks = self.ca.downcast_chunks();
        let current_array = chunks[0];
        let current_data = current_array.data();
        let idx_left = self.offset;
        let idx_right = self.offset + self.len;

        Utf8IterSingleChunkNullCheck {
            current_data,
            current_array,
            idx_left,
            idx_right,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        (
            Utf8ProducerSingleChunkNullCheck {
                ca: self.ca,
                offset: self.offset,
                len: index,
            },
            Utf8ProducerSingleChunkNullCheck {
                ca: self.ca,
                offset: self.offset + index,
                len: self.len - index,
            },
        )
    }
}

/// Parallel Iterator for chunked arrays with more than one chunk.
/// It does NOT perform null check, then, it is appropriated
/// for chunks whose contents are never null.
/// 
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct Utf8ParIterManyChunk<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> ParallelIterator for Utf8ParIterManyChunk<'a> {
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

impl<'a> IndexedParallelIterator for Utf8ParIterManyChunk<'a> {
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
        callback.callback(Utf8ProducerManyChunk {
            ca: &self.ca,
            offset: 0,
            len: self.ca.len(),
        })
    }
}

struct Utf8ProducerManyChunk<'a> {
    ca: &'a Utf8Chunked,
    offset: usize,
    len: usize,
}

impl<'a> Producer for Utf8ProducerManyChunk<'a> {
    type Item = Option<&'a str>;
    type IntoIter = Utf8IterManyChunk<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let ca = self.ca;
        let chunks = ca.downcast_chunks();
        let idx_left = self.offset;
        let (chunk_idx_left, current_array_idx_left) = ca.index_to_chunked_index(idx_left);
        let current_array_left = chunks[chunk_idx_left];
        let idx_right = self.offset + self.len;
        let (chunk_idx_right, current_array_idx_right) = ca.index_to_chunked_index(idx_right);
        let current_array_right = chunks[chunk_idx_right];
        let current_array_left_len = current_array_left.len();

        Utf8IterManyChunk {
            ca,
            chunks,
            current_array_left,
            current_array_right,
            current_array_idx_left,
            current_array_idx_right,
            current_array_left_len,
            idx_left,
            idx_right,
            chunk_idx_left,
            chunk_idx_right,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        (
            Utf8ProducerManyChunk {
                ca: self.ca,
                offset: self.offset,
                len: index,
            },
            Utf8ProducerManyChunk {
                ca: self.ca,
                offset: self.offset + index,
                len: self.len - index,
            },
        )
    }
}


/// Parallel Iterator for chunked arrays with more than one chunk.
/// It DOES perform null check, then, it is appropriated
/// for chunks whose contents can be null.
/// 
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct Utf8ParIterManyChunkNullCheck<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> ParallelIterator for Utf8ParIterManyChunkNullCheck<'a> {
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

impl<'a> IndexedParallelIterator for Utf8ParIterManyChunkNullCheck<'a> {
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
        callback.callback(Utf8ProducerManyChunkNullCheck {
            ca: &self.ca,
            offset: 0,
            len: self.ca.len(),
        })
    }
}

struct Utf8ProducerManyChunkNullCheck<'a> {
    ca: &'a Utf8Chunked,
    offset: usize,
    len: usize,
}

impl<'a> Producer for Utf8ProducerManyChunkNullCheck<'a> {
    type Item = Option<&'a str>;
    type IntoIter = Utf8IterManyChunkNullCheck<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let ca = self.ca;
        let chunks = ca.downcast_chunks();
        let idx_left = self.offset;
        let (chunk_idx_left, current_array_idx_left) = ca.index_to_chunked_index(idx_left);
        let current_array_left = chunks[chunk_idx_left];
        let current_data_left = current_array_left.data();
        let idx_right = self.offset + self.len;
        let (chunk_idx_right, current_array_idx_right) = ca.index_to_chunked_index(idx_right);
        let current_array_right = chunks[chunk_idx_right];
        let current_data_right = current_array_right.data();
        let current_array_left_len = current_array_left.len();

        Utf8IterManyChunkNullCheck {
            ca,
            chunks,
            current_data_left,
            current_array_left,
            current_data_right,
            current_array_right,
            current_array_idx_left,
            current_array_idx_right,
            current_array_left_len,
            idx_left,
            idx_right,
            chunk_idx_left,
            chunk_idx_right,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        (
            Utf8ProducerManyChunkNullCheck {
                ca: self.ca,
                offset: self.offset,
                len: index,
            },
            Utf8ProducerManyChunkNullCheck {
                ca: self.ca,
                offset: self.offset + index,
                len: self.len - index,
            },
        )
    }
}

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
