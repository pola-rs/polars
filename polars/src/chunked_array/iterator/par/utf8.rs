use crate::chunked_array::iterator::{
    Utf8IterManyChunk, Utf8IterManyChunkNullCheck, Utf8IterSingleChunk,
    Utf8IterSingleChunkNullCheck,
};
use crate::prelude::*;
use arrow::array::{Array, StringArray};
use rayon::iter::plumbing::*;
use rayon::iter::plumbing::{Consumer, ProducerCallback};
use rayon::prelude::*;
use std::{mem, ops::Range};

/// Generate the code for Utf8Chunked parallel iterators.
///
/// # Input
///
/// parallel_iterator: The name of the structure used as parallel iterator. This structure
///   MUST EXIST as it is not created by this macro. It must consist on a wrapper around
///   a reference to a chunked array.
///
/// parallel_producer: The name used to create the parallel producer.
///
/// sequential_iterator: The sequential iterator used to traverse the iterator once the
///   chunked array has been divided in different cells. This structure MUST EXIST as it
///   is not created by this macro. This iterator MUST IMPLEMENT the trait From<parallel_producer>.
macro_rules! impl_utf8_parallel_iterator {
    ($parallel_iterator:ident, $parallel_producer:ident, $sequential_iterator:ident) => {
        impl<'a> ParallelIterator for $parallel_iterator<'a> {
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

        impl<'a> IndexedParallelIterator for $parallel_iterator<'a> {
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
                callback.callback($parallel_producer {
                    ca: &self.ca,
                    offset: 0,
                    len: self.ca.len(),
                })
            }
        }

        struct $parallel_producer<'a> {
            ca: &'a Utf8Chunked,
            offset: usize,
            len: usize,
        }

        impl<'a> Producer for $parallel_producer<'a> {
            type Item = Option<&'a str>;
            type IntoIter = $sequential_iterator<'a>;

            fn into_iter(self) -> Self::IntoIter {
                self.into()
            }

            fn split_at(self, index: usize) -> (Self, Self) {
                (
                    $parallel_producer {
                        ca: self.ca,
                        offset: self.offset,
                        len: index,
                    },
                    $parallel_producer {
                        ca: self.ca,
                        offset: self.offset + index,
                        len: self.len - index,
                    },
                )
            }
        }
    };
}

/// Parallel Iterator for chunked arrays with just one chunk.
/// It does NOT perform null check, then, it is appropriated
/// for chunks whose contents are never null.
///
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct Utf8ParIterSingleChunk<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> From<Utf8ProducerSingleChunk<'a>> for Utf8IterSingleChunk<'a> {
    fn from(prod: Utf8ProducerSingleChunk<'a>) -> Self {
        let chunks = prod.ca.downcast_chunks();
        let current_array = chunks[0];
        let idx_left = prod.offset;
        let idx_right = prod.offset + prod.len;

        Utf8IterSingleChunk {
            current_array,
            idx_left,
            idx_right,
        }
    }
}

impl_utf8_parallel_iterator!(
    Utf8ParIterSingleChunk,
    Utf8ProducerSingleChunk,
    Utf8IterSingleChunk
);

/// Parallel Iterator for chunked arrays with just one chunk.
/// It DOES perform null check, then, it is appropriated
/// for chunks whose contents can be null.
///
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct Utf8ParIterSingleChunkNullCheck<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> From<Utf8ProducerSingleChunkNullCheck<'a>> for Utf8IterSingleChunkNullCheck<'a> {
    fn from(prod: Utf8ProducerSingleChunkNullCheck<'a>) -> Self {
        let chunks = prod.ca.downcast_chunks();
        let current_array = chunks[0];
        let current_data = current_array.data();
        let idx_left = prod.offset;
        let idx_right = prod.offset + prod.len;

        Utf8IterSingleChunkNullCheck {
            current_data,
            current_array,
            idx_left,
            idx_right,
        }
    }
}

impl_utf8_parallel_iterator!(
    Utf8ParIterSingleChunkNullCheck,
    Utf8ProducerSingleChunkNullCheck,
    Utf8IterSingleChunkNullCheck
);

/// Parallel Iterator for chunked arrays with more than one chunk.
/// It does NOT perform null check, then, it is appropriated
/// for chunks whose contents are never null.
///
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct Utf8ParIterManyChunk<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> From<Utf8ProducerManyChunk<'a>> for Utf8IterManyChunk<'a> {
    fn from(prod: Utf8ProducerManyChunk<'a>) -> Self {
        let ca = prod.ca;
        let chunks = ca.downcast_chunks();
        let idx_left = prod.offset;
        let (chunk_idx_left, current_array_idx_left) = ca.index_to_chunked_index(idx_left);
        let current_array_left = chunks[chunk_idx_left];
        let idx_right = prod.offset + prod.len;
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
}

impl_utf8_parallel_iterator!(
    Utf8ParIterManyChunk,
    Utf8ProducerManyChunk,
    Utf8IterManyChunk
);

/// Parallel Iterator for chunked arrays with more than one chunk.
/// It DOES perform null check, then, it is appropriated
/// for chunks whose contents can be null.
///
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct Utf8ParIterManyChunkNullCheck<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> From<Utf8ProducerManyChunkNullCheck<'a>> for Utf8IterManyChunkNullCheck<'a> {
    fn from(prod: Utf8ProducerManyChunkNullCheck<'a>) -> Self {
        let ca = prod.ca;
        let chunks = ca.downcast_chunks();
        let idx_left = prod.offset;
        let (chunk_idx_left, current_array_idx_left) = ca.index_to_chunked_index(idx_left);
        let current_array_left = chunks[chunk_idx_left];
        let current_data_left = current_array_left.data();
        let idx_right = prod.offset + prod.len;
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
}

impl_utf8_parallel_iterator!(
    Utf8ParIterManyChunkNullCheck,
    Utf8ProducerManyChunkNullCheck,
    Utf8IterManyChunkNullCheck
);

impl<'a> IntoParallelIterator for &'a Utf8Chunked {
    type Iter = Utf8ParIterSingleChunk<'a>;
    type Item = Option<&'a str>;

    fn into_par_iter(self) -> Self::Iter {
        Utf8ParIterSingleChunk { ca: self }
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
