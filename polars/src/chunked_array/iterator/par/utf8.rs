use crate::chunked_array::iterator::{
    Utf8IterCont, Utf8IterManyChunk, Utf8IterManyChunkNullCheck, Utf8IterSingleChunk,
    Utf8IterSingleChunkNullCheck,
};
use crate::prelude::*;
use arrow::array::Array;
use rayon::iter::plumbing::*;
use rayon::iter::plumbing::{Consumer, ProducerCallback};
use rayon::prelude::*;

/// Generate the code for Utf8Chunked parallel iterators.
///
/// # Input
///
/// parallel_iterator: The name of the structure used as parallel iterator. This structure
///   MUST EXIST as it is not created by this macro. It must consist on a wrapper around
///   a reference to a chunked array.
///
/// parallel_producer: The name used to create the parallel producer. This structure is
///   created in this macro and is compose of three parts:
///   - ca: a reference to the iterator chunked array.
///   - offset: the index in the chunked array where to start to process.
///   - len: the number of items this producer is in charge of processing.
///
/// sequential_iterator: The sequential iterator used to traverse the iterator once the
///   chunked array has been divided in different cells. This structure MUST EXIST as it
///   is not created by this macro. This iterator MUST IMPLEMENT the trait `From<parallel_producer>`.
///
/// iter_item: The iterator `Item`, it represents the iterator return type.
macro_rules! impl_utf8_parallel_iterator {
    ($parallel_iterator:ident, $parallel_producer:ident, $sequential_iterator:ident, $iter_item:ty) => {
        impl<'a> ParallelIterator for $parallel_iterator<'a> {
            type Item = $iter_item;

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
            type Item = $iter_item;
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

impl<'a> Utf8ParIterSingleChunk<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
        Utf8ParIterSingleChunk { ca }
    }
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
    Utf8IterSingleChunk,
    Option<&'a str>
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

impl<'a> Utf8ParIterSingleChunkNullCheck<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
        Utf8ParIterSingleChunkNullCheck { ca }
    }
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
    Utf8IterSingleChunkNullCheck,
    Option<&'a str>
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

impl<'a> Utf8ParIterManyChunk<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
        Utf8ParIterManyChunk { ca }
    }
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
    Utf8IterManyChunk,
    Option<&'a str>
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

impl<'a> Utf8ParIterManyChunkNullCheck<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
        Utf8ParIterManyChunkNullCheck { ca }
    }
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
    Utf8IterManyChunkNullCheck,
    Option<&'a str>
);

/// Static dispatching structure to allow static polymorphism of chunked
/// parallel iterators.
///
/// All the iterators of the dispatcher returns `Option<&'a str>`.
pub enum Utf8ParChunkIterDispatch<'a> {
    SingleChunk(Utf8ParIterSingleChunk<'a>),
    SingleChunkNullCheck(Utf8ParIterSingleChunkNullCheck<'a>),
    ManyChunk(Utf8ParIterManyChunk<'a>),
    ManyChunkNullCheck(Utf8ParIterManyChunkNullCheck<'a>),
}

/// Convert `&'a Utf8Chunked` into a `ParallelIterator` using the most
/// efficient `ParallelIterator` for the given `&'a Utf8Chunked`.
///
/// - If `&'a Utf8Chunked` has only a chunk and has no null values, it uses `Utf8ParIterSingleChunk`.
/// - If `&'a Utf8Chunked` has only a chunk and does have null values, it uses `Utf8ParIterSingleChunkNullCheck`.
/// - If `&'a Utf8Chunked` has many chunks and has no null values, it uses `Utf8ParIterManyChunk`.
/// - If `&'a Utf8Chunked` has many chunks and does have null values, it uses `Utf8ParIterManyChunkNullCheck`.
impl<'a> IntoParallelIterator for &'a Utf8Chunked {
    type Iter = Utf8ParChunkIterDispatch<'a>;
    type Item = Option<&'a str>;

    fn into_par_iter(self) -> Self::Iter {
        let chunks = self.downcast_chunks();
        match chunks.len() {
            1 => {
                if self.null_count() == 0 {
                    Utf8ParChunkIterDispatch::SingleChunk(Utf8ParIterSingleChunk::new(self))
                } else {
                    Utf8ParChunkIterDispatch::SingleChunkNullCheck(
                        Utf8ParIterSingleChunkNullCheck::new(self),
                    )
                }
            }
            _ => {
                if self.null_count() == 0 {
                    Utf8ParChunkIterDispatch::ManyChunk(Utf8ParIterManyChunk::new(self))
                } else {
                    Utf8ParChunkIterDispatch::ManyChunkNullCheck(
                        Utf8ParIterManyChunkNullCheck::new(self),
                    )
                }
            }
        }
    }
}

impl<'a> ParallelIterator for Utf8ParChunkIterDispatch<'a> {
    type Item = Option<&'a str>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        match self {
            Utf8ParChunkIterDispatch::SingleChunk(a) => a.drive_unindexed(consumer),
            Utf8ParChunkIterDispatch::SingleChunkNullCheck(a) => a.drive_unindexed(consumer),
            Utf8ParChunkIterDispatch::ManyChunk(a) => a.drive_unindexed(consumer),
            Utf8ParChunkIterDispatch::ManyChunkNullCheck(a) => a.drive_unindexed(consumer),
        }
    }

    fn opt_len(&self) -> Option<usize> {
        match self {
            Utf8ParChunkIterDispatch::SingleChunk(a) => a.opt_len(),
            Utf8ParChunkIterDispatch::SingleChunkNullCheck(a) => a.opt_len(),
            Utf8ParChunkIterDispatch::ManyChunk(a) => a.opt_len(),
            Utf8ParChunkIterDispatch::ManyChunkNullCheck(a) => a.opt_len(),
        }
    }
}

impl<'a> IndexedParallelIterator for Utf8ParChunkIterDispatch<'a> {
    fn len(&self) -> usize {
        match self {
            Utf8ParChunkIterDispatch::SingleChunk(a) => a.len(),
            Utf8ParChunkIterDispatch::SingleChunkNullCheck(a) => a.len(),
            Utf8ParChunkIterDispatch::ManyChunk(a) => a.len(),
            Utf8ParChunkIterDispatch::ManyChunkNullCheck(a) => a.len(),
        }
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        match self {
            Utf8ParChunkIterDispatch::SingleChunk(a) => a.drive(consumer),
            Utf8ParChunkIterDispatch::SingleChunkNullCheck(a) => a.drive(consumer),
            Utf8ParChunkIterDispatch::ManyChunk(a) => a.drive(consumer),
            Utf8ParChunkIterDispatch::ManyChunkNullCheck(a) => a.drive(consumer),
        }
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        match self {
            Utf8ParChunkIterDispatch::SingleChunk(a) => a.with_producer(callback),
            Utf8ParChunkIterDispatch::SingleChunkNullCheck(a) => a.with_producer(callback),
            Utf8ParChunkIterDispatch::ManyChunk(a) => a.with_producer(callback),
            Utf8ParChunkIterDispatch::ManyChunkNullCheck(a) => a.with_producer(callback),
        }
    }
}

/// Parallel Iterator for chunked arrays with just one chunk.
/// The chunks cannot have null values so it does NOT perform null
/// checks.
///
/// The return tipe is `&'a str`.
#[derive(Debug, Clone)]
pub struct Utf8ParIterCont<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> From<Utf8ProducerCont<'a>> for Utf8IterCont<'a> {
    fn from(prod: Utf8ProducerCont<'a>) -> Self {
        let chunks = prod.ca.downcast_chunks();
        let current_array = chunks[0];
        let idx_left = prod.offset;
        let idx_right = prod.offset + prod.len;

        Utf8IterCont {
            current_array,
            idx_left,
            idx_right,
        }
    }
}

impl_utf8_parallel_iterator!(Utf8ParIterCont, Utf8ProducerCont, Utf8IterCont, &'a str);

impl<'a> IntoParallelIterator for NoNull<&'a Utf8Chunked> {
    type Iter = Utf8ParIterCont<'a>;
    type Item = &'a str;

    fn into_par_iter(self) -> Self::Iter {
        Utf8ParIterCont { ca: self.0 }
    }
}
