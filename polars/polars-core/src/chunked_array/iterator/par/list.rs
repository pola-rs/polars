use crate::chunked_array::iterator::{
    ListIterManyChunk, ListIterManyChunkNullCheck, ListIterSingleChunk,
    ListIterSingleChunkNullCheck, SomeIterator,
};
use crate::prelude::*;
use arrow::array::Array;
use rayon::iter::plumbing::*;
use rayon::iter::plumbing::{Consumer, ProducerCallback};
use rayon::prelude::*;

// Implement the parallel iterators for Seriesº. It also implement the trait `IntoParallelIterator`
// for `&'a ListChunked` and `NoNull<&'a ListChunked>`, which use static dispatcher to use
// the best implementation of parallel iterators depending on the number of chunks and the
// existence of null values.
impl_all_parallel_iterators!(
    // Chunked array.
    &'a ListChunked,

    // Sequential iterators.
    ListIterSingleChunk<'a>,
    ListIterSingleChunkNullCheck<'a>,
    ListIterManyChunk<'a>,
    ListIterManyChunkNullCheck<'a>,

    // Parallel iterators.
    ListParIterSingleChunkReturnOption,
    ListParIterSingleChunkNullCheckReturnOption,
    ListParIterManyChunkReturnOption,
    ListParIterManyChunkNullCheckReturnOption,
    ListParIterSingleChunkReturnUnwrapped,
    ListParIterManyChunkReturnUnwrapped,

    // Producers.
    ListProducerSingleChunkReturnOption,
    ListProducerSingleChunkNullCheckReturnOption,
    ListProducerManyChunkReturnOption,
    ListProducerManyChunkNullCheckReturnOption,
    ListProducerSingleChunkReturnUnwrapped,
    ListProducerManyChunkReturnUnwrapped,

    // Dispatchers.
    ListParIterDispatcher,
    ListNoNullParIterDispatcher,

    // Iter item.
    Series,

    // Lifetime.
    lifetime = 'a
);
