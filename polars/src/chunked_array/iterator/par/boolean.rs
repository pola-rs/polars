use crate::chunked_array::iterator::{
    SomeIterator, BooleanIterManyChunk, BooleanIterManyChunkNullCheck, BooleanIterSingleChunk,
    BooleanIterSingleChunkNullCheck,
};
use crate::prelude::*;
use arrow::array::Array;
use rayon::iter::plumbing::*;
use rayon::iter::plumbing::{Consumer, ProducerCallback};
use rayon::prelude::*;

// Implement the parallel iterators for Utf8. It also implement the trait `IntoParallelIterator`
// for `&'a BooleanChunked` and `NoNull<&'a BooleanChunked>`, which use static dispatcher to use
// the best implementation of parallel iterators depending on the number of chunks and the
// existence of null values.
impl_all_parallel_iterators!(
    // Chunked array.
    &'a BooleanChunked,

    // Sequential iterators.
    BooleanIterSingleChunk<'a>,
    BooleanIterSingleChunkNullCheck<'a>,
    BooleanIterManyChunk<'a>,
    BooleanIterManyChunkNullCheck<'a>,

    // Parallel iterators.
    BooleanParIterSingleChunkReturnOption,
    BooleanParIterSingleChunkNullCheckReturnOption,
    BooleanParIterManyChunkReturnOption,
    BooleanParIterManyChunkNullCheckReturnOption,
    BooleanParIterSingleChunkReturnUnwrapped,
    BooleanParIterManyChunkReturnUnwrapped,

    // Producers.
    BooleanProducerSingleChunkReturnOption,
    BooleanProducerSingleChunkNullCheckReturnOption,
    BooleanProducerManyChunkReturnOption,
    BooleanProducerManyChunkNullCheckReturnOption,
    BooleanProducerSingleChunkReturnUnwrapped,
    BooleanProducerManyChunkReturnUnwrapped,

    // Dispatchers.
    BooleanParIterDispatcher,
    BooleanNoNullParIterDispatcher,

    // Iter item.
    bool,

    // Lifetime.
    lifetime = 'a
);
