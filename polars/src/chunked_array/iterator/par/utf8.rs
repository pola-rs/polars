use crate::chunked_array::iterator::{
    Utf8IterManyChunkNullCheckReturnOption, Utf8IterManyChunkReturnOption,
    Utf8IterManyChunkReturnUnwrapped, Utf8IterSingleChunkNullCheckReturnOption,
    Utf8IterSingleChunkReturnOption, Utf8IterSingleChunkReturnUnwrapped,
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
pub struct Utf8ParIterSingleChunkReturnOption<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> Utf8ParIterSingleChunkReturnOption<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
        Utf8ParIterSingleChunkReturnOption { ca }
    }
}

impl<'a> From<Utf8ProducerSingleChunkReturnOption<'a>> for Utf8IterSingleChunkReturnOption<'a> {
    fn from(prod: Utf8ProducerSingleChunkReturnOption<'a>) -> Self {
        let chunks = prod.ca.downcast_chunks();
        let current_array = chunks[0];
        let idx_left = prod.offset;
        let idx_right = prod.offset + prod.len;

        Utf8IterSingleChunkReturnOption {
            current_array,
            idx_left,
            idx_right,
        }
    }
}

impl_utf8_parallel_iterator!(
    Utf8ParIterSingleChunkReturnOption,
    Utf8ProducerSingleChunkReturnOption,
    Utf8IterSingleChunkReturnOption,
    Option<&'a str>
);

/// Parallel Iterator for chunked arrays with just one chunk.
/// It DOES perform null check, then, it is appropriated
/// for chunks whose contents can be null.
///
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct Utf8ParIterSingleChunkNullCheckReturnOption<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> Utf8ParIterSingleChunkNullCheckReturnOption<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
        Utf8ParIterSingleChunkNullCheckReturnOption { ca }
    }
}

impl<'a> From<Utf8ProducerSingleChunkNullCheckReturnOption<'a>>
    for Utf8IterSingleChunkNullCheckReturnOption<'a>
{
    fn from(prod: Utf8ProducerSingleChunkNullCheckReturnOption<'a>) -> Self {
        let chunks = prod.ca.downcast_chunks();
        let current_array = chunks[0];
        let current_data = current_array.data();
        let idx_left = prod.offset;
        let idx_right = prod.offset + prod.len;

        Utf8IterSingleChunkNullCheckReturnOption {
            current_data,
            current_array,
            idx_left,
            idx_right,
        }
    }
}

impl_utf8_parallel_iterator!(
    Utf8ParIterSingleChunkNullCheckReturnOption,
    Utf8ProducerSingleChunkNullCheckReturnOption,
    Utf8IterSingleChunkNullCheckReturnOption,
    Option<&'a str>
);

/// Parallel Iterator for chunked arrays with more than one chunk.
/// It does NOT perform null check, then, it is appropriated
/// for chunks whose contents are never null.
///
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct Utf8ParIterManyChunkReturnOption<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> Utf8ParIterManyChunkReturnOption<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
        Utf8ParIterManyChunkReturnOption { ca }
    }
}

impl<'a> From<Utf8ProducerManyChunkReturnOption<'a>> for Utf8IterManyChunkReturnOption<'a> {
    fn from(prod: Utf8ProducerManyChunkReturnOption<'a>) -> Self {
        let ca = prod.ca;
        let chunks = ca.downcast_chunks();
        let idx_left = prod.offset;
        let (chunk_idx_left, current_array_idx_left) = ca.index_to_chunked_index(idx_left);
        let current_array_left = chunks[chunk_idx_left];
        let idx_right = prod.offset + prod.len;
        let (chunk_idx_right, current_array_idx_right) = ca.right_index_to_chunked_index(idx_right);
        let current_array_right = chunks[chunk_idx_right];
        let current_array_left_len = current_array_left.len();

        Utf8IterManyChunkReturnOption {
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
    Utf8ParIterManyChunkReturnOption,
    Utf8ProducerManyChunkReturnOption,
    Utf8IterManyChunkReturnOption,
    Option<&'a str>
);

/// Parallel Iterator for chunked arrays with more than one chunk.
/// It DOES perform null check, then, it is appropriated
/// for chunks whose contents can be null.
///
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct Utf8ParIterManyChunkNullCheckReturnOption<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> Utf8ParIterManyChunkNullCheckReturnOption<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
        Utf8ParIterManyChunkNullCheckReturnOption { ca }
    }
}

impl<'a> From<Utf8ProducerManyChunkNullCheckReturnOption<'a>>
    for Utf8IterManyChunkNullCheckReturnOption<'a>
{
    fn from(prod: Utf8ProducerManyChunkNullCheckReturnOption<'a>) -> Self {
        let ca = prod.ca;
        let chunks = ca.downcast_chunks();

        // Compute left chunk indexes.
        let idx_left = prod.offset;
        let (chunk_idx_left, current_array_idx_left) = ca.index_to_chunked_index(idx_left);
        let current_array_left = chunks[chunk_idx_left];
        let current_data_left = current_array_left.data();
        let current_array_left_len = current_array_left.len();

        // Compute right chunk indexes.
        let idx_right = prod.offset + prod.len;
        let (chunk_idx_right, current_array_idx_right) = ca.right_index_to_chunked_index(idx_right);
        let current_array_right = chunks[chunk_idx_right];
        let current_data_right = current_array_right.data();

        Utf8IterManyChunkNullCheckReturnOption {
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
    Utf8ParIterManyChunkNullCheckReturnOption,
    Utf8ProducerManyChunkNullCheckReturnOption,
    Utf8IterManyChunkNullCheckReturnOption,
    Option<&'a str>
);

/// Static dispatching structure to allow static polymorphism of chunked
/// parallel iterators.
///
/// All the iterators of the dispatcher returns `Option<&'a str>`.
pub enum Utf8ChunkParIterReturnOptionDispatch<'a> {
    SingleChunk(Utf8ParIterSingleChunkReturnOption<'a>),
    SingleChunkNullCheck(Utf8ParIterSingleChunkNullCheckReturnOption<'a>),
    ManyChunk(Utf8ParIterManyChunkReturnOption<'a>),
    ManyChunkNullCheck(Utf8ParIterManyChunkNullCheckReturnOption<'a>),
}

/// Convert `&'a Utf8Chunked` into a `ParallelIterator` using the most
/// efficient `ParallelIterator` for the given `&'a Utf8Chunked`.
///
/// - If `&'a Utf8Chunked` has only a chunk and has no null values, it uses `Utf8ParIterSingleChunkReturnOption`.
/// - If `&'a Utf8Chunked` has only a chunk and does have null values, it uses `Utf8ParIterSingleChunkNullCheckReturnOption`.
/// - If `&'a Utf8Chunked` has many chunks and has no null values, it uses `Utf8ParIterManyChunkReturnOption`.
/// - If `&'a Utf8Chunked` has many chunks and does have null values, it uses `Utf8ParIterManyChunkNullCheckReturnOption`.
impl<'a> IntoParallelIterator for &'a Utf8Chunked {
    type Iter = Utf8ChunkParIterReturnOptionDispatch<'a>;
    type Item = Option<&'a str>;

    fn into_par_iter(self) -> Self::Iter {
        let chunks = self.downcast_chunks();
        match chunks.len() {
            1 => {
                if self.null_count() == 0 {
                    Utf8ChunkParIterReturnOptionDispatch::SingleChunk(
                        Utf8ParIterSingleChunkReturnOption::new(self),
                    )
                } else {
                    Utf8ChunkParIterReturnOptionDispatch::SingleChunkNullCheck(
                        Utf8ParIterSingleChunkNullCheckReturnOption::new(self),
                    )
                }
            }
            _ => {
                if self.null_count() == 0 {
                    Utf8ChunkParIterReturnOptionDispatch::ManyChunk(
                        Utf8ParIterManyChunkReturnOption::new(self),
                    )
                } else {
                    Utf8ChunkParIterReturnOptionDispatch::ManyChunkNullCheck(
                        Utf8ParIterManyChunkNullCheckReturnOption::new(self),
                    )
                }
            }
        }
    }
}

impl<'a> ParallelIterator for Utf8ChunkParIterReturnOptionDispatch<'a> {
    type Item = Option<&'a str>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        match self {
            Utf8ChunkParIterReturnOptionDispatch::SingleChunk(a) => a.drive_unindexed(consumer),
            Utf8ChunkParIterReturnOptionDispatch::SingleChunkNullCheck(a) => {
                a.drive_unindexed(consumer)
            }
            Utf8ChunkParIterReturnOptionDispatch::ManyChunk(a) => a.drive_unindexed(consumer),
            Utf8ChunkParIterReturnOptionDispatch::ManyChunkNullCheck(a) => {
                a.drive_unindexed(consumer)
            }
        }
    }

    fn opt_len(&self) -> Option<usize> {
        match self {
            Utf8ChunkParIterReturnOptionDispatch::SingleChunk(a) => a.opt_len(),
            Utf8ChunkParIterReturnOptionDispatch::SingleChunkNullCheck(a) => a.opt_len(),
            Utf8ChunkParIterReturnOptionDispatch::ManyChunk(a) => a.opt_len(),
            Utf8ChunkParIterReturnOptionDispatch::ManyChunkNullCheck(a) => a.opt_len(),
        }
    }
}

impl<'a> IndexedParallelIterator for Utf8ChunkParIterReturnOptionDispatch<'a> {
    fn len(&self) -> usize {
        match self {
            Utf8ChunkParIterReturnOptionDispatch::SingleChunk(a) => a.len(),
            Utf8ChunkParIterReturnOptionDispatch::SingleChunkNullCheck(a) => a.len(),
            Utf8ChunkParIterReturnOptionDispatch::ManyChunk(a) => a.len(),
            Utf8ChunkParIterReturnOptionDispatch::ManyChunkNullCheck(a) => a.len(),
        }
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        match self {
            Utf8ChunkParIterReturnOptionDispatch::SingleChunk(a) => a.drive(consumer),
            Utf8ChunkParIterReturnOptionDispatch::SingleChunkNullCheck(a) => a.drive(consumer),
            Utf8ChunkParIterReturnOptionDispatch::ManyChunk(a) => a.drive(consumer),
            Utf8ChunkParIterReturnOptionDispatch::ManyChunkNullCheck(a) => a.drive(consumer),
        }
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        match self {
            Utf8ChunkParIterReturnOptionDispatch::SingleChunk(a) => a.with_producer(callback),
            Utf8ChunkParIterReturnOptionDispatch::SingleChunkNullCheck(a) => {
                a.with_producer(callback)
            }
            Utf8ChunkParIterReturnOptionDispatch::ManyChunk(a) => a.with_producer(callback),
            Utf8ChunkParIterReturnOptionDispatch::ManyChunkNullCheck(a) => {
                a.with_producer(callback)
            }
        }
    }
}

/// Parallel Iterator for chunked arrays with just one chunk.
/// The chunks cannot have null values so it does NOT perform null checks.
///
/// The return type is `&'a str`. So this structure cannot be handled by the `Utf8ChunkParIterReturnOptionDispatch` but
/// by `Utf8ParChunkIterReturnOptionUnwrapped` which is aimed for non-nullable chunked arrays.
#[derive(Debug, Clone)]
pub struct Utf8ParIterSingleChunkReturnUnwrapped<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> Utf8ParIterSingleChunkReturnUnwrapped<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
        Utf8ParIterSingleChunkReturnUnwrapped { ca }
    }
}

impl<'a> From<Utf8ProducerSingleChunkReturnUnwrapped<'a>>
    for Utf8IterSingleChunkReturnUnwrapped<'a>
{
    fn from(prod: Utf8ProducerSingleChunkReturnUnwrapped<'a>) -> Self {
        let chunks = prod.ca.downcast_chunks();
        let current_array = chunks[0];
        let idx_left = prod.offset;
        let idx_right = prod.offset + prod.len;

        Utf8IterSingleChunkReturnUnwrapped {
            current_array,
            idx_left,
            idx_right,
        }
    }
}

impl_utf8_parallel_iterator!(
    Utf8ParIterSingleChunkReturnUnwrapped,
    Utf8ProducerSingleChunkReturnUnwrapped,
    Utf8IterSingleChunkReturnUnwrapped,
    &'a str
);

/// Parallel Iterator for chunked arrays with many chunk.
/// The chunks cannot have null values so it does NOT perform null checks.
///
/// The return type is `&'a str`. So this structure cannot be handled by the `Utf8ChunkParIterReturnOptionDispatch` but
/// by `Utf8ChunkParIterReturnUnwrapppedDispatch` which is aimed for non-nullable chunked arrays.
#[derive(Debug, Clone)]
pub struct Utf8ParIterManyChunkReturnUnwrapped<'a> {
    ca: &'a Utf8Chunked,
}

impl<'a> Utf8ParIterManyChunkReturnUnwrapped<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
        Utf8ParIterManyChunkReturnUnwrapped { ca }
    }
}

impl<'a> From<Utf8ProducerManyChunkReturnUnwrapped<'a>> for Utf8IterManyChunkReturnUnwrapped<'a> {
    fn from(prod: Utf8ProducerManyChunkReturnUnwrapped<'a>) -> Self {
        let ca = prod.ca;
        let chunks = ca.downcast_chunks();
        let idx_left = prod.offset;
        let (chunk_idx_left, current_array_idx_left) = ca.index_to_chunked_index(idx_left);
        let current_array_left = chunks[chunk_idx_left];
        let idx_right = prod.offset + prod.len;
        let (chunk_idx_right, current_array_idx_right) = ca.right_index_to_chunked_index(idx_right);
        let current_array_right = chunks[chunk_idx_right];
        let current_array_left_len = current_array_left.len();

        Utf8IterManyChunkReturnUnwrapped {
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
    Utf8ParIterManyChunkReturnUnwrapped,
    Utf8ProducerManyChunkReturnUnwrapped,
    Utf8IterManyChunkReturnUnwrapped,
    &'a str
);

/// Static dispatching structure to allow static polymorphism of non-nullable chunked parallel iterators.
///
/// All the iterators of the dispatcher returns `&'a str`, as there are no nulls in the chunked array.
pub enum Utf8ChunkParIterReturnUnwrapppedDispatch<'a> {
    SingleChunk(Utf8ParIterSingleChunkReturnUnwrapped<'a>),
    ManyChunk(Utf8ParIterManyChunkReturnUnwrapped<'a>),
}

/// Convert non-nullable `&'a Utf8Chunked` into a non-nullable `ParallelIterator` using the most
/// efficient `ParallelIterator` for the given `&'a Utf8Chunked`.
///
/// - If `&'a Utf8Chunked` has only a chunk, it uses `Utf8ParIterSingleChunkReturnUnwrapped`.
/// - If `&'a Utf8Chunked` has many chunks, it uses `Utf8ParIterManyChunkReturnUnwrapped`.
impl<'a> IntoParallelIterator for NoNull<&'a Utf8Chunked> {
    type Iter = Utf8ChunkParIterReturnUnwrapppedDispatch<'a>;
    type Item = &'a str;

    fn into_par_iter(self) -> Self::Iter {
        let ca = self.0;
        let chunks = ca.downcast_chunks();
        match chunks.len() {
            1 => Utf8ChunkParIterReturnUnwrapppedDispatch::SingleChunk(
                Utf8ParIterSingleChunkReturnUnwrapped::new(ca),
            ),
            _ => Utf8ChunkParIterReturnUnwrapppedDispatch::ManyChunk(
                Utf8ParIterManyChunkReturnUnwrapped::new(ca),
            ),
        }
    }
}

impl<'a> ParallelIterator for Utf8ChunkParIterReturnUnwrapppedDispatch<'a> {
    type Item = &'a str;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        match self {
            Utf8ChunkParIterReturnUnwrapppedDispatch::SingleChunk(a) => a.drive_unindexed(consumer),
            Utf8ChunkParIterReturnUnwrapppedDispatch::ManyChunk(a) => a.drive_unindexed(consumer),
        }
    }

    fn opt_len(&self) -> Option<usize> {
        match self {
            Utf8ChunkParIterReturnUnwrapppedDispatch::SingleChunk(a) => a.opt_len(),
            Utf8ChunkParIterReturnUnwrapppedDispatch::ManyChunk(a) => a.opt_len(),
        }
    }
}

impl<'a> IndexedParallelIterator for Utf8ChunkParIterReturnUnwrapppedDispatch<'a> {
    fn len(&self) -> usize {
        match self {
            Utf8ChunkParIterReturnUnwrapppedDispatch::SingleChunk(a) => a.len(),
            Utf8ChunkParIterReturnUnwrapppedDispatch::ManyChunk(a) => a.len(),
        }
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        match self {
            Utf8ChunkParIterReturnUnwrapppedDispatch::SingleChunk(a) => a.drive(consumer),
            Utf8ChunkParIterReturnUnwrapppedDispatch::ManyChunk(a) => a.drive(consumer),
        }
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        match self {
            Utf8ChunkParIterReturnUnwrapppedDispatch::SingleChunk(a) => a.with_producer(callback),
            Utf8ChunkParIterReturnUnwrapppedDispatch::ManyChunk(a) => a.with_producer(callback),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};

    /// The size of the chunked array used in tests.
    const UTF8_CHUNKED_ARRAY_SIZE: usize = 10_000;

    /// Generates a `Vec` of `Strings`, where every position is the `String` representation of its index.
    fn generate_utf8_vec(size: usize) -> Vec<String> {
        (0..size).map(|n| n.to_string()).collect()
    }

    /// Generate a `Vec` of `Option<String`, where even indexes are `None` and odd indexes are `Some("{idx}")`.
    fn generate_opt_utf8_vec(size: usize) -> Vec<Option<String>> {
        (0..size)
            .map(|n| {
                if n % 2 == 0 {
                    Some(n.to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Implement a test which performs a map over a `ParallelIterator` and over its correspondant `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// Option<&str>.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to implement, it is a function name so it shall be unique.
    /// ca_init_block: The block which initialize the chunked array. It shall return the chunked array.
    macro_rules! impl_par_iter_return_option_map_test {
        ($test_name:ident, $ca_init_block:block) => {
            #[test]
            fn $test_name() {
                let a = $ca_init_block;

                // Perform a parallel maping.
                let par_result = a
                    .into_par_iter()
                    .map(|opt_s| opt_s.map(|s| s.replace("0", "a")))
                    .collect::<Vec<_>>();

                // Perform a sequetial maping.
                let seq_result = a
                    .into_iter()
                    .map(|opt_s| opt_s.map(|s| s.replace("0", "a")))
                    .collect::<Vec<_>>();

                // Check sequetial and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    /// Implement a test which performs a filter over a `ParallelIterator` and over its correspondant `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// Option<&str>.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to implement, it is a function name so it shall be unique.
    /// ca_init_block: The block which initialize the chunked array. It shall return the chunked array.
    macro_rules! impl_par_iter_return_option_filter_test {
        ($test_name:ident, $ca_init_block:block) => {
            #[test]
            fn $test_name() {
                let a = $ca_init_block;

                // Perform a parallel filter.
                let par_result = a
                    .into_par_iter()
                    .filter(|opt_s| {
                        let opt_s = opt_s.map(|s| s.contains("0"));

                        opt_s.unwrap_or(false)
                    })
                    .collect::<Vec<_>>();

                // Perform a sequetial filter.
                let seq_result = a
                    .into_iter()
                    .filter(|opt_s| {
                        let opt_s = opt_s.map(|s| s.contains("0"));

                        opt_s.unwrap_or(false)
                    })
                    .collect::<Vec<_>>();

                // Check sequetial and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    /// Implement a test which performs a fold over a `ParallelIterator` and over its correspondant `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// Option<&str>.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to implement, it is a function name so it shall be unique.
    /// ca_init_block: The block which initialize the chunked array. It shall return the chunked array.
    macro_rules! impl_par_iter_return_option_fold_test {
        ($test_name:ident, $ca_init_block:block) => {
            #[test]
            fn $test_name() {
                let a = $ca_init_block;

                // Perform a parallel sum of length.
                let par_result = a
                    .into_par_iter()
                    .fold(
                        || 0u64,
                        |acc, opt_s| {
                            let opt_s = opt_s.map(|s| s.len() as u64);

                            let len = opt_s.unwrap_or(0);
                            acc + len
                        },
                    )
                    .reduce(|| 0u64, |left, right| left + right);

                // Perform a sequential sum of length.
                let seq_result = a.into_iter().fold(0u64, |acc, opt_s| {
                    let opt_s = opt_s.map(|s| s.len() as u64);

                    let len = opt_s.unwrap_or(0);
                    acc + len
                });

                // Check sequetial and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    // Single Chunk Parallel Iterator Tests.
    impl_par_iter_return_option_map_test!(utf8_par_iter_single_chunk_return_option_map, {
        Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE))
    });

    impl_par_iter_return_option_filter_test!(utf8_par_iter_single_chunk_return_option_filter, {
        Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE))
    });

    impl_par_iter_return_option_fold_test!(utf8_par_iter_single_chunk_return_option_fold, {
        Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE))
    });

    // Single Chunk Null Check Parallel Iterator Tests.
    impl_par_iter_return_option_map_test!(
        utf8_par_iter_single_chunk_null_check_return_option_map,
        { Utf8Chunked::new_from_opt_slice("a", &generate_opt_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE)) }
    );

    impl_par_iter_return_option_filter_test!(
        utf8_par_iter_single_chunk_null_check_return_option_filter,
        { Utf8Chunked::new_from_opt_slice("a", &generate_opt_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE)) }
    );

    impl_par_iter_return_option_fold_test!(
        utf8_par_iter_single_chunk_null_check_return_option_fold,
        { Utf8Chunked::new_from_opt_slice("a", &generate_opt_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE)) }
    );

    // Many Chunk Parallel Iterator Tests.
    impl_par_iter_return_option_map_test!(utf8_par_iter_many_chunk_return_option_map, {
        let mut a = Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE));
        let a_b = Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });

    impl_par_iter_return_option_filter_test!(utf8_par_iter_many_chunk_return_option_filter, {
        let mut a = Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE));
        let a_b = Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });

    impl_par_iter_return_option_fold_test!(utf8_par_iter_many_chunk_return_option_fold, {
        let mut a = Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE));
        let a_b = Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });

    // Many Chunk Null Check Parallel Iterator Tests.
    impl_par_iter_return_option_map_test!(utf8_par_iter_many_chunk_null_check_return_option_map, {
        let mut a =
            Utf8Chunked::new_from_opt_slice("a", &generate_opt_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE));
        let a_b =
            Utf8Chunked::new_from_opt_slice("a", &generate_opt_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });

    impl_par_iter_return_option_filter_test!(
        utf8_par_iter_many_chunk_null_check_return_option_filter,
        {
            let mut a = Utf8Chunked::new_from_opt_slice(
                "a",
                &generate_opt_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE),
            );
            let a_b = Utf8Chunked::new_from_opt_slice(
                "a",
                &generate_opt_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE),
            );
            a.append(&a_b);
            a
        }
    );

    impl_par_iter_return_option_fold_test!(
        utf8_par_iter_many_chunk_null_check_return_option_fold,
        {
            let mut a = Utf8Chunked::new_from_opt_slice(
                "a",
                &generate_opt_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE),
            );
            let a_b = Utf8Chunked::new_from_opt_slice(
                "a",
                &generate_opt_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE),
            );
            a.append(&a_b);
            a
        }
    );

    /// Implement a test which performs a map over a `ParallelIterator` and over its correspondant `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// &str.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to implement, it is a function name so it shall be unique.
    /// ca_init_block: The block which initialize the chunked array. It shall return the chunked array.
    macro_rules! impl_par_iter_return_unwrapped_map_test {
        ($test_name:ident, $ca_init_block:block) => {
            #[test]
            fn $test_name() {
                let a = $ca_init_block;

                // Perform a parallel maping.
                let par_result = NoNull(&a)
                    .into_par_iter()
                    .map(|s| s.replace("0", "a"))
                    .collect::<Vec<_>>();

                // Perform a sequetial maping.
                let seq_result = a
                    .into_no_null_iter()
                    .map(|s| s.replace("0", "a"))
                    .collect::<Vec<_>>();

                // Check sequetial and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    /// Implement a test which performs a filter over a `ParallelIterator` and over its correspondant `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// &str.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to implement, it is a function name so it shall be unique.
    /// ca_init_block: The block which initialize the chunked array. It shall return the chunked array.
    macro_rules! impl_par_iter_return_unwrapped_filter_test {
        ($test_name:ident, $ca_init_block:block) => {
            #[test]
            fn $test_name() {
                let a = $ca_init_block;

                // Perform a parallel filter.
                let par_result = NoNull(&a)
                    .into_par_iter()
                    .filter(|s| s.contains("0"))
                    .collect::<Vec<_>>();

                // Perform a sequetial filter.
                let seq_result = a
                    .into_no_null_iter()
                    .filter(|s| s.contains("0"))
                    .collect::<Vec<_>>();

                // Check sequetial and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    /// Implement a test which performs a fold over a `ParallelIterator` and over its correspondant `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// &str.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to implement, it is a function name so it shall be unique.
    /// ca_init_block: The block which initialize the chunked array. It shall return the chunked array.
    macro_rules! impl_par_iter_return_unwrapped_fold_test {
        ($test_name:ident, $ca_init_block:block) => {
            #[test]
            fn $test_name() {
                let a = $ca_init_block;

                // Perform a parallel sum of length.
                let par_result = NoNull(&a)
                    .into_par_iter()
                    .fold(|| 0u64, |acc, s| acc + s.len() as u64)
                    .reduce(|| 0u64, |left, right| left + right);

                // Perform a sequential sum of length.
                let seq_result = a
                    .into_no_null_iter()
                    .fold(0u64, |acc, s| acc + s.len() as u64);

                // Check sequetial and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    // Single Chunk Return Unwrapped
    impl_par_iter_return_unwrapped_map_test!(utf8_par_iter_single_chunk_return_unwrapped_map, {
        Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE))
    });

    impl_par_iter_return_unwrapped_filter_test!(
        utf8_par_iter_single_chunk_return_unwrapped_filter,
        { Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE)) }
    );

    impl_par_iter_return_unwrapped_fold_test!(utf8_par_iter_single_chunk_return_unwrapped_fold, {
        Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE))
    });

    // Many Chunk Return Unwrapped
    impl_par_iter_return_unwrapped_map_test!(utf8_par_iter_many_chunk_return_unwrapped_map, {
        let mut a = Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE));
        let a_b = Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });

    impl_par_iter_return_unwrapped_filter_test!(
        utf8_par_iter_many_chunk_return_unwrapped_filter,
        {
            let mut a =
                Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE));
            let a_b = Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE));
            a.append(&a_b);
            a
        }
    );

    impl_par_iter_return_unwrapped_fold_test!(utf8_par_iter_many_chunk_return_unwrapped_fold, {
        let mut a = Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE));
        let a_b = Utf8Chunked::new_from_slice("a", &generate_utf8_vec(UTF8_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });
}
