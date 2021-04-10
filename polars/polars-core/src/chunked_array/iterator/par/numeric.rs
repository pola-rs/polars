use crate::chunked_array::iterator::{
    NumIterManyChunk, NumIterManyChunkNullCheck, NumIterSingleChunk, NumIterSingleChunkNullCheck,
    SomeIterator,
};
use crate::prelude::*;
use arrow::array::Array;
use rayon::iter::plumbing::*;
use rayon::iter::plumbing::{Consumer, ProducerCallback};
use rayon::prelude::*;

/// Generate the code for body of a parallel iterator based on the associated sequential iterator.
/// It implements the trait methods.
///
/// # Input
///
/// seq_iter: The sequential iterator to cast the parallel iterator once it is splitted in threads.
macro_rules! impl_numeric_parallel_iterator_body {
    ($seq_iter:ty) => {
        type Item = <$seq_iter as Iterator>::Item;

        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>,
        {
            bridge(self, consumer)
        }

        fn opt_len(&self) -> Option<usize> {
            Some(self.ca.len())
        }
    };
}

/// Generate the code for body of an unindexed parallel iterator. It implements the trait methods.
///
/// # Input
///
/// producer: The producer used to split the iterator into smaller pieces before cast to a sequential iterator.
macro_rules! impl_numeric_indexed_parallel_iterator_body {
    ($producer:ident) => {
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
            callback.callback($producer {
                ca: &self.ca,
                offset: 0,
                len: self.ca.len(),
            })
        }
    };
}

/// Generate the code for body of a producer. It implements the trait methods.
///
/// # Input
///
/// seq_iter: The sequential iterator this producer cast after spliting.
macro_rules! impl_numeric_producer_body {
    ($seq_iter:ty) => {
        type Item = <$seq_iter as Iterator>::Item;
        type IntoIter = $seq_iter;

        fn into_iter(self) -> Self::IntoIter {
            self.into()
        }

        fn split_at(self, index: usize) -> (Self, Self) {
            (
                Self {
                    ca: self.ca,
                    offset: self.offset,
                    len: index,
                },
                Self {
                    ca: self.ca,
                    offset: self.offset + index,
                    len: self.len - index,
                },
            )
        }
    };
}

// Implement methods to generate sequential iterators from raw parts.
// The methods are the same for the `ReturnOption` and `ReturnUnwrap` variant.
impl<'a, T> NumIterSingleChunk<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    fn from_parts(ca: &'a ChunkedArray<T>, offset: usize, len: usize) -> Self {
        let chunk = ca.downcast_iter()[0];
        let slice = &chunk.values()[offset..len];
        let iter = slice.iter().copied();

        Self { iter }
    }
}

impl<'a, T> NumIterManyChunk<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    fn from_parts(ca: &'a ChunkedArray<T>, offset: usize, len: usize) -> Self {
        let chunks = ca.downcast_iter();

        // Compute left array indexes.
        let idx_left = offset;
        let (chunk_idx_left, current_array_idx_left) = ca.index_to_chunked_index(idx_left);
        let current_array_left = chunks[chunk_idx_left];

        // Compute right array indexes.
        let idx_right = offset + len;
        let (chunk_idx_right, current_array_idx_right) = ca.right_index_to_chunked_index(idx_right);

        let (current_array_left_len, current_iter_right) = if chunk_idx_left == chunk_idx_right {
            // If both iterators belong to the same chunk, then, only the left chunk is going to be used
            // and iterate from both sides. This iterator will be the left one and will go from
            // `current_array_idx_left` to `current_array_idx_right`.
            (len, None)
        } else {
            // If the iterators belong to different chunks, then, an iterator for chunk is needed.
            // The left iterator will go from `current_array_idx_left` to the end of the chunk, and
            // the right one will go from the beginning of the chunk to `current_array_idx_right`.
            let current_array_left_len = current_array_left.len() - current_array_idx_left;

            let current_array_right = chunks[chunk_idx_right];
            let current_iter_right = Some(
                current_array_right.values()[..current_array_idx_right]
                    .iter()
                    .copied(),
            );

            (current_array_left_len, current_iter_right)
        };

        let current_iter_left = current_array_left.values()
            [current_array_idx_left..current_array_left_len]
            .iter()
            .copied();

        Self {
            ca,
            current_iter_left,
            chunks,
            current_iter_right,
            idx_left,
            chunk_idx_left,
            idx_right,
            chunk_idx_right,
        }
    }
}

/// Parallel Iterator for chunked arrays with just one chunk.
/// It does NOT perform null check, then, it is appropriated for chunks whose contents are never null.
///
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct NumParIterSingleChunkReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    ca: &'a ChunkedArray<T>,
}

impl<'a, T> NumParIterSingleChunkReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    fn new(ca: &'a ChunkedArray<T>) -> Self {
        Self { ca }
    }
}

impl<'a, T> From<NumProducerSingleChunkReturnOption<'a, T>>
    for SomeIterator<NumIterSingleChunk<'a, T>>
where
    T: PolarsNumericType + Send + Sync,
{
    fn from(prod: NumProducerSingleChunkReturnOption<'a, T>) -> Self {
        SomeIterator(<NumIterSingleChunk<'a, T>>::from_parts(
            prod.ca,
            prod.offset,
            prod.len,
        ))
    }
}

// Implement parallel iterator for NumParIterSingleChunkReturnOption.
impl<'a, T> ParallelIterator for NumParIterSingleChunkReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_parallel_iterator_body!(SomeIterator<NumIterSingleChunk<'a, T>>);
}

struct NumProducerSingleChunkReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    ca: &'a ChunkedArray<T>,
    offset: usize,
    len: usize,
}

impl<'a, T> IndexedParallelIterator for NumParIterSingleChunkReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_indexed_parallel_iterator_body!(NumProducerSingleChunkReturnOption);
}

impl<'a, T> Producer for NumProducerSingleChunkReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_producer_body!(SomeIterator<NumIterSingleChunk<'a, T>>);
}

/// Parallel Iterator for chunked arrays with just one chunk.
/// It DOES perform null check, then, it is appropriated for chunks whose contents can be null.
///
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct NumParIterSingleChunkNullCheckReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    ca: &'a ChunkedArray<T>,
}

impl<'a, T> NumParIterSingleChunkNullCheckReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    fn new(ca: &'a ChunkedArray<T>) -> Self {
        Self { ca }
    }
}

impl<'a, T> From<NumProducerSingleChunkNullCheckReturnOption<'a, T>>
    for NumIterSingleChunkNullCheck<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    fn from(prod: NumProducerSingleChunkNullCheckReturnOption<'a, T>) -> Self {
        let chunks = prod.ca.downcast_iter();
        let arr = chunks[0];
        let idx_left = prod.offset;
        let idx_right = prod.offset + prod.len;

        Self {
            arr,
            idx_left,
            idx_right,
        }
    }
}

// Implement parallel iterator for NumParIterSingleChunkNullCheckReturnOption.
impl<'a, T> ParallelIterator for NumParIterSingleChunkNullCheckReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_parallel_iterator_body!(NumIterSingleChunkNullCheck<'a, T>);
}

struct NumProducerSingleChunkNullCheckReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    ca: &'a ChunkedArray<T>,
    offset: usize,
    len: usize,
}

impl<'a, T> IndexedParallelIterator for NumParIterSingleChunkNullCheckReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_indexed_parallel_iterator_body!(NumProducerSingleChunkNullCheckReturnOption);
}

impl<'a, T> Producer for NumProducerSingleChunkNullCheckReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_producer_body!(NumIterSingleChunkNullCheck<'a, T>);
}

/// Parallel Iterator for chunked arrays with more than one chunk.
/// It does NOT perform null check, then, it is appropriated for chunks whose contents are never null.
///
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct NumParIterManyChunkReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    ca: &'a ChunkedArray<T>,
}

impl<'a, T> NumParIterManyChunkReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    fn new(ca: &'a ChunkedArray<T>) -> Self {
        Self { ca }
    }
}

impl<'a, T> From<NumProducerManyChunkReturnOption<'a, T>> for SomeIterator<NumIterManyChunk<'a, T>>
where
    T: PolarsNumericType + Send + Sync,
{
    fn from(prod: NumProducerManyChunkReturnOption<'a, T>) -> Self {
        SomeIterator(<NumIterManyChunk<'a, T>>::from_parts(
            prod.ca,
            prod.offset,
            prod.len,
        ))
    }
}

// Implement parallel iterator for NumParIterManyChunkReturnOption.
impl<'a, T> ParallelIterator for NumParIterManyChunkReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_parallel_iterator_body!(SomeIterator<NumIterManyChunk<'a, T>>);
}

struct NumProducerManyChunkReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    ca: &'a ChunkedArray<T>,
    offset: usize,
    len: usize,
}

impl<'a, T> IndexedParallelIterator for NumParIterManyChunkReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_indexed_parallel_iterator_body!(NumProducerManyChunkReturnOption);
}

impl<'a, T> Producer for NumProducerManyChunkReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_producer_body!(SomeIterator<NumIterManyChunk<'a, T>>);
}

/// Parallel Iterator for chunked arrays with more than one chunk.
/// It DOES perform null check, then, it is appropriated for chunks whose contents can be null.
///
/// It returns the result wrapped in an `Option`.
#[derive(Debug, Clone)]
pub struct NumParIterManyChunkNullCheckReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    ca: &'a ChunkedArray<T>,
}

impl<'a, T> NumParIterManyChunkNullCheckReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    fn new(ca: &'a ChunkedArray<T>) -> Self {
        Self { ca }
    }
}

impl<'a, T> From<NumProducerManyChunkNullCheckReturnOption<'a, T>>
    for NumIterManyChunkNullCheck<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    fn from(prod: NumProducerManyChunkNullCheckReturnOption<'a, T>) -> Self {
        let ca = prod.ca;
        let chunks = prod.ca.downcast_iter();

        // Compute left array indexes and data.
        let idx_left = prod.offset;
        let (chunk_idx_left, current_array_idx_left) = ca.index_to_chunked_index(idx_left);
        let current_array_left = chunks[chunk_idx_left];
        let current_data_left = current_array_left.data();

        // Compute right array indexes and data.
        let idx_right = prod.offset + prod.len;
        let (chunk_idx_right, current_array_idx_right) = ca.right_index_to_chunked_index(idx_right);
        let current_array_right = chunks[chunk_idx_right];
        let current_data_right = current_array_right.data();

        let (current_array_left_len, current_iter_right) = if chunk_idx_left == chunk_idx_right {
            // If both iterators belong to the same chunk, then, only the left chunk is going to be used
            // and iterate from both sides. This iterator will be the left one and will go from
            // `current_array_idx_left` to `current_array_idx_right`.
            (prod.len, None)
        } else {
            // If the iterators belong to different chunks, then, an iterator for chunk is needed.
            // The left iterator will go from `current_array_idx_left` to the end of the chunk, and
            // the right one will go from the beginning of the chunk to `current_array_idx_right`.
            let current_array_left_len = current_array_left.len() - current_array_idx_left;

            let current_iter_right = Some(
                current_array_right.values()[..current_array_idx_right]
                    .iter()
                    .copied(),
            );

            (current_array_left_len, current_iter_right)
        };

        let current_iter_left = current_array_left.values()
            [current_array_idx_left..current_array_left_len]
            .iter()
            .copied();

        Self {
            ca,
            current_iter_left,
            current_data_left,
            current_array_idx_left,
            chunks,
            current_iter_right,
            current_data_right,
            current_array_idx_right,
            idx_left,
            chunk_idx_left,
            idx_right,
            chunk_idx_right,
        }
    }
}

// Implement parallel iterator for NumParIterManyChunkNullCheckReturnOption.
impl<'a, T> ParallelIterator for NumParIterManyChunkNullCheckReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_parallel_iterator_body!(NumIterManyChunkNullCheck<'a, T>);
}

struct NumProducerManyChunkNullCheckReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    ca: &'a ChunkedArray<T>,
    offset: usize,
    len: usize,
}

impl<'a, T> IndexedParallelIterator for NumParIterManyChunkNullCheckReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_indexed_parallel_iterator_body!(NumProducerManyChunkNullCheckReturnOption);
}

impl<'a, T> Producer for NumProducerManyChunkNullCheckReturnOption<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_producer_body!(NumIterManyChunkNullCheck<'a, T>);
}

/// Parallel Iterator for chunked arrays with just one chunk.
/// The chunks cannot have null values so it does NOT perform null checks.
///
/// The return type is `PolarsNumericType`. So this structure cannot be handled by the `NumParIterDispatcher` but
/// by `NumNoNullParIterDispatcher` which is aimed for non-nullable chunked arrays.
#[derive(Debug, Clone)]
pub struct NumParIterSingleChunkReturnUnwrapped<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    ca: &'a ChunkedArray<T>,
}

impl<'a, T> NumParIterSingleChunkReturnUnwrapped<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    fn new(ca: &'a ChunkedArray<T>) -> Self {
        Self { ca }
    }
}

impl<'a, T> From<NumProducerSingleChunkReturnUnwrapped<'a, T>> for NumIterSingleChunk<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    fn from(prod: NumProducerSingleChunkReturnUnwrapped<'a, T>) -> Self {
        Self::from_parts(prod.ca, prod.offset, prod.len)
    }
}

// Implement parallel iterator for NumParIterSingleChunkReturnUnwrapped.
impl<'a, T> ParallelIterator for NumParIterSingleChunkReturnUnwrapped<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_parallel_iterator_body!(NumIterSingleChunk<'a, T>);
}

struct NumProducerSingleChunkReturnUnwrapped<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    ca: &'a ChunkedArray<T>,
    offset: usize,
    len: usize,
}

impl<'a, T> IndexedParallelIterator for NumParIterSingleChunkReturnUnwrapped<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_indexed_parallel_iterator_body!(NumProducerSingleChunkReturnUnwrapped);
}

impl<'a, T> Producer for NumProducerSingleChunkReturnUnwrapped<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_producer_body!(NumIterSingleChunk<'a, T>);
}

/// Parallel Iterator for chunked arrays with many chunk.
/// The chunks cannot have null values so it does NOT perform null checks.
///
/// The return type is `PolarsNumericType`. So this structure cannot be handled by the `NumParIterDispatcher` but
/// by `NumNoNullParIterDispatcher` which is aimed for non-nullable chunked arrays.
#[derive(Debug, Clone)]
pub struct NumParIterManyChunkReturnUnwrapped<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    ca: &'a ChunkedArray<T>,
}

impl<'a, T> NumParIterManyChunkReturnUnwrapped<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    fn new(ca: &'a ChunkedArray<T>) -> Self {
        Self { ca }
    }
}

impl<'a, T> From<NumProducerManyChunkReturnUnwrapped<'a, T>> for NumIterManyChunk<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    fn from(prod: NumProducerManyChunkReturnUnwrapped<'a, T>) -> Self {
        Self::from_parts(prod.ca, prod.offset, prod.len)
    }
}

// Implement parallel iterator for NumParIterManyChunkReturnUnwrapped.
impl<'a, T> ParallelIterator for NumParIterManyChunkReturnUnwrapped<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_parallel_iterator_body!(NumIterManyChunk<'a, T>);
}

struct NumProducerManyChunkReturnUnwrapped<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    ca: &'a ChunkedArray<T>,
    offset: usize,
    len: usize,
}

impl<'a, T> IndexedParallelIterator for NumParIterManyChunkReturnUnwrapped<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_indexed_parallel_iterator_body!(NumProducerManyChunkReturnUnwrapped);
}

impl<'a, T> Producer for NumProducerManyChunkReturnUnwrapped<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    impl_numeric_producer_body!(NumIterManyChunk<'a, T>);
}

/// Static dispatching structure to allow static polymorphism of chunked parallel iterators.
///
/// All the iterators of the dispatcher returns `Option<PolarsNumericType::Native>`.
pub enum NumParIterDispatcher<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    SingleChunk(NumParIterSingleChunkReturnOption<'a, T>),
    SingleChunkNullCheck(NumParIterSingleChunkNullCheckReturnOption<'a, T>),
    ManyChunk(NumParIterManyChunkReturnOption<'a, T>),
    ManyChunkNullCheck(NumParIterManyChunkNullCheckReturnOption<'a, T>),
}

/// Convert `ChunkedArray` into a `ParallelIterator` using the most efficient
/// `ParallelIterator` implementation for the given `ChunkedArray<T>` of polars numeric types.
///
/// - If `ChunkedArray<T>` has only a chunk and has no null values, it uses `NumParIterSingleChunkReturnOption`.
/// - If `ChunkedArray<T>` has only a chunk and does have null values, it uses `NumParIterSingleChunkNullCheckReturnOption`.
/// - If `ChunkedArray<T>` has many chunks and has no null values, it uses `NumParIterManyChunkReturnOption`.
/// - If `ChunkedArray<T>` has many chunks and does have null values, it uses `NumParIterManyChunkNullCheckReturnOption`.
impl<'a, T> IntoParallelIterator for &'a ChunkedArray<T>
where
    T: PolarsNumericType + Send + Sync,
{
    type Iter = NumParIterDispatcher<'a, T>;
    type Item = Option<T::Native>;

    fn into_par_iter(self) -> Self::Iter {
        let chunks = self.downcast_iter();
        match chunks.len() {
            1 => {
                if self.null_count() == 0 {
                    NumParIterDispatcher::SingleChunk(NumParIterSingleChunkReturnOption::new(self))
                } else {
                    NumParIterDispatcher::SingleChunkNullCheck(
                        NumParIterSingleChunkNullCheckReturnOption::new(self),
                    )
                }
            }
            _ => {
                if self.null_count() == 0 {
                    NumParIterDispatcher::ManyChunk(NumParIterManyChunkReturnOption::new(self))
                } else {
                    NumParIterDispatcher::ManyChunkNullCheck(
                        NumParIterManyChunkNullCheckReturnOption::new(self),
                    )
                }
            }
        }
    }
}

impl<'a, T> ParallelIterator for NumParIterDispatcher<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    type Item = Option<T::Native>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        match self {
            NumParIterDispatcher::SingleChunk(a) => a.drive_unindexed(consumer),
            NumParIterDispatcher::SingleChunkNullCheck(a) => a.drive_unindexed(consumer),
            NumParIterDispatcher::ManyChunk(a) => a.drive_unindexed(consumer),
            NumParIterDispatcher::ManyChunkNullCheck(a) => a.drive_unindexed(consumer),
        }
    }

    fn opt_len(&self) -> Option<usize> {
        match self {
            NumParIterDispatcher::SingleChunk(a) => a.opt_len(),
            NumParIterDispatcher::SingleChunkNullCheck(a) => a.opt_len(),
            NumParIterDispatcher::ManyChunk(a) => a.opt_len(),
            NumParIterDispatcher::ManyChunkNullCheck(a) => a.opt_len(),
        }
    }
}

impl<'a, T> IndexedParallelIterator for NumParIterDispatcher<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    fn len(&self) -> usize {
        match self {
            NumParIterDispatcher::SingleChunk(a) => a.len(),
            NumParIterDispatcher::SingleChunkNullCheck(a) => a.len(),
            NumParIterDispatcher::ManyChunk(a) => a.len(),
            NumParIterDispatcher::ManyChunkNullCheck(a) => a.len(),
        }
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        match self {
            NumParIterDispatcher::SingleChunk(a) => a.drive(consumer),
            NumParIterDispatcher::SingleChunkNullCheck(a) => a.drive(consumer),
            NumParIterDispatcher::ManyChunk(a) => a.drive(consumer),
            NumParIterDispatcher::ManyChunkNullCheck(a) => a.drive(consumer),
        }
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        match self {
            NumParIterDispatcher::SingleChunk(a) => a.with_producer(callback),
            NumParIterDispatcher::SingleChunkNullCheck(a) => a.with_producer(callback),
            NumParIterDispatcher::ManyChunk(a) => a.with_producer(callback),
            NumParIterDispatcher::ManyChunkNullCheck(a) => a.with_producer(callback),
        }
    }
}

/// Static dispatching structure to allow static polymorphism of non-nullable chunked parallel iterators.
///
/// All the iterators of the dispatcher returns `PolarsNumericType::Native`, as there are no nulls in the chunked array.
pub enum NumNoNullParIterDispatcher<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    SingleChunk(NumParIterSingleChunkReturnUnwrapped<'a, T>),
    ManyChunk(NumParIterManyChunkReturnUnwrapped<'a, T>),
}

/// Convert a `ChunkedArray` of numeric types into a non-nullable `ParallelIterator` using the most
/// efficient `ParallelIterator` implementation for the given `ChunkeArray`.
///
/// - If `ChunkeArray<T>` has only a chunk, it uses `NumParIterSingleChunkReturnUnwrapped`.
/// - If `ChunkeArray<T>` has many chunks, it uses `NumParIterManyChunkReturnUnwrapped`.
impl<'a, T> IntoParallelIterator for NoNull<&'a ChunkedArray<T>>
where
    T: PolarsNumericType + Send + Sync,
{
    type Iter = NumNoNullParIterDispatcher<'a, T>;
    type Item = T::Native;

    fn into_par_iter(self) -> Self::Iter {
        let ca = self.0;
        let chunks = ca.downcast_iter();
        match chunks.len() {
            1 => NumNoNullParIterDispatcher::SingleChunk(
                NumParIterSingleChunkReturnUnwrapped::new(ca),
            ),
            _ => NumNoNullParIterDispatcher::ManyChunk(NumParIterManyChunkReturnUnwrapped::new(ca)),
        }
    }
}

impl<'a, T> ParallelIterator for NumNoNullParIterDispatcher<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    type Item = T::Native;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        match self {
            NumNoNullParIterDispatcher::SingleChunk(a) => a.drive_unindexed(consumer),
            NumNoNullParIterDispatcher::ManyChunk(a) => a.drive_unindexed(consumer),
        }
    }

    fn opt_len(&self) -> Option<usize> {
        match self {
            NumNoNullParIterDispatcher::SingleChunk(a) => a.opt_len(),
            NumNoNullParIterDispatcher::ManyChunk(a) => a.opt_len(),
        }
    }
}

impl<'a, T> IndexedParallelIterator for NumNoNullParIterDispatcher<'a, T>
where
    T: PolarsNumericType + Send + Sync,
{
    fn len(&self) -> usize {
        match self {
            NumNoNullParIterDispatcher::SingleChunk(a) => a.len(),
            NumNoNullParIterDispatcher::ManyChunk(a) => a.len(),
        }
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        match self {
            NumNoNullParIterDispatcher::SingleChunk(a) => a.drive(consumer),
            NumNoNullParIterDispatcher::ManyChunk(a) => a.drive(consumer),
        }
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        match self {
            NumNoNullParIterDispatcher::SingleChunk(a) => a.with_producer(callback),
            NumNoNullParIterDispatcher::ManyChunk(a) => a.with_producer(callback),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};

    /// The size of the chunked array used in tests.
    const UINT32_CHUNKED_ARRAY_SIZE: usize = 10;

    /// Generates a `Vec` of `u32`, where every position is the `u32` representation of its index.
    fn generate_uint32_vec(size: usize) -> Vec<u32> {
        (0..size).map(|n| n as u32).collect()
    }

    /// Generate a `Vec` of `Option<u32>`, where even indexes are `Some(idx)` and odd indexes are `None`.
    fn generate_opt_uint32_vec(size: usize) -> Vec<Option<u32>> {
        (0..size)
            .map(|n| {
                let n = n as u32;
                if n % 2 == 0 {
                    Some(n)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Implement a test which performs a map over a `ParallelIterator` and over its correspondent `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// Option<u32>.
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

                // Perform a parallel mapping.
                let par_result = a
                    .into_par_iter()
                    .map(|opt_u| opt_u.map(|u| u + 10))
                    .collect::<Vec<_>>();

                // Perform a sequential mapping.
                let seq_result = a
                    .into_iter()
                    .map(|opt_u| opt_u.map(|u| u + 10))
                    .collect::<Vec<_>>();

                // Check sequential and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    /// Implement a test which performs a filter over a `ParallelIterator` and over its correspondent `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// Option<u32>.
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
                    .filter(|opt_u| {
                        let opt_u = opt_u.map(|u| u % 10 == 0);

                        opt_u.unwrap_or(false)
                    })
                    .collect::<Vec<_>>();

                // Perform a sequential filter.
                let seq_result = a
                    .into_iter()
                    .filter(|opt_u| {
                        let opt_u = opt_u.map(|u| u % 10 == 0);

                        opt_u.unwrap_or(false)
                    })
                    .collect::<Vec<_>>();

                // Check sequential and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    /// Implement a test which performs a fold over a `ParallelIterator` and over its correspondent `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// Option<u32>.
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

                // Perform a parallel sum of values.
                let par_result = a
                    .into_par_iter()
                    .fold(
                        || 0u64,
                        |acc, opt_u| {
                            let val = opt_u.unwrap_or(0) as u64;
                            acc + val
                        },
                    )
                    .reduce(|| 0u64, |left, right| left + right);

                // Perform a sequential sum of values.
                let seq_result = a.into_iter().fold(0u64, |acc, opt_u| {
                    let val = opt_u.unwrap_or(0) as u64;
                    acc + val
                });

                // Check sequential and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    // Single Chunk Parallel Iterator Tests.
    impl_par_iter_return_option_map_test!(uint32_par_iter_single_chunk_return_option_map, {
        UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE))
    });

    impl_par_iter_return_option_filter_test!(uint32_par_iter_single_chunk_return_option_filter, {
        UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE))
    });

    impl_par_iter_return_option_fold_test!(uint32_par_iter_single_chunk_return_option_fold, {
        UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE))
    });

    // Single Chunk Null Check Parallel Iterator Tests.
    impl_par_iter_return_option_map_test!(
        uint32_par_iter_single_chunk_null_check_return_option_map,
        {
            UInt32Chunked::new_from_opt_slice(
                "a",
                &generate_opt_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE),
            )
        }
    );

    impl_par_iter_return_option_filter_test!(
        uint32_par_iter_single_chunk_null_check_return_option_filter,
        {
            UInt32Chunked::new_from_opt_slice(
                "a",
                &generate_opt_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE),
            )
        }
    );

    impl_par_iter_return_option_fold_test!(
        uint32_par_iter_single_chunk_null_check_return_option_fold,
        {
            UInt32Chunked::new_from_opt_slice(
                "a",
                &generate_opt_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE),
            )
        }
    );

    // Many Chunk Parallel Iterator Tests.
    impl_par_iter_return_option_map_test!(uint32_par_iter_many_chunk_return_option_map, {
        let mut a =
            UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE));
        let a_b =
            UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });

    impl_par_iter_return_option_filter_test!(uint32_par_iter_many_chunk_return_option_filter, {
        let mut a =
            UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE));
        let a_b =
            UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });

    impl_par_iter_return_option_fold_test!(uint32_par_iter_many_chunk_return_option_fold, {
        let mut a =
            UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE));
        let a_b =
            UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });

    // Many Chunk Null Check Parallel Iterator Tests.
    impl_par_iter_return_option_map_test!(
        uint32_par_iter_many_chunk_null_check_return_option_map,
        {
            let mut a = UInt32Chunked::new_from_opt_slice(
                "a",
                &generate_opt_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE),
            );
            let a_b = UInt32Chunked::new_from_opt_slice(
                "a",
                &generate_opt_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE),
            );
            a.append(&a_b);
            a
        }
    );

    impl_par_iter_return_option_filter_test!(
        uint32_par_iter_many_chunk_null_check_return_option_filter,
        {
            let mut a = UInt32Chunked::new_from_opt_slice(
                "a",
                &generate_opt_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE),
            );
            let a_b = UInt32Chunked::new_from_opt_slice(
                "a",
                &generate_opt_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE),
            );
            a.append(&a_b);
            a
        }
    );

    impl_par_iter_return_option_fold_test!(
        uint32_par_iter_many_chunk_null_check_return_option_fold,
        {
            let mut a = UInt32Chunked::new_from_opt_slice(
                "a",
                &generate_opt_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE),
            );
            let a_b = UInt32Chunked::new_from_opt_slice(
                "a",
                &generate_opt_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE),
            );
            a.append(&a_b);
            a
        }
    );

    /// Implement a test which performs a map over a `ParallelIterator` and over its correspondent `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// u32.
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

                // Perform a parallel mapping.
                let par_result = NoNull(&a)
                    .into_par_iter()
                    .map(|u| u + 10)
                    .collect::<Vec<_>>();

                // Perform a sequential mapping.
                let seq_result = a.into_no_null_iter().map(|u| u + 10).collect::<Vec<_>>();

                // Check sequential and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    /// Implement a test which performs a filter over a `ParallelIterator` and over its correspondent `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// u32.
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
                    .filter(|u| u % 10 == 0)
                    .collect::<Vec<_>>();

                // Perform a sequential filter.
                let seq_result = a
                    .into_no_null_iter()
                    .filter(|u| u % 10 == 0)
                    .collect::<Vec<_>>();

                // Check sequential and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    /// Implement a test which performs a fold over a `ParallelIterator` and over its correspondent `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// u32.
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
                    .fold(|| 0u64, |acc, u| acc + u as u64)
                    .reduce(|| 0u64, |left, right| left + right);

                // Perform a sequential sum of length.
                let seq_result = a.into_no_null_iter().fold(0u64, |acc, u| acc + u as u64);

                // Check sequential and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    // Single Chunk Return Unwrapped
    impl_par_iter_return_unwrapped_map_test!(uint32_par_iter_single_chunk_return_unwrapped_map, {
        UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE))
    });

    impl_par_iter_return_unwrapped_filter_test!(
        uint32_par_iter_single_chunk_return_unwrapped_filter,
        { UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE)) }
    );

    impl_par_iter_return_unwrapped_fold_test!(
        uint32_par_iter_single_chunk_return_unwrapped_fold,
        { UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE)) }
    );

    // Many Chunk Return Unwrapped
    impl_par_iter_return_unwrapped_map_test!(uint32_par_iter_many_chunk_return_unwrapped_map, {
        let mut a =
            UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE));
        let a_b =
            UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });

    impl_par_iter_return_unwrapped_filter_test!(
        uint32_par_iter_many_chunk_return_unwrapped_filter,
        {
            let mut a =
                UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE));
            let a_b =
                UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE));
            a.append(&a_b);
            a
        }
    );

    impl_par_iter_return_unwrapped_fold_test!(uint32_par_iter_many_chunk_return_unwrapped_fold, {
        let mut a =
            UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE));
        let a_b =
            UInt32Chunked::new_from_slice("a", &generate_uint32_vec(UINT32_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });
}
