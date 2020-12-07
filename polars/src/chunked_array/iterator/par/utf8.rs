use crate::chunked_array::iterator::{
    SomeIterator, Utf8IterManyChunk, Utf8IterManyChunkNullCheck, Utf8IterSingleChunk,
    Utf8IterSingleChunkNullCheck,
};
use crate::prelude::*;
use arrow::array::Array;
use rayon::iter::plumbing::*;
use rayon::iter::plumbing::{Consumer, ProducerCallback};
use rayon::prelude::*;

// Implement methods to generate sequential iterators from raw parts.
// The methods are the same for the `ReturnOption` and `ReturnUnwrap` variant.
impl<'a> Utf8IterSingleChunk<'a> {
    fn from_parts(ca: &'a Utf8Chunked, offset: usize, len: usize) -> Utf8IterSingleChunk {
        let chunks = ca.downcast_chunks();
        let current_array = chunks[0];
        let idx_left = offset;
        let idx_right = offset + len;

        Utf8IterSingleChunk {
            current_array,
            idx_left,
            idx_right,
        }
    }
}

impl<'a> Utf8IterManyChunk<'a> {
    fn from_parts(ca: &'a Utf8Chunked, offset: usize, len: usize) -> Utf8IterManyChunk {
        let ca = ca;
        let chunks = ca.downcast_chunks();
        let idx_left = offset;
        let (chunk_idx_left, current_array_idx_left) = ca.index_to_chunked_index(idx_left);
        let current_array_left = chunks[chunk_idx_left];
        let idx_right = offset + len;
        let (chunk_idx_right, current_array_idx_right) = ca.right_index_to_chunked_index(idx_right);
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

impl<'a> From<Utf8ProducerSingleChunkReturnOption<'a>> for SomeIterator<Utf8IterSingleChunk<'a>> {
    fn from(prod: Utf8ProducerSingleChunkReturnOption<'a>) -> Self {
        SomeIterator(Utf8IterSingleChunk::from_parts(
            prod.ca,
            prod.offset,
            prod.len,
        ))
    }
}

impl_parallel_iterator!(
    &'a Utf8Chunked,
    Utf8ParIterSingleChunkReturnOption<'a>,
    Utf8ProducerSingleChunkReturnOption<'a>,
    SomeIterator<Utf8IterSingleChunk<'a>>,
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
    for Utf8IterSingleChunkNullCheck<'a>
{
    fn from(prod: Utf8ProducerSingleChunkNullCheckReturnOption<'a>) -> Self {
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

impl_parallel_iterator!(
    &'a Utf8Chunked,
    Utf8ParIterSingleChunkNullCheckReturnOption<'a>,
    Utf8ProducerSingleChunkNullCheckReturnOption<'a>,
    Utf8IterSingleChunkNullCheck<'a>,
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

impl<'a> From<Utf8ProducerManyChunkReturnOption<'a>> for SomeIterator<Utf8IterManyChunk<'a>> {
    fn from(prod: Utf8ProducerManyChunkReturnOption<'a>) -> Self {
        SomeIterator(Utf8IterManyChunk::from_parts(
            prod.ca,
            prod.offset,
            prod.len,
        ))
    }
}

impl_parallel_iterator!(
    &'a Utf8Chunked,
    Utf8ParIterManyChunkReturnOption<'a>,
    Utf8ProducerManyChunkReturnOption<'a>,
    SomeIterator<Utf8IterManyChunk<'a>>,
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

impl<'a> From<Utf8ProducerManyChunkNullCheckReturnOption<'a>> for Utf8IterManyChunkNullCheck<'a> {
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

impl_parallel_iterator!(
    &'a Utf8Chunked,
    Utf8ParIterManyChunkNullCheckReturnOption<'a>,
    Utf8ProducerManyChunkNullCheckReturnOption<'a>,
    Utf8IterManyChunkNullCheck<'a>,
    Option<&'a str>
);

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

impl<'a> From<Utf8ProducerSingleChunkReturnUnwrapped<'a>> for Utf8IterSingleChunk<'a> {
    fn from(prod: Utf8ProducerSingleChunkReturnUnwrapped<'a>) -> Self {
        Utf8IterSingleChunk::from_parts(prod.ca, prod.offset, prod.len)
    }
}

impl_parallel_iterator!(
    &'a Utf8Chunked,
    Utf8ParIterSingleChunkReturnUnwrapped<'a>,
    Utf8ProducerSingleChunkReturnUnwrapped<'a>,
    Utf8IterSingleChunk<'a>,
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

impl<'a> From<Utf8ProducerManyChunkReturnUnwrapped<'a>> for Utf8IterManyChunk<'a> {
    fn from(prod: Utf8ProducerManyChunkReturnUnwrapped<'a>) -> Self {
        Utf8IterManyChunk::from_parts(prod.ca, prod.offset, prod.len)
    }
}

impl_parallel_iterator!(
    &'a Utf8Chunked,
    Utf8ParIterManyChunkReturnUnwrapped<'a>,
    Utf8ProducerManyChunkReturnUnwrapped<'a>,
    Utf8IterManyChunk<'a>,
    &'a str
);

// Implement into parallel iterator and into no null parallel iterator for &'a Utf8Chunked.
// In both implementation it creates a static dispatcher which chooses the best implementation
// of the parallel iterator, depending of the state of the chunked array.
impl_into_par_iter!(
    &'a Utf8Chunked,
    Utf8ParIterDispatcher<'a>,
    Utf8ParIterSingleChunkReturnOption<'a>,
    Utf8ParIterSingleChunkNullCheckReturnOption<'a>,
    Utf8ParIterManyChunkReturnOption<'a>,
    Utf8ParIterManyChunkNullCheckReturnOption<'a>,
    &'a str
);

impl_into_no_null_par_iter!(
    &'a Utf8Chunked,
    Utf8NoNullParIterDispatcher<'a>,
    Utf8ParIterSingleChunkReturnUnwrapped<'a>,
    Utf8ParIterManyChunkReturnUnwrapped<'a>,
    &'a str
);

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
