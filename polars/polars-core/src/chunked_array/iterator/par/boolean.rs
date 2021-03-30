use crate::chunked_array::iterator::{
    BooleanIterManyChunk, BooleanIterManyChunkNullCheck, BooleanIterSingleChunk,
    BooleanIterSingleChunkNullCheck, SomeIterator,
};
use crate::prelude::*;
use arrow::array::Array;
use rayon::iter::plumbing::*;
use rayon::iter::plumbing::{Consumer, ProducerCallback};
use rayon::prelude::*;

// Implement the parallel iterators for Boolean. It also implement the trait `IntoParallelIterator`
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

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};

    /// The size of the chunked array used in tests.
    const BOOLEAN_CHUNKED_ARRAY_SIZE: usize = 1_000_000;

    /// Generates a `Vec` of `bool`, with even indexes are true, and odd indexes are false.
    fn generate_boolean_vec(size: usize) -> Vec<bool> {
        (0..size).map(|n| n % 2 == 0).collect()
    }

    /// Generate a `Vec` of `Option<bool>`, where:
    /// - If the index is divisible by 3, then, the value is `None`.
    /// - If the index is not divisible by 3 and it is even, then, the value is `Some(true)`.
    /// - Otherwise, the value is `Some(false)`.
    fn generate_opt_boolean_vec(size: usize) -> Vec<Option<bool>> {
        (0..size)
            .map(|n| if n % 3 == 0 { None } else { Some(n % 2 == 0) })
            .collect()
    }

    /// Implement a test which performs a map over a `ParallelIterator` and over its correspondent `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// Option<bool>.
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
                    .map(|opt_b| opt_b.map(|b| !b))
                    .collect::<Vec<_>>();

                // Perform a sequential mapping.
                let seq_result = a
                    .into_iter()
                    .map(|opt_b| opt_b.map(|b| !b))
                    .collect::<Vec<_>>();

                // Check sequential and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    /// Implement a test which performs a filter over a `ParallelIterator` and over its correspondent `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// Option<bool>.
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
                    .filter(|opt_b| opt_b.unwrap_or(false))
                    .collect::<Vec<_>>();

                // Perform a sequential filter.
                let seq_result = a
                    .into_iter()
                    .filter(|opt_b| opt_b.unwrap_or(false))
                    .collect::<Vec<_>>();

                // Check sequential and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    /// Implement a test which performs a fold over a `ParallelIterator` and over its correspondent `Iterator`,
    /// and compares that the result of both iterators is the same. It performs over iterators which return
    /// Option<bool>.
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
                        |acc, opt_b| {
                            let opt_u = opt_b.map(|b| if b { 1 } else { 2 });

                            let val = opt_u.unwrap_or(0);
                            acc + val
                        },
                    )
                    .reduce(|| 0u64, |left, right| left + right);

                // Perform a sequential sum of length.
                let seq_result = a.into_iter().fold(0u64, |acc, opt_b| {
                    let opt_u = opt_b.map(|b| if b { 1 } else { 2 });

                    let val = opt_u.unwrap_or(0);
                    acc + val
                });

                // Check sequential and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    // Single Chunk Parallel Iterator Tests.
    impl_par_iter_return_option_map_test!(boolean_par_iter_single_chunk_return_option_map, {
        BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE))
    });

    impl_par_iter_return_option_filter_test!(boolean_par_iter_single_chunk_return_option_filter, {
        BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE))
    });

    impl_par_iter_return_option_fold_test!(boolean_par_iter_single_chunk_return_option_fold, {
        BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE))
    });

    // Single Chunk Null Check Parallel Iterator Tests.
    impl_par_iter_return_option_map_test!(
        boolean_par_iter_single_chunk_null_check_return_option_map,
        {
            BooleanChunked::new_from_opt_slice(
                "a",
                &generate_opt_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE),
            )
        }
    );

    impl_par_iter_return_option_filter_test!(
        boolean_par_iter_single_chunk_null_check_return_option_filter,
        {
            BooleanChunked::new_from_opt_slice(
                "a",
                &generate_opt_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE),
            )
        }
    );

    impl_par_iter_return_option_fold_test!(
        boolean_par_iter_single_chunk_null_check_return_option_fold,
        {
            BooleanChunked::new_from_opt_slice(
                "a",
                &generate_opt_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE),
            )
        }
    );

    // Many Chunk Parallel Iterator Tests.
    impl_par_iter_return_option_map_test!(boolean_par_iter_many_chunk_return_option_map, {
        let mut a =
            BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE));
        let a_b =
            BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });

    impl_par_iter_return_option_filter_test!(boolean_par_iter_many_chunk_return_option_filter, {
        let mut a =
            BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE));
        let a_b =
            BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });

    impl_par_iter_return_option_fold_test!(boolean_par_iter_many_chunk_return_option_fold, {
        let mut a =
            BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE));
        let a_b =
            BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });

    // Many Chunk Null Check Parallel Iterator Tests.
    impl_par_iter_return_option_map_test!(
        boolean_par_iter_many_chunk_null_check_return_option_map,
        {
            let mut a = BooleanChunked::new_from_opt_slice(
                "a",
                &generate_opt_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE),
            );
            let a_b = BooleanChunked::new_from_opt_slice(
                "a",
                &generate_opt_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE),
            );
            a.append(&a_b);
            a
        }
    );

    impl_par_iter_return_option_filter_test!(
        boolean_par_iter_many_chunk_null_check_return_option_filter,
        {
            let mut a = BooleanChunked::new_from_opt_slice(
                "a",
                &generate_opt_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE),
            );
            let a_b = BooleanChunked::new_from_opt_slice(
                "a",
                &generate_opt_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE),
            );
            a.append(&a_b);
            a
        }
    );

    impl_par_iter_return_option_fold_test!(
        boolean_par_iter_many_chunk_null_check_return_option_fold,
        {
            let mut a = BooleanChunked::new_from_opt_slice(
                "a",
                &generate_opt_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE),
            );
            let a_b = BooleanChunked::new_from_opt_slice(
                "a",
                &generate_opt_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE),
            );
            a.append(&a_b);
            a
        }
    );

    /// Implement a test which performs a map over a `ParallelIterator` and over its correspondent `Iterator`,
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

                // Perform a parallel mapping.
                let par_result = NoNull(&a).into_par_iter().map(|b| !b).collect::<Vec<_>>();

                // Perform a sequential mapping.
                let seq_result = a.into_no_null_iter().map(|b| !b).collect::<Vec<_>>();

                // Check sequential and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    /// Implement a test which performs a filter over a `ParallelIterator` and over its correspondent `Iterator`,
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
                    .filter(|&b| b)
                    .collect::<Vec<_>>();

                // Perform a sequential filter.
                let seq_result = a.into_no_null_iter().filter(|&b| b).collect::<Vec<_>>();

                // Check sequential and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    /// Implement a test which performs a fold over a `ParallelIterator` and over its correspondent `Iterator`,
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
                    .fold(
                        || 0u64,
                        |acc, b| {
                            let val = if b { 1 } else { 2 };
                            acc + val
                        },
                    )
                    .reduce(|| 0u64, |left, right| left + right);

                // Perform a sequential sum of length.
                let seq_result = a.into_no_null_iter().fold(0u64, |acc, b| {
                    let val = if b { 1 } else { 2 };
                    acc + val
                });

                // Check sequential and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    // Single Chunk Return Unwrapped
    impl_par_iter_return_unwrapped_map_test!(boolean_par_iter_single_chunk_return_unwrapped_map, {
        BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE))
    });

    impl_par_iter_return_unwrapped_filter_test!(
        boolean_par_iter_single_chunk_return_unwrapped_filter,
        { BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE)) }
    );

    impl_par_iter_return_unwrapped_fold_test!(
        boolean_par_iter_single_chunk_return_unwrapped_fold,
        { BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE)) }
    );

    // Many Chunk Return Unwrapped
    impl_par_iter_return_unwrapped_map_test!(boolean_par_iter_many_chunk_return_unwrapped_map, {
        let mut a =
            BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE));
        let a_b =
            BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });

    impl_par_iter_return_unwrapped_filter_test!(
        boolean_par_iter_many_chunk_return_unwrapped_filter,
        {
            let mut a = BooleanChunked::new_from_slice(
                "a",
                &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE),
            );
            let a_b = BooleanChunked::new_from_slice(
                "a",
                &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE),
            );
            a.append(&a_b);
            a
        }
    );

    impl_par_iter_return_unwrapped_fold_test!(boolean_par_iter_many_chunk_return_unwrapped_fold, {
        let mut a =
            BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE));
        let a_b =
            BooleanChunked::new_from_slice("a", &generate_boolean_vec(BOOLEAN_CHUNKED_ARRAY_SIZE));
        a.append(&a_b);
        a
    });
}
