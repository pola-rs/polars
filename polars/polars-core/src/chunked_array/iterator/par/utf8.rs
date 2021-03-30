use crate::chunked_array::iterator::{
    SomeIterator, Utf8IterManyChunk, Utf8IterManyChunkNullCheck, Utf8IterSingleChunk,
    Utf8IterSingleChunkNullCheck,
};
use crate::prelude::*;
use arrow::array::Array;
use rayon::iter::plumbing::*;
use rayon::iter::plumbing::{Consumer, ProducerCallback};
use rayon::prelude::*;

// Implement the parallel iterators for Utf8. It also implement the trait `IntoParallelIterator`
// for `&'a Utf8Chunked` and `NoNull<&'a Utf8Chunked>`, which use static dispatcher to use
// the best implementation of parallel iterators depending on the number of chunks and the
// existence of null values.
impl_all_parallel_iterators!(
    // Chunked array.
    &'a Utf8Chunked,

    // Sequential iterators.
    Utf8IterSingleChunk<'a>,
    Utf8IterSingleChunkNullCheck<'a>,
    Utf8IterManyChunk<'a>,
    Utf8IterManyChunkNullCheck<'a>,

    // Parallel iterators.
    Utf8ParIterSingleChunkReturnOption,
    Utf8ParIterSingleChunkNullCheckReturnOption,
    Utf8ParIterManyChunkReturnOption,
    Utf8ParIterManyChunkNullCheckReturnOption,
    Utf8ParIterSingleChunkReturnUnwrapped,
    Utf8ParIterManyChunkReturnUnwrapped,

    // Producers.
    Utf8ProducerSingleChunkReturnOption,
    Utf8ProducerSingleChunkNullCheckReturnOption,
    Utf8ProducerManyChunkReturnOption,
    Utf8ProducerManyChunkNullCheckReturnOption,
    Utf8ProducerSingleChunkReturnUnwrapped,
    Utf8ProducerManyChunkReturnUnwrapped,

    // Dispatchers.
    Utf8ParIterDispatcher,
    Utf8NoNullParIterDispatcher,

    // Iter item.
    &'a str,

    // Lifetime.
    lifetime = 'a
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

    /// Generate a `Vec` of `Option<String>`, where even indexes are `Some("{idx}")` and odd indexes are `None`.
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

    /// Implement a test which performs a map over a `ParallelIterator` and over its correspondent `Iterator`,
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

                // Perform a parallel mapping.
                let par_result = a
                    .into_par_iter()
                    .map(|opt_s| opt_s.map(|s| s.replace("0", "a")))
                    .collect::<Vec<_>>();

                // Perform a sequential mapping.
                let seq_result = a
                    .into_iter()
                    .map(|opt_s| opt_s.map(|s| s.replace("0", "a")))
                    .collect::<Vec<_>>();

                // Check sequential and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    /// Implement a test which performs a filter over a `ParallelIterator` and over its correspondent `Iterator`,
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

                // Perform a sequential filter.
                let seq_result = a
                    .into_iter()
                    .filter(|opt_s| {
                        let opt_s = opt_s.map(|s| s.contains("0"));

                        opt_s.unwrap_or(false)
                    })
                    .collect::<Vec<_>>();

                // Check sequential and parallel results are equal.
                assert_eq!(par_result, seq_result);
            }
        };
    }

    /// Implement a test which performs a fold over a `ParallelIterator` and over its correspondent `Iterator`,
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

                // Check sequential and parallel results are equal.
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
                let par_result = NoNull(&a)
                    .into_par_iter()
                    .map(|s| s.replace("0", "a"))
                    .collect::<Vec<_>>();

                // Perform a sequential mapping.
                let seq_result = a
                    .into_no_null_iter()
                    .map(|s| s.replace("0", "a"))
                    .collect::<Vec<_>>();

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
                    .filter(|s| s.contains("0"))
                    .collect::<Vec<_>>();

                // Perform a sequential filter.
                let seq_result = a
                    .into_no_null_iter()
                    .filter(|s| s.contains("0"))
                    .collect::<Vec<_>>();

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
                    .fold(|| 0u64, |acc, s| acc + s.len() as u64)
                    .reduce(|| 0u64, |left, right| left + right);

                // Perform a sequential sum of length.
                let seq_result = a
                    .into_no_null_iter()
                    .fold(0u64, |acc, s| acc + s.len() as u64);

                // Check sequential and parallel results are equal.
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
