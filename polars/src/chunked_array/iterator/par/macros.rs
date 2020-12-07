//! This module defines the macros used to implement parallel iterators.

/// Generate the code for non-aligned parallel iterators.
///
/// # Input
///
/// ca_type: The chunked array for which these parallel iterator and producer are implemented.
/// par_iter: The name of the structure used as parallel iterator. This structure MUST EXIST as
///   it is not created by this macro. It must consist on a wrapper around a reference to a chunked array.
/// producer: The name used to create the parallel producer. This structure is created in this macro
///   and it is compose of three parts:
///   - ca: a reference to the iterator chunked array.
///   - offset: the index in the chunked array where to start to process.
///   - len: the number of items this producer is in charge of processing.
/// seq_iter: The sequential iterator used to traverse the iterator once the chunked array has been
///   divided in different producer. This structure MUST EXIST as it is not created by this macro.
///   This iterator MUST IMPLEMENT the trait `From<producer>`.
/// iter_item: The iterator `Item`, it represents the iterator return type.
macro_rules! impl_parallel_iterator {
    (
        $ca_type:ty,
        $par_iter:ty ,
        $producer:ident $( < $lifetime:lifetime > )?,
        $seq_iter:ty,
        $iter_item:ty
    ) => {
        impl<$( $lifetime )?> ParallelIterator for $par_iter {
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

        impl<$( $lifetime )?> IndexedParallelIterator for $par_iter {
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
        }

        struct $producer<$( $lifetime )?> {
            ca: $ca_type,
            offset: usize,
            len: usize,
        }

        impl<$( $lifetime )?> Producer for $producer<$( $lifetime )?> {
            type Item = $iter_item;
            type IntoIter = $seq_iter;

            fn into_iter(self) -> Self::IntoIter {
                self.into()
            }

            fn split_at(self, index: usize) -> (Self, Self) {
                (
                    $producer {
                        ca: self.ca,
                        offset: self.offset,
                        len: index,
                    },
                    $producer {
                        ca: self.ca,
                        offset: self.offset + index,
                        len: self.len - index,
                    },
                )
            }
        }
    };
}

/// Implement the code to convert a chunked array into the best implementation of a parallel iterator.
/// The implemented parallel iterator uses a static dispatcher, which returns `Option<iter_type>`, to
/// choose the best implementation of the parallel iterator. The rules to choose the best iterator are:
/// - If the chunked array has just one chunk and no null values, it will use `single_chunk_return_option`.
/// - If the chunked array has just one chunk and null values, it will use `single_chunk_null_check_return_option`.
/// - If the chunked array has many chunks and no null values, it will use `many_chunk_return_option`.
/// - If the chunked array has many chunks and null values, it will use `many_chunk_null_check_return_option`.
///
/// # Input
///
/// ca_type: The chunked array for which the `IntoParallelIterator` implemented.
/// dispatcher: The name of the static dispatcher to be created to implement the `IntoParallelIterator`.
///   It can contain an optional lifetime.
/// single_chunk_return_option: A parallel iterator for chunked arrays with just one chunk and no null values.
/// single_chunk_null_check_return_option: A parallel iterator for chunked arrays with just one chunk and null values.
/// many_chunk_return_option: A parallel iterator for chunked arrays with many chunks and no null values.
/// many_chunk_null_check_return_option: A parallel iterator for chunked arrays with many chunks and null values.
/// iter_item: The iterator `Item`, it represents the iterator return type. It will be wrapped into
///   an `Option<iter_type>`.
macro_rules! impl_into_par_iter {
    (
        $ca_type:ty,
        $dispatcher:ident $( < $lifetime:lifetime > )?,
        $single_chunk_return_option:ty,
        $single_chunk_null_check_return_option:ty,
        $many_chunk_return_option:ty,
        $many_chunk_null_check_return_option:ty,
        $iter_item:ty
    ) => {
        /// Static dispatching structure to allow static polymorphism of chunked parallel iterators.
        ///
        /// All the iterators of the dispatcher returns `Option<$iter_item>`.
        pub enum $dispatcher<$( $lifetime )?> {
            SingleChunk($single_chunk_return_option),
            SingleChunkNullCheck($single_chunk_null_check_return_option),
            ManyChunk($many_chunk_return_option),
            ManyChunkNullCheck($many_chunk_null_check_return_option),
        }

        /// Convert `$ca_iter` into a `ParallelIterator` using the most efficient
        /// `ParallelIterator` implementation for the given `$ca_type`.
        ///
        /// - If `$ca_type` has only a chunk and has no null values, it uses `$single_chunk_return_option`.
        /// - If `$ca_type` has only a chunk and does have null values, it uses `$single_chunk_null_check_return_option`.
        /// - If `$ca_type` has many chunks and has no null values, it uses `$many_chunk_return_option`.
        /// - If `$ca_type` has many chunks and does have null values, it uses `$many_chunk_null_check_return_option`.
        impl<$( $lifetime )?> IntoParallelIterator for $ca_type {
            type Iter = $dispatcher<$( $lifetime )?>;
            type Item = Option<$iter_item>;

            fn into_par_iter(self) -> Self::Iter {
                let chunks = self.downcast_chunks();
                match chunks.len() {
                    1 => {
                        if self.null_count() == 0 {
                            $dispatcher::SingleChunk(
                                <$single_chunk_return_option>::new(self),
                            )
                        } else {
                            $dispatcher::SingleChunkNullCheck(
                                <$single_chunk_null_check_return_option>::new(self),
                            )
                        }
                    }
                    _ => {
                        if self.null_count() == 0 {
                            $dispatcher::ManyChunk(
                                <$many_chunk_return_option>::new(self),
                            )
                        } else {
                            $dispatcher::ManyChunkNullCheck(
                                <$many_chunk_null_check_return_option>::new(self),
                            )
                        }
                    }
                }
            }
        }

        impl<$( $lifetime )?> ParallelIterator for $dispatcher<$( $lifetime )?> {
            type Item = Option<$iter_item>;

            fn drive_unindexed<C>(self, consumer: C) -> C::Result
            where
                C: UnindexedConsumer<Self::Item>,
            {
                match self {
                    $dispatcher::SingleChunk(a) => a.drive_unindexed(consumer),
                    $dispatcher::SingleChunkNullCheck(a) => a.drive_unindexed(consumer),
                    $dispatcher::ManyChunk(a) => a.drive_unindexed(consumer),
                    $dispatcher::ManyChunkNullCheck(a) => a.drive_unindexed(consumer),
                }
            }

            fn opt_len(&self) -> Option<usize> {
                match self {
                    $dispatcher::SingleChunk(a) => a.opt_len(),
                    $dispatcher::SingleChunkNullCheck(a) => a.opt_len(),
                    $dispatcher::ManyChunk(a) => a.opt_len(),
                    $dispatcher::ManyChunkNullCheck(a) => a.opt_len(),
                }
            }
        }

        impl<$( $lifetime )?> IndexedParallelIterator for $dispatcher<$( $lifetime )?> {
            fn len(&self) -> usize {
                match self {
                    $dispatcher::SingleChunk(a) => a.len(),
                    $dispatcher::SingleChunkNullCheck(a) => a.len(),
                    $dispatcher::ManyChunk(a) => a.len(),
                    $dispatcher::ManyChunkNullCheck(a) => a.len(),
                }
            }

            fn drive<C>(self, consumer: C) -> C::Result
            where
                C: Consumer<Self::Item>,
            {
                match self {
                    $dispatcher::SingleChunk(a) => a.drive(consumer),
                    $dispatcher::SingleChunkNullCheck(a) => a.drive(consumer),
                    $dispatcher::ManyChunk(a) => a.drive(consumer),
                    $dispatcher::ManyChunkNullCheck(a) => a.drive(consumer),
                }
            }

            fn with_producer<CB>(self, callback: CB) -> CB::Output
            where
                CB: ProducerCallback<Self::Item>,
            {
                match self {
                    $dispatcher::SingleChunk(a) => a.with_producer(callback),
                    $dispatcher::SingleChunkNullCheck(a) => a.with_producer(callback),
                    $dispatcher::ManyChunk(a) => a.with_producer(callback),
                    $dispatcher::ManyChunkNullCheck(a) => a.with_producer(callback),
                }
            }
        }
    };
}

/// Implement the code to convert a non-nullable chunked array into the best implementation of a
/// parallel iterator which cannot contain null values. The implemented parallel iterator uses a static
/// dispatcher, which returns `iter_type`, to choose the best implementation of the parallel iterator.
/// The rules to choose the best iterator are:
/// - If the chunked array has just one chunk and no null values, it will use `single_chunk_return_unwrapped`.
/// - If the chunked array has many chunks and no null values, it will use `many_chunk_return_unwrapped`.
///
/// # Input
///
/// ca_type: The chunked array for which the `IntoParallelIterator` implemented.
/// dispatcher: The name of the static dispatcher to be created to implement the `IntoParallelIterator`.
///   It can contain an optional lifetime.
/// single_chunk_return_unwrapped: A parallel iterator for chunked arrays with just one chunk and no null values.
/// many_chunk_return_unwrapped: A parallel iterator for chunked arrays with many chunks and no null values.
/// iter_item: The iterator `Item`, it represents the iterator return type.
macro_rules! impl_into_no_null_par_iter {
    (
        $ca_type:ty,
        $dispatcher:ident $( < $lifetime:lifetime > )?,
        $single_chunk_return_unwrapped:ty,
        $many_chunk_return_unwrapped:ty,
        $iter_item:ty
    ) => {
        /// Static dispatching structure to allow static polymorphism of non-nullable chunked parallel iterators.
        ///
        /// All the iterators of the dispatcher returns `$iter_item`, as there are no nulls in the chunked array.
        pub enum $dispatcher<$( $lifetime )?> {
            SingleChunk($single_chunk_return_unwrapped),
            ManyChunk($many_chunk_return_unwrapped),
        }

        /// Convert non-nullable `$ca_type` into a non-nullable `ParallelIterator` using the most
        /// efficient `ParallelIterator` implementation for the given `$ca_type`.
        ///
        /// - If `$ca_type` has only a chunk, it uses `$single_chunk_return_unwrapped`.
        /// - If `$ca_type` has many chunks, it uses `$many_chunk_return_unwrapped`.
        impl<$( $lifetime )?> IntoParallelIterator for NoNull<$ca_type> {
            type Iter = $dispatcher<$( $lifetime )?>;
            type Item = $iter_item;

            fn into_par_iter(self) -> Self::Iter {
                let ca = self.0;
                let chunks = ca.downcast_chunks();
                match chunks.len() {
                    1 => $dispatcher::SingleChunk(
                        <$single_chunk_return_unwrapped>::new(ca),
                    ),
                    _ => $dispatcher::ManyChunk(
                        <$many_chunk_return_unwrapped>::new(ca),
                    ),
                }
            }
        }

        impl<$( $lifetime )?> ParallelIterator for $dispatcher<$( $lifetime )?> {
            type Item = $iter_item;

            fn drive_unindexed<C>(self, consumer: C) -> C::Result
            where
                C: UnindexedConsumer<Self::Item>,
            {
                match self {
                    $dispatcher::SingleChunk(a) => a.drive_unindexed(consumer),
                    $dispatcher::ManyChunk(a) => a.drive_unindexed(consumer),
                }
            }

            fn opt_len(&self) -> Option<usize> {
                match self {
                    $dispatcher::SingleChunk(a) => a.opt_len(),
                    $dispatcher::ManyChunk(a) => a.opt_len(),
                }
            }
        }

        impl<$( $lifetime )?> IndexedParallelIterator for $dispatcher<$( $lifetime )?> {
            fn len(&self) -> usize {
                match self {
                    $dispatcher::SingleChunk(a) => a.len(),
                    $dispatcher::ManyChunk(a) => a.len(),
                }
            }

            fn drive<C>(self, consumer: C) -> C::Result
            where
                C: Consumer<Self::Item>,
            {
                match self {
                    $dispatcher::SingleChunk(a) => a.drive(consumer),
                    $dispatcher::ManyChunk(a) => a.drive(consumer),
                }
            }

            fn with_producer<CB>(self, callback: CB) -> CB::Output
            where
                CB: ProducerCallback<Self::Item>,
            {
                match self {
                    $dispatcher::SingleChunk(a) => a.with_producer(callback),
                    $dispatcher::ManyChunk(a) => a.with_producer(callback),
                }
            }
        }
    };
}
