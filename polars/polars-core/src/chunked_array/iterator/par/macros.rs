//! This module defines the macros used to implement parallel iterators.

/// Implement all parallel iterators and producers for a given chunked array. It also implements the
/// `IntoParallelIterator` trait for `ca_type` and for `NoNull<ca_type>`.
/// It will implement and iterator and a producer for the following chunke arrays:
/// - Chunked arrays with one chunk without null value, which returns the value wrapped in an `Option`.
/// - Chunked arrays with one chunk with null values, which returns the value wrapped in an `Option`.
/// - Chunked arrays with many chunks without null value, which returns the value wrapped in an `Option`.
/// - Chunked arrays with many chunks with null values, which returns the value wrapped in an `Option`.
/// - Chunked arrays with one chunk without null value, which returns the value unwrapped.
/// - Chunked arrays with many chunk with null values, which returns the value unwrapped.
///
/// # Input
///
/// ca_type: The chunked array for which implement all the parallel iterators.
///
/// seq_iter_single_chunk: The sequential iterator type for a chunked array with one chunk and without null values.
///   It is a type, so it MUST exist and its lifetime, if any, shall be included.
/// seq_iter_single_chunk_null_check: The sequential iterator type for a chunked array with one chunk and with
///   null values. It is a type, so it MUST exist and its lifetime, if any, shall be included.
/// seq_iter_many_chunk: The sequential iterator type for a chunked array with many chunks and without null values.
///   It is a type, so it MUST exist and its lifetime, if any, shall be included.
/// seq_iter_many_chunk_null_check: The sequential iterator type for a chunked array with many chunks and with
///   null values. It is a type, so it MUST exist and its lifetime, if any, shall be included.
///
/// par_iter_single_chunk_return_option: The parallel iterator for chunked arrays with one chunk and no null values.
///   This parallel iterator return values wrapped in an `Option`. Name must be unique.
/// par_iter_single_chunk_null_check_return_option: The parallel iterator for chunked arrays with one chunk and
///   null values. This parallel iterator return valuew wrapped in an `Option`. Name must be unique.
/// par_iter_many_chunk_return_option: The parallel iterator for chunked arrays with many chunks and no null values.
///   This parallel iterator return values wrapped in an `Option`. Name must be unique.
/// par_iter_many_chunk_null_check_return_option: The parallel iterator for chunked arrays with many chunks and
///   null values. This parallel iterator return valuew wrapped in an `Option`. Name must be unique.
/// par_iter_single_chunk_return_unwrapped: The parallel iterator for chunked arrays with one chunk and no null values.
///   This parallel iterator return values unwrapped. Name must be unique.
/// par_iter_many_chunk_return_unwrapped: The parallel iterator for chunked arrays with many chunks and no null values.
///   This parallel iterator return values unwrapped. Name must be unique.
///
/// producer_single_chunk_return_option: Producer for `par_iter_single_chunk_return_option`. Name must be unique.
/// producer_single_chunk_null_check_return_option: Producer for `par_iter_single_chunk_null_check_return_option`.
///   Name must be unique.
/// producer_many_chunk_return_option: Producer for `par_iter_many_chunk_return_option`. Name must be unique.
/// producer_many_chunk_null_check_return_option: Producer for `par_iter_many_chunk_null_check_return_option`.
///   Name must be unique.
/// producer_single_chunk_return_unwrapped: Producer for `par_iter_single_chunk_return_unwrapped`. Name must be unique.
/// producer_many_chunk_return_unwrapped: Producer for `par_iter_many_chunk_return_unwrapped`. Name must be unique.
///
/// into_par_iter_dispatcher: Static dispatcher for `IntoParallelIterator` trait for `ca_type`. Name must be unique.
/// into_no_null_par_iter_dispatcher: Static dispatcher for `IntoParallelIterator` trait for `NoNull<ca_type>`.
///   Name must be unique.
///
/// iter_item: The type returned by this parallel iterator.
///
/// lifetime: Optional keyword argument. It is the lifetime that will be applied to the new created classes.
///   It shall match the `ca_type` lifetime.
macro_rules! impl_all_parallel_iterators {
    (
        // Chunked array.
        $ca_type:ty,

        // Sequential iterators.
        $seq_iter_single_chunk:ty,
        $seq_iter_single_chunk_null_check:ty,
        $seq_iter_many_chunk:ty,
        $seq_iter_many_chunk_null_check:ty,

        // Parallel iterators.
        $par_iter_single_chunk_return_option:ident,
        $par_iter_single_chunk_null_check_return_option:ident,
        $par_iter_many_chunk_return_option:ident,
        $par_iter_many_chunk_null_check_return_option:ident,
        $par_iter_single_chunk_return_unwrapped:ident,
        $par_iter_many_chunk_return_unwrapped:ident,

        // Producers.
        $producer_single_chunk_return_option:ident,
        $producer_single_chunk_null_check_return_option:ident,
        $producer_many_chunk_return_option:ident,
        $producer_many_chunk_null_check_return_option:ident,
        $producer_single_chunk_return_unwrapped:ident,
        $producer_many_chunk_return_unwrapped:ident,

        // Dispatchers.
        $into_par_iter_dispatcher:ident,
        $into_no_null_par_iter_dispatcher:ident,

        // Item iterator.
        $iter_item:ty

        // Optional lifetime.
        $(, lifetime = $lifetime:lifetime )?
    ) => {
        // Implement methods to generate sequential iterators from raw parts.
        // The methods are the same for the `ReturnOption` and `ReturnUnwrap` variant.
        impl<$( $lifetime )?> $seq_iter_single_chunk {
            fn from_parts(ca: $ca_type, offset: usize, len: usize) -> Self {
                let chunks = ca.downcast_iter();
                let current_array = chunks[0];
                let idx_left = offset;
                let idx_right = offset + len;

                Self {
                    current_array,
                    idx_left,
                    idx_right,
                }
            }
        }

        impl<$( $lifetime )?> $seq_iter_many_chunk {
            fn from_parts(ca: $ca_type, offset: usize, len: usize) -> Self {
                let ca = ca;
                let chunks = ca.downcast_iter();
                let idx_left = offset;
                let (chunk_idx_left, current_array_idx_left) = ca.index_to_chunked_index(idx_left);
                let current_array_left = chunks[chunk_idx_left];
                let idx_right = offset + len;
                let (chunk_idx_right, current_array_idx_right) = ca.right_index_to_chunked_index(idx_right);
                let current_array_right = chunks[chunk_idx_right];
                let current_array_left_len = current_array_left.len();

                Self {
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
        /// It does NOT perform null check, then, it is appropriated for chunks whose contents are never null.
        ///
        /// It returns the result wrapped in an `Option`.
        #[derive(Debug, Clone)]
        pub struct $par_iter_single_chunk_return_option<$( $lifetime )?> {
            ca: $ca_type,
        }

        impl<$( $lifetime )?> $par_iter_single_chunk_return_option<$( $lifetime )?> {
            fn new(ca: $ca_type) -> Self {
                Self { ca }
            }
        }

        impl<$( $lifetime )?> From<$producer_single_chunk_return_option<$( $lifetime )?>>
            for SomeIterator<$seq_iter_single_chunk>
        {
            fn from(prod: $producer_single_chunk_return_option<$( $lifetime )?>) -> Self {
                SomeIterator(<$seq_iter_single_chunk>::from_parts(
                    prod.ca,
                    prod.offset,
                    prod.len,
                ))
            }
        }

        impl_parallel_iterator!(
            $ca_type,
            $par_iter_single_chunk_return_option<$( $lifetime )?>,
            $producer_single_chunk_return_option,
            SomeIterator<$seq_iter_single_chunk>,
            Option<$iter_item>
            $(, lifetime =  $lifetime )?
        );

        /// Parallel Iterator for chunked arrays with just one chunk.
        /// It DOES perform null check, then, it is appropriated for chunks whose contents can be null.
        ///
        /// It returns the result wrapped in an `Option`.
        #[derive(Debug, Clone)]
        pub struct $par_iter_single_chunk_null_check_return_option<$( $lifetime )?> {
            ca: $ca_type,
        }

        impl<$( $lifetime )?> $par_iter_single_chunk_null_check_return_option<$( $lifetime )?> {
            fn new(ca: $ca_type) -> Self {
                Self { ca }
            }
        }

        impl<$( $lifetime )?> From<$producer_single_chunk_null_check_return_option<$( $lifetime )?>>
            for $seq_iter_single_chunk_null_check
        {
            fn from(prod: $producer_single_chunk_null_check_return_option<$( $lifetime )?>) -> Self {
                let chunks = prod.ca.downcast_iter();
                let current_array = chunks[0];
                let current_data = current_array.data();
                let idx_left = prod.offset;
                let idx_right = prod.offset + prod.len;

                Self {
                    current_data,
                    current_array,
                    idx_left,
                    idx_right,
                }
            }
        }

        impl_parallel_iterator!(
            $ca_type,
            $par_iter_single_chunk_null_check_return_option<$( $lifetime )?>,
            $producer_single_chunk_null_check_return_option,
            $seq_iter_single_chunk_null_check,
            Option<$iter_item>
            $(, lifetime =  $lifetime )?
        );

        /// Parallel Iterator for chunked arrays with more than one chunk.
        /// It does NOT perform null check, then, it is appropriated for chunks whose contents are never null.
        ///
        /// It returns the result wrapped in an `Option`.
        #[derive(Debug, Clone)]
        pub struct $par_iter_many_chunk_return_option<$( $lifetime )?> {
            ca: $ca_type,
        }

        impl<$( $lifetime )?> $par_iter_many_chunk_return_option<$( $lifetime )?> {
            fn new(ca: $ca_type) -> Self {
                Self { ca }
            }
        }

        impl<$( $lifetime )?> From<$producer_many_chunk_return_option<$( $lifetime )?>>
            for SomeIterator<$seq_iter_many_chunk>
        {
            fn from(prod: $producer_many_chunk_return_option<$( $lifetime )?>) -> Self {
                SomeIterator(<$seq_iter_many_chunk>::from_parts(
                    prod.ca,
                    prod.offset,
                    prod.len,
                ))
            }
        }

        impl_parallel_iterator!(
            $ca_type,
            $par_iter_many_chunk_return_option<$( $lifetime )?>,
            $producer_many_chunk_return_option,
            SomeIterator<$seq_iter_many_chunk>,
            Option<$iter_item>
            $(, lifetime =  $lifetime )?
        );

        /// Parallel Iterator for chunked arrays with more than one chunk.
        /// It DOES perform null check, then, it is appropriated for chunks whose contents can be null.
        ///
        /// It returns the result wrapped in an `Option`.
        #[derive(Debug, Clone)]
        pub struct $par_iter_many_chunk_null_check_return_option<$( $lifetime )?> {
            ca: $ca_type,
        }

        impl<$( $lifetime )?> $par_iter_many_chunk_null_check_return_option<$( $lifetime )?> {
            fn new(ca: $ca_type) -> Self {
                Self { ca }
            }
        }

        impl<$( $lifetime )?> From<$producer_many_chunk_null_check_return_option<$( $lifetime )?>>
            for $seq_iter_many_chunk_null_check
        {
            fn from(prod: $producer_many_chunk_null_check_return_option<$( $lifetime )?>) -> Self {
                let ca = prod.ca;
                let chunks = ca.downcast_iter();

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

                Self {
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
            $ca_type,
            $par_iter_many_chunk_null_check_return_option<$( $lifetime )?>,
            $producer_many_chunk_null_check_return_option,
            $seq_iter_many_chunk_null_check,
            Option<$iter_item>
            $(, lifetime =  $lifetime )?
        );

        /// Parallel Iterator for chunked arrays with just one chunk.
        /// The chunks cannot have null values so it does NOT perform null checks.
        ///
        /// The return type is `$iter_item`. So this structure cannot be handled by the `$into_par_iter_dispatcher` but
        /// by `$into_no_null_par_iter_dispatcher` which is aimed for non-nullable chunked arrays.
        #[derive(Debug, Clone)]
        pub struct $par_iter_single_chunk_return_unwrapped<$( $lifetime )?> {
            ca: $ca_type,
        }

        impl<$( $lifetime )?> $par_iter_single_chunk_return_unwrapped<$( $lifetime )?> {
            fn new(ca: $ca_type) -> Self {
                Self { ca }
            }
        }

        impl<$( $lifetime )?> From<$producer_single_chunk_return_unwrapped<$( $lifetime )?>>
            for $seq_iter_single_chunk
        {
            fn from(prod: $producer_single_chunk_return_unwrapped<$( $lifetime )?>) -> Self {
                Self::from_parts(prod.ca, prod.offset, prod.len)
            }
        }

        impl_parallel_iterator!(
            $ca_type,
            $par_iter_single_chunk_return_unwrapped<$( $lifetime )?>,
            $producer_single_chunk_return_unwrapped,
            $seq_iter_single_chunk,
            $iter_item
            $(, lifetime =  $lifetime )?
        );

        /// Parallel Iterator for chunked arrays with many chunk.
        /// The chunks cannot have null values so it does NOT perform null checks.
        ///
        /// The return type is `$iter_item`. So this structure cannot be handled by the `$into_par_iter_dispatcher` but
        /// by `$into_no_null_par_iter_dispatcher` which is aimed for non-nullable chunked arrays.
        #[derive(Debug, Clone)]
        pub struct $par_iter_many_chunk_return_unwrapped<$( $lifetime )?> {
            ca: $ca_type,
        }

        impl<$( $lifetime )?> $par_iter_many_chunk_return_unwrapped<$( $lifetime )?> {
            fn new(ca: $ca_type) -> Self {
                Self { ca }
            }
        }

        impl<$( $lifetime )?> From<$producer_many_chunk_return_unwrapped<$( $lifetime )?>>
            for $seq_iter_many_chunk
        {
            fn from(prod: $producer_many_chunk_return_unwrapped<$( $lifetime )?>) -> Self {
                Self::from_parts(prod.ca, prod.offset, prod.len)
            }
        }

        impl_parallel_iterator!(
            $ca_type,
            $par_iter_many_chunk_return_unwrapped<$( $lifetime )?>,
            $producer_many_chunk_return_unwrapped,
            $seq_iter_many_chunk,
            $iter_item
            $(, lifetime =  $lifetime )?
        );

        // Implement into parallel iterator and into no null parallel iterator for $ca_type.
        // In both implementation it creates a static dispatcher which chooses the best implementation
        // of the parallel iterator, depending of the state of the chunked array.
        impl_into_par_iter!(
            $ca_type,
            $into_par_iter_dispatcher,
            $par_iter_single_chunk_return_option<$( $lifetime )?>,
            $par_iter_single_chunk_null_check_return_option<$( $lifetime )?>,
            $par_iter_many_chunk_return_option<$( $lifetime )?>,
            $par_iter_many_chunk_null_check_return_option<$( $lifetime )?>,
            $iter_item
            $(, lifetime =  $lifetime )?
        );

        impl_into_no_null_par_iter!(
            $ca_type,
            $into_no_null_par_iter_dispatcher,
            $par_iter_single_chunk_return_unwrapped<$( $lifetime )?>,
            $par_iter_many_chunk_return_unwrapped<$( $lifetime )?>,
            $iter_item
            $(, lifetime =  $lifetime )?
        );
    };
}

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
/// lifetime: Optional keyword argument. It is the lifetime that will be applied to the new created classes.
///   It shall match the `ca_type` lifetime.
macro_rules! impl_parallel_iterator {
    (
        $ca_type:ty,
        $par_iter:ty ,
        $producer:ident,
        $seq_iter:ty,
        $iter_item:ty
        $(, lifetime = $lifetime:lifetime )?
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
/// single_chunk_return_option: A parallel iterator for chunked arrays with just one chunk and no null values.
/// single_chunk_null_check_return_option: A parallel iterator for chunked arrays with just one chunk and null values.
/// many_chunk_return_option: A parallel iterator for chunked arrays with many chunks and no null values.
/// many_chunk_null_check_return_option: A parallel iterator for chunked arrays with many chunks and null values.
/// iter_item: The iterator `Item`, it represents the iterator return type. It will be wrapped into
///   an `Option<iter_type>`.
/// lifetime: Optional keyword argument. It is the lifetime that will be applied to the new created classes.
///   It shall match the `ca_type` lifetime.
macro_rules! impl_into_par_iter {
    (
        $ca_type:ty,
        $dispatcher:ident,
        $single_chunk_return_option:ty,
        $single_chunk_null_check_return_option:ty,
        $many_chunk_return_option:ty,
        $many_chunk_null_check_return_option:ty,
        $iter_item:ty
        $(, lifetime = $lifetime:lifetime )?
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
                let chunks = self.downcast_iter();
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
/// single_chunk_return_unwrapped: A parallel iterator for chunked arrays with just one chunk and no null values.
/// many_chunk_return_unwrapped: A parallel iterator for chunked arrays with many chunks and no null values.
/// iter_item: The iterator `Item`, it represents the iterator return type.
/// lifetime: Optional keyword argument. It is the lifetime that will be applied to the new created classes.
///   It shall match the `ca_type` lifetime.
macro_rules! impl_into_no_null_par_iter {
    (
        $ca_type:ty,
        $dispatcher:ident,
        $single_chunk_return_unwrapped:ty,
        $many_chunk_return_unwrapped:ty,
        $iter_item:ty
        $(, lifetime = $lifetime:lifetime )?
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
                let chunks = ca.downcast_iter();
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
