use crate::prelude::*;
use arrow::array::{
    Array, ArrayDataRef, ArrayRef, BooleanArray, ListArray, PrimitiveArray, PrimitiveArrayOps,
    StringArray,
};
use std::iter::Copied;
use std::slice::Iter;

// If parallel feature is enable, then, activate the parallel module.
#[cfg(feature = "parallel")]
#[doc(cfg(feature = "parallel"))]
pub mod par;

// ExactSizeIterator trait implementations for all Iterator structs in this file
impl<'a, T> ExactSizeIterator for NumIterSingleChunk<'a, T> where T: PolarsNumericType {}
impl<'a, T> ExactSizeIterator for NumIterSingleChunkNullCheck<'a, T> where T: PolarsNumericType {}
impl<'a, T> ExactSizeIterator for NumIterManyChunkNullCheck<'a, T> where T: PolarsNumericType {}
impl<'a, T> ExactSizeIterator for NumIterManyChunk<'a, T> where T: PolarsNumericType {}
impl<'a, T> PolarsIterator for NumIterSingleChunk<'a, T> where T: PolarsNumericType {}
impl<'a, T> PolarsIterator for NumIterSingleChunkNullCheck<'a, T> where T: PolarsNumericType {}
impl<'a, T> PolarsIterator for NumIterManyChunkNullCheck<'a, T> where T: PolarsNumericType {}
impl<'a, T> PolarsIterator for NumIterManyChunk<'a, T> where T: PolarsNumericType {}

/// A `PolarsIterator` is an iterator over a `ChunkedArray` which contains polars types. A `PolarsIterator`
/// must implement `ExactSizeIterator` and `DoubleEndedIterator`.
pub trait PolarsIterator: ExactSizeIterator + DoubleEndedIterator {}

/// Trait for ChunkedArrays that don't have null values.
/// The result is the most efficient implementation `Iterator`, according to the number of chunks.
pub trait IntoNoNullIterator {
    type Item;
    type IntoIter: Iterator<Item = Self::Item>;

    fn into_no_null_iter(self) -> Self::IntoIter;
}

/// Wrapper strunct to convert an iterator of type `T` into one of type `Option<T>`.  It is useful to make the
/// `IntoIterator` trait, in which every iterator shall return an `Option<T>`.
struct SomeIterator<I>(I)
where
    I: Iterator;

impl<I> Iterator for SomeIterator<I>
where
    I: Iterator,
{
    type Item = Option<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(Some)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<I> DoubleEndedIterator for SomeIterator<I>
where
    I: DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(Some)
    }
}

impl<I> ExactSizeIterator for SomeIterator<I> where I: ExactSizeIterator {}
impl<I> PolarsIterator for SomeIterator<I> where I: PolarsIterator {}

/// Iterator for chunked arrays with just one chunk.
/// The chunk cannot have null values so it does NOT perform null checks.
///
/// The return type is `PolarsNumericType::Native`.
pub struct NumIterSingleChunk<'a, T>
where
    T: PolarsNumericType,
{
    iter: Copied<Iter<'a, T::Native>>,
}

impl<'a, T> NumIterSingleChunk<'a, T>
where
    T: PolarsNumericType,
{
    fn new(ca: &'a ChunkedArray<T>) -> Self {
        let chunk = ca.downcast_chunks()[0];
        let slice = chunk.value_slice(0, chunk.len());
        let iter = slice.iter().copied();

        NumIterSingleChunk { iter }
    }
}

impl<'a, T> Iterator for NumIterSingleChunk<'a, T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for NumIterSingleChunk<'a, T>
where
    T: PolarsNumericType,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

/// Iterator for chunked arrays with just one chunk.
/// The chunk have null values so it DOES perform null checks.
///
/// The return type is `Option<PolarsNumericType::Native>`.
pub struct NumIterSingleChunkNullCheck<'a, T>
where
    T: PolarsNumericType,
{
    arr: &'a PrimitiveArray<T>,
    idx: usize,
    back_idx: usize,
}

impl<'a, T> NumIterSingleChunkNullCheck<'a, T>
where
    T: PolarsNumericType,
{
    fn new(ca: &'a ChunkedArray<T>) -> Self {
        let chunks = ca.downcast_chunks();
        let arr = chunks[0];
        let idx = 0;
        let back_idx = arr.len();

        NumIterSingleChunkNullCheck { arr, idx, back_idx }
    }

    fn return_opt_val(&self, index: usize) -> Option<T::Native> {
        if self.arr.is_null(index) {
            None
        } else {
            Some(self.arr.value(index))
        }
    }
}

impl<'a, T> Iterator for NumIterSingleChunkNullCheck<'a, T>
where
    T: PolarsNumericType,
{
    type Item = Option<T::Native>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.back_idx {
            None
        } else {
            let ret = self.return_opt_val(self.idx);
            self.idx += 1;

            Some(ret)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.arr.len();
        (len, Some(len))
    }
}

impl<'a, T> DoubleEndedIterator for NumIterSingleChunkNullCheck<'a, T>
where
    T: PolarsNumericType,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.idx == self.back_idx {
            None
        } else {
            self.back_idx -= 1;
            Some(self.return_opt_val(self.back_idx))
        }
    }
}

/// Iterator for chunked arrays with many chunks.
/// The chunks cannot have null values so it does NOT perform null checks.
///
/// The return type is `PolarsNumericType::Native`.
pub struct NumIterManyChunk<'a, T>
where
    T: PolarsNumericType,
{
    ca: &'a ChunkedArray<T>,
    chunks: Vec<&'a PrimitiveArray<T>>,
    // current iterator if we iterate from the left
    current_iter_left: Copied<Iter<'a, T::Native>>,
    // If iter_left and iter_right are the same, this is None and we need to use iter left
    // This is done because we can only have one owner
    current_iter_right: Option<Copied<Iter<'a, T::Native>>>,
    chunk_idx_left: usize,
    idx_left: usize,
    idx_right: usize,
    chunk_idx_right: usize,
}

impl<'a, T> NumIterManyChunk<'a, T>
where
    T: PolarsNumericType,
{
    fn set_current_iter_left(&mut self) {
        let current_chunk = unsafe { self.chunks.get_unchecked(self.chunk_idx_left) };
        self.current_iter_left = current_chunk
            .value_slice(0, current_chunk.len())
            .iter()
            .copied();
    }

    fn set_current_iter_right(&mut self) {
        if self.chunk_idx_left == self.chunk_idx_right {
            // from left and right we use the same iterator
            self.current_iter_right = None
        } else {
            let current_chunk = unsafe { self.chunks.get_unchecked(self.chunk_idx_right) };
            self.current_iter_right = Some(
                current_chunk
                    .value_slice(0, current_chunk.len())
                    .iter()
                    .copied(),
            );
        }
    }
    fn new(ca: &'a ChunkedArray<T>) -> Self {
        let chunks = ca.downcast_chunks();
        let current_len_left = chunks[0].len();
        let current_iter_left = chunks[0].value_slice(0, current_len_left).iter().copied();

        let idx_left = 0;
        let chunk_idx_left = 0;
        let idx_right = ca.len();

        let chunk_idx_right = chunks.len() - 1;
        let current_iter_right;
        if chunk_idx_left == chunk_idx_right {
            current_iter_right = None
        } else {
            let arr = chunks[chunk_idx_right];
            current_iter_right = Some(arr.value_slice(0, arr.len()).iter().copied())
        }

        NumIterManyChunk {
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

impl<'a, T> Iterator for NumIterManyChunk<'a, T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;

    fn next(&mut self) -> Option<Self::Item> {
        let opt_val = self.current_iter_left.next();

        let opt_val = if opt_val.is_none() {
            // iterators have met in the middle or at the end
            if self.idx_left == self.idx_right {
                return None;
            // one chunk is finished but there are still more chunks
            } else {
                self.chunk_idx_left += 1;
                self.set_current_iter_left();
            }
            // so we return the first value of the next chunk
            self.current_iter_left.next()
        } else {
            // we got a value
            opt_val
        };
        self.idx_left += 1;
        opt_val
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.ca.len();
        (len, Some(len))
    }
}

impl<'a, T> DoubleEndedIterator for NumIterManyChunk<'a, T>
where
    T: PolarsNumericType,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let opt_val = match &mut self.current_iter_right {
            Some(it) => it.next_back(),
            None => self.current_iter_left.next_back(),
        };

        let opt_val = if opt_val.is_none() {
            // iterators have met in the middle or at the beginning
            if self.idx_left == self.idx_right {
                return None;
            // one chunk is finished but there are still more chunks
            } else {
                self.chunk_idx_right -= 1;
                self.set_current_iter_right();
            }
            // so we return the first value of the next chunk from the back
            self.current_iter_left.next_back()
        } else {
            // we got a value
            opt_val
        };
        self.idx_right -= 1;
        opt_val
    }
}

/// Iterator for chunked arrays with many chunks.
/// The chunks have null values so it DOES perform null checks.
///
/// The return type is `Option<PolarsNumericType::Native>`.
pub struct NumIterManyChunkNullCheck<'a, T>
where
    T: PolarsNumericType,
{
    ca: &'a ChunkedArray<T>,
    chunks: Vec<&'a PrimitiveArray<T>>,
    // current iterator if we iterate from the left
    current_iter_left: Copied<Iter<'a, T::Native>>,
    current_data_left: ArrayDataRef,
    // index in the current iterator from left
    current_array_i_left: usize,
    // If iter_left and iter_right are the same, this is None and we need to use iter left
    // This is done because we can only have one owner
    current_iter_right: Option<Copied<Iter<'a, T::Native>>>,
    current_data_right: ArrayDataRef,
    current_array_i_right: usize,
    chunk_idx_left: usize,
    idx_left: usize,
    idx_right: usize,
    chunk_idx_right: usize,
}

impl<'a, T> NumIterManyChunkNullCheck<'a, T>
where
    T: PolarsNumericType,
{
    fn set_current_iter_left(&mut self) {
        let current_chunk = unsafe { self.chunks.get_unchecked(self.chunk_idx_left) };
        self.current_data_left = current_chunk.data();
        self.current_iter_left = current_chunk
            .value_slice(0, current_chunk.len())
            .iter()
            .copied();
    }

    fn set_current_iter_right(&mut self) {
        let current_chunk = unsafe { self.chunks.get_unchecked(self.chunk_idx_right) };
        self.current_data_right = current_chunk.data();
        if self.chunk_idx_left == self.chunk_idx_right {
            // from left and right we use the same iterator
            self.current_iter_right = None
        } else {
            self.current_iter_right = Some(
                current_chunk
                    .value_slice(0, current_chunk.len())
                    .iter()
                    .copied(),
            );
        }
    }

    fn new(ca: &'a ChunkedArray<T>) -> Self {
        let chunks = ca.downcast_chunks();
        let arr_left = chunks[0];
        let current_len_left = arr_left.len();
        let current_iter_left = arr_left.value_slice(0, current_len_left).iter().copied();
        let current_data_left = arr_left.data();

        let idx_left = 0;
        let chunk_idx_left = 0;
        let idx_right = ca.len();

        let chunk_idx_right = chunks.len() - 1;
        let current_iter_right;
        let arr = chunks[chunk_idx_right];
        let current_data_right = arr.data();
        if chunk_idx_left == chunk_idx_right {
            current_iter_right = None
        } else {
            current_iter_right = Some(arr.value_slice(0, arr.len()).iter().copied())
        }
        let current_array_i_right = arr.len();

        NumIterManyChunkNullCheck {
            ca,
            current_iter_left,
            current_data_left,
            current_array_i_left: 0,
            chunks,
            current_iter_right,
            current_data_right,
            current_array_i_right,
            idx_left,
            chunk_idx_left,
            idx_right,
            chunk_idx_right,
        }
    }
}

impl<'a, T> Iterator for NumIterManyChunkNullCheck<'a, T>
where
    T: PolarsNumericType,
{
    type Item = Option<T::Native>;

    fn next(&mut self) -> Option<Self::Item> {
        let opt_val = self.current_iter_left.next();

        let opt_val = if opt_val.is_none() {
            // iterators have met in the middle or at the end
            if self.idx_left == self.idx_right {
                return None;
            // one chunk is finished but there are still more chunks
            } else {
                self.chunk_idx_left += 1;
                // reset the index
                self.current_array_i_left = 0;
                self.set_current_iter_left();
            }
            // so we return the first value of the next chunk
            self.current_iter_left.next()
        } else {
            // we got a value
            opt_val
        };
        self.idx_left += 1;
        self.current_array_i_left += 1;
        if self
            .current_data_left
            .is_null(self.current_array_i_left - 1)
        {
            Some(None)
        } else {
            opt_val.map(Some)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.ca.len();
        (len, Some(len))
    }
}

impl<'a, T> DoubleEndedIterator for NumIterManyChunkNullCheck<'a, T>
where
    T: PolarsNumericType,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let opt_val = match &mut self.current_iter_right {
            Some(it) => it.next_back(),
            None => self.current_iter_left.next_back(),
        };

        let opt_val = if opt_val.is_none() {
            // iterators have met in the middle or at the beginning
            if self.idx_left == self.idx_right {
                return None;
            // one chunk is finished but there are still more chunks
            } else {
                self.chunk_idx_right -= 1;
                self.set_current_iter_right();
                // reset the index accumulator
                self.current_array_i_right = self.current_data_right.len()
            }
            // so we return the first value of the next chunk from the back
            self.current_iter_left.next_back()
        } else {
            // we got a value
            opt_val
        };
        self.idx_right -= 1;
        self.current_array_i_right -= 1;

        if self.current_data_right.is_null(self.current_array_i_right) {
            Some(None)
        } else {
            opt_val.map(Some)
        }
    }
}

impl<'a, T> IntoIterator for &'a ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = Option<T::Native>;
    type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        let chunks = self.downcast_chunks();
        match chunks.len() {
            1 => {
                if self.null_count() == 0 {
                    Box::new(SomeIterator(NumIterSingleChunk::new(self)))
                } else {
                    Box::new(NumIterSingleChunkNullCheck::new(self))
                }
            }
            _ => {
                if self.null_count() == 0 {
                    Box::new(SomeIterator(NumIterManyChunk::new(self)))
                } else {
                    Box::new(NumIterManyChunkNullCheck::new(self))
                }
            }
        }
    }
}

impl<'a, T> IntoNoNullIterator for &'a ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;
    type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;

    fn into_no_null_iter(self) -> Self::IntoIter {
        match self.chunks.len() {
            1 => Box::new(NumIterSingleChunk::new(self)),
            _ => Box::new(NumIterManyChunk::new(self)),
        }
    }
}

/// Creates and implement a iterator for chunked arrays with a single chunks and no null values, so
/// it iterates over the only chunk without performing null checks. Returns `iter_item`, as elements
/// cannot be null.
///
/// It also implements, for the created iterator, the following traits:
/// - Iterator
/// - DoubleEndedIterator
/// - ExactSizeIterator
/// - PolarsIterator
///
/// # Input
///
/// ca_type: The chunked array for which the single chunks iterator is implemented.
/// arrow_array: The arrow type of the chunked array chunks.
/// iterator_name: The name of the iterator struct to be implemented for a `SingleChunk` iterator.
/// iter_item: The iterator `Item`, the type which is going to be returned by the iterator.
/// (Optional) return_function: The function to apply to the each value of the chunked array before returning
///     the value.
macro_rules! impl_single_chunk_iterator {
    ($ca_type:ident, $arrow_array:ident, $iterator_name:ident, $iter_item:ty $(, $return_function:ident)?) => {
        impl<'a> ExactSizeIterator for $iterator_name<'a> {}
        impl<'a> PolarsIterator for $iterator_name<'a> {}

        /// Iterator for chunked arrays with just one chunk.
        /// The chunk cannot have null values so it does NOT perform null checks.
        ///
        /// The return type is `$iter_item`.
        pub struct $iterator_name<'a> {
            current_array: &'a $arrow_array,
            idx_left: usize,
            idx_right: usize,
        }

        impl<'a> $iterator_name<'a> {
            fn new(ca: &'a $ca_type) -> Self {
                let chunks = ca.downcast_chunks();
                let current_array = chunks[0];
                let idx_left = 0;
                let idx_right = current_array.len();

                Self {
                    current_array,
                    idx_left,
                    idx_right,
                }
            }
        }

        impl<'a> Iterator for $iterator_name<'a> {
            type Item = $iter_item;

            fn next(&mut self) -> Option<Self::Item> {
                // end of iterator or meet reversed iterator in the middle
                if self.idx_left == self.idx_right {
                    return None;
                }

                let v = self.current_array.value(self.idx_left);
                self.idx_left += 1;

                $(
                    // If return function is provided, apply the next function to the value.
                    // The result value will shadow, the `v` variable.
                    let v = $return_function("next", v);
                )?

                Some(v)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.current_array.len();
                (len, Some(len))
            }
        }

        impl<'a> DoubleEndedIterator for $iterator_name<'a> {
            fn next_back(&mut self) -> Option<Self::Item> {
                // end of iterator or meet reversed iterator in the middle
                if self.idx_left == self.idx_right {
                    return None;
                }
                self.idx_right -= 1;
                let v = self.current_array.value(self.idx_right);

                $(
                    // If return function is provided, apply the next_back function to the value.
                    // The result value will shadow, the `v` variable.
                    let v = $return_function("next_back", v);
                )?

                Some(v)
            }
        }
    };
}

/// Creates and implement a iterator for chunked arrays with a single chunks and null values, so
/// it iterates over the only chunk and performing null checks. Returns `Option<iter_item>`, as elements
/// can be null.
///
/// It also implements, for the created iterator, the following traits:
/// - Iterator
/// - DoubleEndedIterator
/// - ExactSizeIterator
/// - PolarsIterator
///
/// # Input
///
/// ca_type: The chunked array for which the which the single chunks with null check iterator is implemented.
/// arrow_array: The arrow type of the chunked array chunks.
/// iterator_name: The name of the iterator struct to be implemented for a `SingleChunkNullCheck` iterator.
/// iter_item: The iterator `Item`, the type which is going to be returned by the iterator.
/// (Optional) return_function: The function to apply to the each value of the chunked array before returning
///     the value.
macro_rules! impl_single_chunk_null_check_iterator {
    ($ca_type:ident, $arrow_array:ident, $iterator_name:ident, $iter_item:ty $(, $return_function:ident)?) => {
        impl<'a> ExactSizeIterator for $iterator_name<'a> {}
        impl<'a> PolarsIterator for $iterator_name<'a> {}

        /// Iterator for chunked arrays with just one chunk.
        /// The chunk have null values so it DOES perform null checks.
        ///
        /// The return type is `Option<$iter_item>`.
        pub struct $iterator_name<'a> {
            current_data: ArrayDataRef,
            current_array: &'a $arrow_array,
            idx_left: usize,
            idx_right: usize,
        }

        impl<'a> $iterator_name<'a> {
            fn new(ca: &'a $ca_type) -> Self {
                let chunks = ca.downcast_chunks();
                let current_array = chunks[0];
                let current_data = current_array.data();
                let idx_left = 0;
                let idx_right = current_array.len();

                Self {
                    current_data,
                    current_array,
                    idx_left,
                    idx_right,
                }
            }
        }

        impl<'a> Iterator for $iterator_name<'a> {
            type Item = Option<$iter_item>;

            fn next(&mut self) -> Option<Self::Item> {
                // end of iterator or meet reversed iterator in the middle
                if self.idx_left == self.idx_right {
                    return None;
                }
                let ret = if self.current_data.is_null(self.idx_left) {
                    Some(None)
                } else {
                    let v = self.current_array.value(self.idx_left);

                    $(
                        // If return function is provided, apply the next function to the value.
                        // The result value will shadow, the `v` variable.
                        let v = $return_function("next", v);
                    )?

                    Some(Some(v))
                };
                self.idx_left += 1;
                ret
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.current_array.len();
                (len, Some(len))
            }
        }

        impl<'a> DoubleEndedIterator for $iterator_name<'a> {
            fn next_back(&mut self) -> Option<Self::Item> {
                // end of iterator or meet reversed iterator in the middle
                if self.idx_left == self.idx_right {
                    return None;
                }
                self.idx_right -= 1;
                if self.current_data.is_null(self.idx_right) {
                    Some(None)
                } else {
                    let v = self.current_array.value(self.idx_right);

                    $(
                        // If return function is provided, apply the next_back function to the value.
                        // The result value will shadow, the `v` variable.
                        let v = $return_function("next_back", v);
                    )?

                    Some(Some(v))
                }
            }
        }
    };
}

/// Creates and implement a iterator for chunked arrays with many chunks and no null values, so
/// it iterates over several chunks without performing null checks. Returns `iter_item`, as elements
/// cannot be null.
///
/// It also implements, for the created iterator, the following traits:
/// - Iterator
/// - DoubleEndedIterator
/// - ExactSizeIterator
/// - PolarsIterator
///
/// # Input
///
/// ca_type: The chunked array for which the which the many chunks iterator is implemented.
/// arrow_array: The arrow type of the chunked array chunks.
/// iterator_name: The name of the iterator struct to be implemented for a `ManyChunk` iterator.
/// iter_item: The iterator `Item`, the type which is going to be returned by the iterator.
/// (Optional) return_function: The function to apply to the each value of the chunked array before returning
///     the value.
macro_rules! impl_many_chunk_iterator {
    ($ca_type:ident, $arrow_array:ident, $iterator_name:ident, $iter_item:ty $(, $return_function:ident)?) => {
        impl<'a> ExactSizeIterator for $iterator_name<'a> {}
        impl<'a> PolarsIterator for $iterator_name<'a> {}

        /// Iterator for chunked arrays with many chunks.
        /// The chunks cannot have null values so it does NOT perform null checks.
        ///
        /// The return type is `$iter_item`.
        pub struct $iterator_name<'a> {
            ca: &'a $ca_type,
            chunks: Vec<&'a $arrow_array>,
            current_array_left: &'a $arrow_array,
            current_array_right: &'a $arrow_array,
            current_array_idx_left: usize,
            current_array_idx_right: usize,
            current_array_left_len: usize,
            idx_left: usize,
            idx_right: usize,
            chunk_idx_left: usize,
            chunk_idx_right: usize,
        }

        impl<'a> $iterator_name<'a> {
            fn new(ca: &'a $ca_type) -> Self {
                let chunks = ca.downcast_chunks();
                let current_array_left = chunks[0];
                let idx_left = 0;
                let chunk_idx_left = 0;
                let chunk_idx_right = chunks.len() - 1;
                let current_array_right = chunks[chunk_idx_right];
                let idx_right = ca.len();
                let current_array_idx_left = 0;
                let current_array_idx_right = current_array_right.len();
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

        impl<'a> Iterator for $iterator_name<'a> {
            type Item = $iter_item;

            fn next(&mut self) -> Option<Self::Item> {
                // end of iterator or meet reversed iterator in the middle
                if self.idx_left == self.idx_right {
                    return None;
                }

                // return value
                let ret = self.current_array_left.value(self.current_array_idx_left);

                // increment index pointers
                self.idx_left += 1;
                self.current_array_idx_left += 1;

                // we've reached the end of the chunk
                if self.current_array_idx_left == self.current_array_left_len {
                    // Set a new chunk as current data
                    self.chunk_idx_left += 1;

                    // if this evaluates to False, next call will be end of iterator
                    if self.chunk_idx_left < self.chunks.len() {
                        // reset to new array
                        self.current_array_idx_left = 0;
                        self.current_array_left = self.chunks[self.chunk_idx_left];
                        self.current_array_left_len = self.current_array_left.len();
                    }
                }

                $(
                    // If return function is provided, apply the next function to the value.
                    // The result value will shadow, the `ret` variable.
                    let ret = $return_function("next", ret);
                )?

                Some(ret)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.ca.len();
                (len, Some(len))
            }
        }

        impl<'a> DoubleEndedIterator for $iterator_name<'a> {
            fn next_back(&mut self) -> Option<Self::Item> {
                // end of iterator or meet reversed iterator in the middle
                if self.idx_left == self.idx_right {
                    return None;
                }
                self.idx_right -= 1;
                self.current_array_idx_right -= 1;

                let ret = self.current_array_right.value(self.current_array_idx_right);

                // we've reached the end of the chunk from the right
                if self.current_array_idx_right == 0 && self.idx_right > 0 {
                    // set a new chunk as current data
                    self.chunk_idx_right -= 1;
                    // reset to new array
                    self.current_array_right = self.chunks[self.chunk_idx_right];
                    self.current_array_idx_right = self.current_array_right.len();
                }

                $(
                    // If return function is provided, apply the next_back function to the value.
                    // The result value will shadow, the `ret` variable.
                    let ret = $return_function("next_back", ret);
                )?

                Some(ret)
            }
        }
    };
}

/// Creates and implement a iterator for chunked arrays with many chunks and null values, so
/// it iterates over several chunks and perform null checks. Returns `Option<iter_item>`, as elements
/// can be null.
///
/// It also implements, for the created iterator, the following traits:
/// - Iterator
/// - DoubleEndedIterator
/// - ExactSizeIterator
/// - PolarsIterator
///
/// # Input
///
/// ca_type: The chunked array for which the which the many chunks with null check iterator is implemented.
/// arrow_array: The arrow type of the chunked array chunks.
/// iterator_name: The name of the iterator struct to be implemented for a `ManyChunkNullCheck` iterator.
/// iter_item: The iterator `Item`, the type which is going to be returned by the iterator, wrapped
///     into and `Option`, as null check is performed by this iterator.
/// (Optional) return_function: The function to apply to the each value of the chunked array before returning
///     the value.
macro_rules! impl_many_chunk_null_check_iterator {
    ($ca_type:ident, $arrow_array:ident, $iterator_name:ident, $iter_item:ty $(, $return_function:ident)? ) => {
        impl<'a> ExactSizeIterator for $iterator_name<'a> {}
        impl<'a> PolarsIterator for $iterator_name<'a> {}

        /// Iterator for chunked arrays with many chunks.
        /// The chunks have null values so it DOES perform null checks.
        ///
        /// The return type is `Option<$iter_item>`.
        pub struct $iterator_name<'a> {
            ca: &'a $ca_type,
            chunks: Vec<&'a $arrow_array>,
            current_data_left: ArrayDataRef,
            current_array_left: &'a $arrow_array,
            current_data_right: ArrayDataRef,
            current_array_right: &'a $arrow_array,
            current_array_idx_left: usize,
            current_array_idx_right: usize,
            current_array_left_len: usize,
            idx_left: usize,
            idx_right: usize,
            chunk_idx_left: usize,
            chunk_idx_right: usize,
        }

        impl<'a> $iterator_name<'a> {
            fn new(ca: &'a $ca_type) -> Self {
                let chunks = ca.downcast_chunks();
                let current_array_left = chunks[0];
                let current_data_left = current_array_left.data();
                let idx_left = 0;
                let chunk_idx_left = 0;
                let chunk_idx_right = chunks.len() - 1;
                let current_array_right = chunks[chunk_idx_right];
                let current_data_right = current_array_right.data();
                let idx_right = ca.len();
                let current_array_idx_left = 0;
                let current_array_idx_right = current_data_right.len();
                let current_array_left_len = current_array_left.len();

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

        impl<'a> Iterator for $iterator_name<'a> {
            type Item = Option<$iter_item>;

            fn next(&mut self) -> Option<Self::Item> {
                // end of iterator or meet reversed iterator in the middle
                if self.idx_left == self.idx_right {
                    return None;
                }

                // return value
                let ret;
                if self.current_array_left.is_null(self.current_array_idx_left) {
                    ret = None
                } else {
                    let v = self.current_array_left.value(self.current_array_idx_left);

                    $(
                        // If return function is provided, apply the next function to the value.
                        // The result value will shadow, the `v` variable.
                        let v = $return_function("next", v);
                    )?

                    ret = Some(v);
                }

                // increment index pointers
                self.idx_left += 1;
                self.current_array_idx_left += 1;

                // we've reached the end of the chunk
                if self.current_array_idx_left == self.current_array_left_len {
                    // Set a new chunk as current data
                    self.chunk_idx_left += 1;

                    // if this evaluates to False, next call will be end of iterator
                    if self.chunk_idx_left < self.chunks.len() {
                        // reset to new array
                        self.current_array_idx_left = 0;
                        self.current_array_left = self.chunks[self.chunk_idx_left];
                        self.current_data_left = self.current_array_left.data();
                        self.current_array_left_len = self.current_array_left.len();
                    }
                }

                Some(ret)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.ca.len();
                (len, Some(len))
            }
        }

        impl<'a> DoubleEndedIterator for $iterator_name<'a> {
            fn next_back(&mut self) -> Option<Self::Item> {
                // end of iterator or meet reversed iterator in the middle
                if self.idx_left == self.idx_right {
                    return None;
                }
                self.idx_right -= 1;
                self.current_array_idx_right -= 1;

                let ret = if self
                    .current_data_right
                    .is_null(self.current_array_idx_right)
                {
                    Some(None)
                } else {
                    let v = self.current_array_right.value(self.current_array_idx_right);

                    $(
                        // If return function is provided, apply the next_back function to the value.
                        // The result value will shadow, the `v` variable.
                        let v = $return_function("next_back", v);
                    )?

                    Some(Some(v))
                };

                // we've reached the end of the chunk from the right
                if self.current_array_idx_right == 0 && self.idx_right > 0 {
                    // set a new chunk as current data
                    self.chunk_idx_right -= 1;
                    // reset to new array
                    self.current_array_right = self.chunks[self.chunk_idx_right];
                    self.current_data_right = self.current_array_right.data();
                    self.current_array_idx_right = self.current_array_right.len();
                }
                ret
            }
        }
    };
}

/// Implement the `IntoIterator` to convert a given chunked array type into a `PolarsIterator`
/// with null checks.
///
/// # Input
///
/// ca_type: The chunked array for which the `IntoIterator` trait is implemented.
/// iter_item: The iterator `Item`, the type which is going to be returned by the iterator, wrapped
///     into and `Option`, as null check is performed.
/// single_chunk_ident: Identifier for the struct representing a single chunk without null
///     check iterator. Which returns `iter_item`.
/// single_chunk_null_ident: Identifier for the struct representing a single chunk with null
///     check iterator. Which returns `Option<iter_item>`.
/// many_chunk_ident: Identifier for the struct representing a many chunk without null
///     check iterator. Which returns `iter_item`.
/// many_chunk_null_ident: Identifier for the struct representing a many chunk with null
///     check iterator. Which returns `Option<iter_item>`.
macro_rules! impl_into_polars_iterator {
    ($ca_type:ident, $iter_item:ty, $single_chunk_ident:ident, $single_chunk_null_ident:ident, $many_chunk_ident:ident, $many_chunk_null_ident:ident) => {
        impl<'a> IntoIterator for &'a $ca_type {
            type Item = Option<$iter_item>;
            type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;

            /// Decides which iterator fits best the current chunked array. The decision are based
            /// on the number of chunks and the existence of null values.
            fn into_iter(self) -> Self::IntoIter {
                let chunks = self.downcast_chunks();
                match chunks.len() {
                    1 => {
                        if self.null_count() == 0 {
                            Box::new(SomeIterator($single_chunk_ident::new(self)))
                        } else {
                            Box::new($single_chunk_null_ident::new(self))
                        }
                    }
                    _ => {
                        if self.null_count() == 0 {
                            Box::new(SomeIterator($many_chunk_ident::new(self)))
                        } else {
                            Box::new($many_chunk_null_ident::new(self))
                        }
                    }
                }
            }
        }
    };
}

/// Implement the `IntoNoNullIterator` to convert a given chunked array type into a `PolarsIterator`
/// without null checks.
///
/// # Input
///
/// ca_type: The chunked array for which the `IntoNoNull` trait is implemented.
/// iter_item: The iterator `Item`, the type which is going to be returned by the iterator.
///     The return type is not wrapped into `Option` as the chunked array shall not have
///     null values.
/// single_chunk_ident: Identifier for the struct representing a single chunk without null
///     check iterator. Which returns `iter_item`.
/// many_chunk_ident: Identifier for the struct representing a many chunk without null
///     check iterator. Which returns `iter_item`.
macro_rules! impl_into_no_null_polars_iterator {
    ($ca_type:ident, $iter_item:ty, $single_chunk_ident:ident, $many_chunk_ident:ident) => {
        impl<'a> IntoNoNullIterator for &'a $ca_type {
            type Item = $iter_item;
            type IntoIter = Box<dyn PolarsIterator<Item = Self::Item> + 'a>;

            /// Decides which iterator fits best the current no null chunked array. The decision are based
            /// on the number of chunks.
            fn into_no_null_iter(self) -> Self::IntoIter {
                match self.chunks.len() {
                    1 => Box::new($single_chunk_ident::new(self)),
                    _ => Box::new($many_chunk_ident::new(self)),
                }
            }
        }
    };
}

/// Generates all the iterators and implements its traits. Also implement the `IntoIterator` and `IntoNoNullIterator`.
/// - SingleChunkIterator
/// - SingleChunkIteratorNullCheck
/// - ManyChunkIterator
/// - ManyChunkIteratorNullCheck
/// - IntoIterator
/// - IntoNoNullIterator
///
/// # Input
///
/// ca_type: The chunked array for which the iterators are implemented. The `IntoIterator` and `IntoNoNullIterator`
///     traits are going to be implemented for this chunked array.
/// arrow_array: The arrow type of the chunked array chunks.
/// single_chunk_ident: The name of the `SingleChunkIterator` to create.
/// single_chunk_null_ident: The name of the `SingleChunkIteratorNullCheck` iterator to create.
/// many_chunk_ident: The name of the `ManyChunkIterator` to create.
/// many_chunk_null_ident: The name of the `ManyChunkIteratorNullCheck` iterator to create.
/// iter_item: The iterator item. `NullCheck` iterators and `IntoIterator` will wrap this iter into an `Option`.
/// (Optional) return_function: The function to apply to the each value of the chunked array before returning
///     the value.
macro_rules! impl_all_iterators {
    ($ca_type:ident,
     $arrow_array:ident,
     $single_chunk_ident:ident,
     $single_chunk_null_ident:ident,
     $many_chunk_ident:ident,
     $many_chunk_null_ident:ident,
     $iter_item:ty
     $(, $return_function: ident )?
    ) => {
        // Generate single chunk iterator.
        impl_single_chunk_iterator!(
            $ca_type,
            $arrow_array,
            $single_chunk_ident,
            $iter_item
            $(, $return_function )? // Optional argument, only used if provided
        );

        // Generate single chunk iterator with null checks.
        impl_single_chunk_null_check_iterator!(
            $ca_type,
            $arrow_array,
            $single_chunk_null_ident,
            $iter_item
            $(, $return_function )? // Optional argument, only used if provided
        );

        // Generate many chunk iterator.
        impl_many_chunk_iterator!(
            $ca_type,
            $arrow_array,
            $many_chunk_ident,
            $iter_item
            $(, $return_function )? // Optional argument, only used if provided
        );

        // Generate many chunk iterator with null checks.
        impl_many_chunk_null_check_iterator!(
            $ca_type,
            $arrow_array,
            $many_chunk_null_ident,
            $iter_item
            $(, $return_function )? // Optional argument, only used if provided
        );

        // Generate into iterator function.
        impl_into_polars_iterator!(
            $ca_type,
            $iter_item,
            $single_chunk_ident,
            $single_chunk_null_ident,
            $many_chunk_ident,
            $many_chunk_null_ident
        );

        // Generate into no null iterator function.
        impl_into_no_null_polars_iterator!(
            $ca_type,
            $iter_item,
            $single_chunk_ident,
            $many_chunk_ident
        );
    }
}

impl_all_iterators!(
    Utf8Chunked,
    StringArray,
    Utf8IterSingleChunk,
    Utf8IterSingleChunkNullCheck,
    Utf8IterManyChunk,
    Utf8IterManyChunkNullCheck,
    &'a str
);
impl_all_iterators!(
    BooleanChunked,
    BooleanArray,
    BooleanIterSingleChunk,
    BooleanIterSingleChunkNullCheck,
    BooleanIterManyChunk,
    BooleanIterManyChunkNullCheck,
    bool
);

// used for macro
fn return_from_list_iter(method_name: &str, v: ArrayRef) -> Series {
    (method_name, v).into()
}

impl_all_iterators!(
    ListChunked,
    ListArray,
    ListIterSingleChunk,
    ListIterSingleChunkNullCheck,
    ListIterManyChunk,
    ListIterManyChunkNullCheck,
    Series,
    return_from_list_iter
);

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn out_of_bounds() {
        let mut a = UInt32Chunked::new_from_slice("a", &[1, 2, 3]);
        let b = UInt32Chunked::new_from_slice("a", &[1, 2, 3]);
        a.append(&b);

        let v = a.into_iter().collect::<Vec<_>>();
        assert_eq!(
            vec![Some(1u32), Some(2), Some(3), Some(1), Some(2), Some(3)],
            v
        )
    }

    /// Generate test for `IntoIterator` trait for chunked arrays with just one chunk and no null values.
    /// The expected return value of the iterator generated by `IntoIterator` trait is `Option<T>`, where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: `Utf8Chunked`, `UInt32Chunked` ...
    /// first_val: The first value contained in the chunked array.
    /// second_val: The second value contained in the chunked array.
    /// third_val: The third value contained in the chunked array.
    macro_rules! impl_test_iter_single_chunk {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let a = <$ca_type>::new_from_slice("test", &[$first_val, $second_val, $third_val]);

                // normal iterator
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some(Some($first_val)));
                assert_eq!(it.next(), Some(Some($second_val)));
                assert_eq!(it.next(), Some(Some($third_val)));
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // reverse iterator
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some(Some($third_val)));
                assert_eq!(it.next_back(), Some(Some($second_val)));
                assert_eq!(it.next_back(), Some(Some($first_val)));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);

                // iterators should not cross
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some(Some($third_val)));
                assert_eq!(it.next(), Some(Some($first_val)));
                assert_eq!(it.next(), Some(Some($second_val)));
                // should stop here as we took this one from the back
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // do the same from the right side
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some(Some($first_val)));
                assert_eq!(it.next_back(), Some(Some($third_val)));
                assert_eq!(it.next_back(), Some(Some($second_val)));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);
            }
        };
    }

    impl_test_iter_single_chunk!(num_iter_single_chunk, UInt32Chunked, 1, 2, 3);
    impl_test_iter_single_chunk!(utf8_iter_single_chunk, Utf8Chunked, "a", "b", "c");
    impl_test_iter_single_chunk!(bool_iter_single_chunk, BooleanChunked, true, true, false);

    /// Generate test for `IntoIterator` trait for chunked arrays with just one chunk and null values.
    /// The expected return value of the iterator generated by `IntoIterator` trait is `Option<T>`, where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: `Utf8Chunked`, `UInt32Chunked` ...
    /// first_val: The first value contained in the chunked array. Must be an `Option<T>`.
    /// second_val: The second value contained in the chunked array. Must be an `Option<T>`.
    /// third_val: The third value contained in the chunked array. Must be an `Option<T>`.
    macro_rules! impl_test_iter_single_chunk_null_check {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let a =
                    <$ca_type>::new_from_opt_slice("test", &[$first_val, $second_val, $third_val]);

                // normal iterator
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                assert_eq!(it.next(), Some($third_val));
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // reverse iterator
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), Some($first_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);

                // iterators should not cross
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                // should stop here as we took this one from the back
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // do the same from the right side
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);
            }
        };
    }

    impl_test_iter_single_chunk_null_check!(
        num_iter_single_chunk_null_check,
        UInt32Chunked,
        Some(1),
        None,
        Some(3)
    );
    impl_test_iter_single_chunk_null_check!(
        utf8_iter_single_chunk_null_check,
        Utf8Chunked,
        Some("a"),
        None,
        Some("c")
    );
    impl_test_iter_single_chunk_null_check!(
        bool_iter_single_chunk_null_check,
        BooleanChunked,
        Some(true),
        None,
        Some(false)
    );

    /// Generate test for `IntoIterator` trait for chunked arrays with many chunks and no null values.
    /// The expected return value of the iterator generated by `IntoIterator` trait is `Option<T>`, where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: `Utf8Chunked`, `UInt32Chunked` ...
    /// first_val: The first value contained in the chunked array.
    /// second_val: The second value contained in the chunked array.
    /// third_val: The third value contained in the chunked array.
    macro_rules! impl_test_iter_many_chunk {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let mut a = <$ca_type>::new_from_slice("test", &[$first_val, $second_val]);
                let a_b = <$ca_type>::new_from_slice("", &[$third_val]);
                a.append(&a_b);

                // normal iterator
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some(Some($first_val)));
                assert_eq!(it.next(), Some(Some($second_val)));
                assert_eq!(it.next(), Some(Some($third_val)));
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // reverse iterator
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some(Some($third_val)));
                assert_eq!(it.next_back(), Some(Some($second_val)));
                assert_eq!(it.next_back(), Some(Some($first_val)));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);

                // iterators should not cross
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some(Some($third_val)));
                assert_eq!(it.next(), Some(Some($first_val)));
                assert_eq!(it.next(), Some(Some($second_val)));
                // should stop here as we took this one from the back
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // do the same from the right side
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some(Some($first_val)));
                assert_eq!(it.next_back(), Some(Some($third_val)));
                assert_eq!(it.next_back(), Some(Some($second_val)));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);
            }
        };
    }

    impl_test_iter_many_chunk!(num_iter_many_chunk, UInt32Chunked, 1, 2, 3);
    impl_test_iter_many_chunk!(utf8_iter_many_chunk, Utf8Chunked, "a", "b", "c");
    impl_test_iter_many_chunk!(bool_iter_many_chunk, BooleanChunked, true, true, false);

    /// Generate test for `IntoIterator` trait for chunked arrays with many chunk and null values.
    /// The expected return value of the iterator generated by `IntoIterator` trait is `Option<T>`, where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: `Utf8Chunked`, `UInt32Chunked` ...
    /// first_val: The first value contained in the chunked array. Must be an `Option<T>`.
    /// second_val: The second value contained in the chunked array. Must be an `Option<T>`.
    /// third_val: The third value contained in the chunked array. Must be an `Option<T>`.
    macro_rules! impl_test_iter_many_chunk_null_check {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let mut a = <$ca_type>::new_from_opt_slice("test", &[$first_val, $second_val]);
                let a_b = <$ca_type>::new_from_opt_slice("", &[$third_val]);
                a.append(&a_b);

                // normal iterator
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                assert_eq!(it.next(), Some($third_val));
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // reverse iterator
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), Some($first_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);

                // iterators should not cross
                let mut it = a.into_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                // should stop here as we took this one from the back
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // do the same from the right side
                let mut it = a.into_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);
            }
        };
    }

    impl_test_iter_many_chunk_null_check!(
        num_iter_many_chunk_null_check,
        UInt32Chunked,
        Some(1),
        None,
        Some(3)
    );
    impl_test_iter_many_chunk_null_check!(
        utf8_iter_many_chunk_null_check,
        Utf8Chunked,
        Some("a"),
        None,
        Some("c")
    );
    impl_test_iter_many_chunk_null_check!(
        bool_iter_many_chunk_null_check,
        BooleanChunked,
        Some(true),
        None,
        Some(false)
    );

    /// Generate test for `IntoNoNullIterator` trait for chunked arrays with just one chunk and no null values.
    /// The expected return value of the iterator generated by `IntoNoNullIterator` trait is `T`, where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: `Utf8Chunked`, `UInt32Chunked` ...
    /// first_val: The first value contained in the chunked array.
    /// second_val: The second value contained in the chunked array.
    /// third_val: The third value contained in the chunked array.
    macro_rules! impl_test_no_null_iter_single_chunk {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let a = <$ca_type>::new_from_slice("test", &[$first_val, $second_val, $third_val]);

                // normal iterator
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                assert_eq!(it.next(), Some($third_val));
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // reverse iterator
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), Some($first_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);

                // iterators should not cross
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                // should stop here as we took this one from the back
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // do the same from the right side
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);
            }
        };
    }

    impl_test_no_null_iter_single_chunk!(num_no_null_iter_single_chunk, UInt32Chunked, 1, 2, 3);
    impl_test_no_null_iter_single_chunk!(
        utf8_no_null_iter_single_chunk,
        Utf8Chunked,
        "a",
        "b",
        "c"
    );
    impl_test_no_null_iter_single_chunk!(
        bool_no_null_iter_single_chunk,
        BooleanChunked,
        true,
        true,
        false
    );

    /// Generate test for `IntoNoNullIterator` trait for chunked arrays with many chunks and no null values.
    /// The expected return value of the iterator generated by `IntoNoNullIterator` trait is `T`, where
    /// `T` is the chunked array type.
    ///
    /// # Input
    ///
    /// test_name: The name of the test to generate.
    /// ca_type: The chunked array to use for this test. Ex: `Utf8Chunked`, `UInt32Chunked` ...
    /// first_val: The first value contained in the chunked array.
    /// second_val: The second value contained in the chunked array.
    /// third_val: The third value contained in the chunked array.
    macro_rules! impl_test_no_null_iter_many_chunk {
        ($test_name:ident, $ca_type:ty, $first_val:expr, $second_val:expr, $third_val:expr) => {
            #[test]
            fn $test_name() {
                let mut a = <$ca_type>::new_from_slice("test", &[$first_val, $second_val]);
                let a_b = <$ca_type>::new_from_slice("", &[$third_val]);
                a.append(&a_b);

                // normal iterator
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                assert_eq!(it.next(), Some($third_val));
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // reverse iterator
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), Some($first_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);

                // iterators should not cross
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next(), Some($second_val));
                // should stop here as we took this one from the back
                assert_eq!(it.next(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next_back(), None);

                // do the same from the right side
                let mut it = a.into_no_null_iter();
                assert_eq!(it.next(), Some($first_val));
                assert_eq!(it.next_back(), Some($third_val));
                assert_eq!(it.next_back(), Some($second_val));
                assert_eq!(it.next_back(), None);
                // ensure both sides are consumes.
                assert_eq!(it.next(), None);
            }
        };
    }

    impl_test_no_null_iter_many_chunk!(num_no_null_iter_many_chunk, UInt32Chunked, 1, 2, 3);
    impl_test_no_null_iter_many_chunk!(utf8_no_null_iter_many_chunk, Utf8Chunked, "a", "b", "c");
    impl_test_no_null_iter_many_chunk!(
        bool_no_null_iter_many_chunk,
        BooleanChunked,
        true,
        true,
        false
    );
}
