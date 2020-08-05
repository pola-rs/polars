use crate::chunked_array::builder::{PrimitiveChunkedBuilder, Utf8ChunkedBuilder};
use crate::prelude::*;
use crate::utils::Xob;
use arrow::array::{Array, ArrayDataRef, BooleanArray, PrimitiveArray, StringArray};
use arrow::datatypes::ArrowPrimitiveType;
use std::iter::Copied;
use std::iter::FromIterator;
use std::slice::Iter;
use unsafe_unwrap::UnsafeUnwrap;

// ExactSizeIterator trait implementations for all Iterator structs in this file
impl<'a, T> ExactSizeIterator for NumIterSingleChunkNullCheck<'a, T> where T: PolarsNumericType {}
impl<'a, T> ExactSizeIterator for NumIterSingleChunk<'a, T>
where
    T: PolarsNumericType,
    T::Native: Copy,
{
}
impl<'a, T> ExactSizeIterator for NumIterManyChunkNullCheck<'a, T> where T: PolarsNumericType {}
impl<'a, T> ExactSizeIterator for NumIterManyChunk<'a, T> where T: PolarsNumericType {}
impl<'a> ExactSizeIterator for Utf8IterSingleChunk<'a> {}
impl<'a> ExactSizeIterator for Utf8IterSingleChunkNullCheck<'a> {}
impl<'a> ExactSizeIterator for Utf8IterManyChunk<'a> {}
impl<'a> ExactSizeIterator for Utf8IterManyChunkNullCheck<'a> {}

// Helper trait needed for dynamic dispatch
pub trait ExactSizeDoubleEndedIterator: ExactSizeIterator + DoubleEndedIterator {}

impl<'a, T: PolarsNumericType> ExactSizeDoubleEndedIterator for NumIterSingleChunk<'a, T> {}
impl<'a, T: PolarsNumericType> ExactSizeDoubleEndedIterator for NumIterSingleChunkNullCheck<'a, T> {}
impl<'a, T: PolarsNumericType> ExactSizeDoubleEndedIterator for NumIterManyChunk<'a, T> {}
impl<'a, T: PolarsNumericType> ExactSizeDoubleEndedIterator for NumIterManyChunkNullCheck<'a, T> {}
impl<'a> ExactSizeDoubleEndedIterator for Utf8IterSingleChunk<'a> {}
impl<'a> ExactSizeDoubleEndedIterator for Utf8IterSingleChunkNullCheck<'a> {}
impl<'a> ExactSizeDoubleEndedIterator for Utf8IterManyChunk<'a> {}
impl<'a> ExactSizeDoubleEndedIterator for Utf8IterManyChunkNullCheck<'a> {}

/// Single chunk with null values
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
    fn return_opt_val(&self, index: usize) -> Option<Option<T::Native>> {
        if self.arr.is_null(index) {
            Some(None)
        } else {
            Some(Some(self.arr.value(index)))
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
            self.idx += 1;
            self.return_opt_val(self.idx - 1)
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
            self.return_opt_val(self.back_idx)
        }
    }
}

/// Single chunk no null values
pub struct NumIterSingleChunk<'a, T>
where
    T: PolarsNumericType,
    T::Native: Copy,
{
    iter: Copied<Iter<'a, T::Native>>,
}

impl<'a, T> Iterator for NumIterSingleChunk<'a, T>
where
    T: PolarsNumericType,
    T::Native: Copy,
{
    type Item = Option<T::Native>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(Some)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for NumIterSingleChunk<'a, T>
where
    T: PolarsNumericType,
    T::Native: Copy,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(Some)
    }
}

/// Many chunks no null checks
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
        if self.chunk_idx_left == self.chunk_idx_left {
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
                self.set_current_iter_left();
            }
            // so we return the first value of the next chunk
            self.current_iter_left.next()
        } else {
            // we got a value
            opt_val
        };
        self.idx_left += 1;
        opt_val.map(Some)
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
        opt_val.map(Some)
    }
}

/// Many chunks with null checks
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
        if self.chunk_idx_left == self.chunk_idx_left {
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
    type IntoIter = Box<dyn ExactSizeDoubleEndedIterator<Item = Option<T::Native>> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        match self.cont_slice() {
            Ok(slice) => {
                // Compile could not infer T.
                let a: NumIterSingleChunk<'_, T> = NumIterSingleChunk {
                    iter: slice.iter().copied(),
                };
                Box::new(a)
            }
            Err(_) => {
                let chunks = self.downcast_chunks();
                match chunks.len() {
                    1 => {
                        let arr = chunks[0];
                        let len = arr.len();
                        Box::new(NumIterSingleChunkNullCheck {
                            arr,
                            idx: 0,
                            back_idx: len,
                        })
                    }
                    _ => {
                        if self.null_count() == 0 {
                            Box::new(NumIterManyChunk::new(self))
                        } else {
                            Box::new(NumIterManyChunkNullCheck::new(self))
                        }
                    }
                }
            }
        }
    }
}

/// No null checks
pub struct Utf8IterSingleChunk<'a> {
    current_array: &'a StringArray,
    idx_left: usize,
    idx_right: usize,
}

impl<'a> Utf8IterSingleChunk<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
        let chunks = ca.downcast_chunks();
        let current_array = chunks[0];
        let idx_left = 0;
        let idx_right = current_array.len();

        Utf8IterSingleChunk {
            current_array,
            idx_left,
            idx_right,
        }
    }
}

impl<'a> Iterator for Utf8IterSingleChunk<'a> {
    type Item = Option<&'a str>;

    fn next(&mut self) -> Option<Self::Item> {
        // end of iterator or meet reversed iterator in the middle
        if self.idx_left == self.idx_right {
            return None;
        }

        let v = self.current_array.value(self.idx_left);
        self.idx_left += 1;
        Some(Some(v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.current_array.len();
        (len, Some(len))
    }
}

impl<'a> DoubleEndedIterator for Utf8IterSingleChunk<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        // end of iterator or meet reversed iterator in the middle
        if self.idx_left == self.idx_right {
            return None;
        }
        self.idx_right -= 1;
        Some(Some(self.current_array.value(self.idx_right)))
    }
}

pub struct Utf8IterSingleChunkNullCheck<'a> {
    current_data: ArrayDataRef,
    current_array: &'a StringArray,
    idx_left: usize,
    idx_right: usize,
}

impl<'a> Utf8IterSingleChunkNullCheck<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
        let chunks = ca.downcast_chunks();
        let current_array = chunks[0];
        let current_data = current_array.data();
        let idx_left = 0;
        let idx_right = current_array.len();

        Utf8IterSingleChunkNullCheck {
            current_data,
            current_array,
            idx_left,
            idx_right,
        }
    }
}

impl<'a> Iterator for Utf8IterSingleChunkNullCheck<'a> {
    type Item = Option<&'a str>;

    fn next(&mut self) -> Option<Self::Item> {
        // end of iterator or meet reversed iterator in the middle
        if self.idx_left == self.idx_right {
            return None;
        }
        let ret;
        if self.current_data.is_null(self.idx_left) {
            ret = Some(None)
        } else {
            let v = self.current_array.value(self.idx_left);
            ret = Some(Some(v))
        }
        self.idx_left += 1;
        ret
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.current_array.len();
        (len, Some(len))
    }
}

impl<'a> DoubleEndedIterator for Utf8IterSingleChunkNullCheck<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        // end of iterator or meet reversed iterator in the middle
        if self.idx_left == self.idx_right {
            return None;
        }
        self.idx_right -= 1;
        if self.current_data.is_null(self.idx_right) {
            Some(None)
        } else {
            Some(Some(self.current_array.value(self.idx_right)))
        }
    }
}

/// Many chunks no nulls
pub struct Utf8IterManyChunk<'a> {
    ca: &'a Utf8Chunked,
    chunks: Vec<&'a StringArray>,
    current_array_left: &'a StringArray,
    current_array_right: &'a StringArray,
    current_array_idx_left: usize,
    current_array_idx_right: usize,
    current_array_left_len: usize,
    idx_left: usize,
    idx_right: usize,
    chunk_idx_left: usize,
    chunk_idx_right: usize,
}

impl<'a> Utf8IterManyChunk<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
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

impl<'a> Iterator for Utf8IterManyChunk<'a> {
    type Item = Option<&'a str>;

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
        Some(Some(ret))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.ca.len();
        (len, Some(len))
    }
}

impl<'a> DoubleEndedIterator for Utf8IterManyChunk<'a> {
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
        Some(Some(ret))
    }
}

/// Many chunks no nulls
pub struct Utf8IterManyChunkNullCheck<'a> {
    ca: &'a Utf8Chunked,
    chunks: Vec<&'a StringArray>,
    current_data_left: ArrayDataRef,
    current_array_left: &'a StringArray,
    current_data_right: ArrayDataRef,
    current_array_right: &'a StringArray,
    current_array_idx_left: usize,
    current_array_idx_right: usize,
    current_array_left_len: usize,
    idx_left: usize,
    idx_right: usize,
    chunk_idx_left: usize,
    chunk_idx_right: usize,
}

impl<'a> Utf8IterManyChunkNullCheck<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
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

impl<'a> Iterator for Utf8IterManyChunkNullCheck<'a> {
    type Item = Option<&'a str>;

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
            ret = Some(self.current_array_left.value(self.current_array_idx_left));
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

impl<'a> DoubleEndedIterator for Utf8IterManyChunkNullCheck<'a> {
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
            Some(Some(
                self.current_array_right.value(self.current_array_idx_right),
            ))
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

impl<'a> IntoIterator for &'a Utf8Chunked {
    type Item = Option<&'a str>;
    type IntoIter = Box<dyn ExactSizeDoubleEndedIterator<Item = Option<&'a str>> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        let chunks = self.downcast_chunks();
        match chunks.len() {
            1 => {
                if self.null_count() == 0 {
                    Box::new(Utf8IterSingleChunk::new(self))
                } else {
                    Box::new(Utf8IterSingleChunkNullCheck::new(self))
                }
            }
            _ => {
                if self.null_count() == 0 {
                    Box::new(Utf8IterManyChunk::new(self))
                } else {
                    Box::new(Utf8IterManyChunkNullCheck::new(self))
                }
            }
        }
    }
}

macro_rules! set_indexes {
    ($self:ident) => {{
        $self.array_i += 1;
        if $self.array_i >= $self.current_len {
            // go to next array in the chunks
            $self.array_i = 0;
            $self.chunk_i += 1;

            if $self.chunk_i < $self.array_chunks.len() {
                // not yet at last chunk
                let arr = unsafe { *$self.array_chunks.get_unchecked($self.chunk_i) };
                $self.current_data = Some(arr.data());
                $self.current_array = Some(arr);
                $self.current_len = arr.len();
            }
        }
    }};
}

macro_rules! get_opt_value_and_set_indexes {
    ($self:ident, $current_data:ident, $current_array:ident) => {{
        let ret;
        if $current_data.is_null($self.array_i) {
            ret = Some(None)
        } else {
            let v = $current_array.value($self.array_i);
            ret = Some(Some(v));
        };
        $self.set_indexes();
        ret
    }};
}

pub struct ChunkBoolIter<'a> {
    array_chunks: Vec<&'a BooleanArray>,
    current_data: Option<ArrayDataRef>,
    current_array: Option<&'a BooleanArray>,
    current_len: usize,
    chunk_i: usize,
    array_i: usize,
    length: usize,
    n_chunks: usize,
}

impl<'a> ChunkBoolIter<'a> {
    #[inline]
    fn set_indexes(&mut self) {
        set_indexes!(self)
    }

    #[inline]
    fn out_of_bounds(&self) -> bool {
        self.chunk_i >= self.n_chunks
    }
}

impl<'a> Iterator for ChunkBoolIter<'a> {
    type Item = Option<bool>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.out_of_bounds() {
            return None;
        }

        let current_data = unsafe { self.current_data.as_ref().unsafe_unwrap() };
        let current_array = unsafe { self.current_array.unsafe_unwrap() };

        debug_assert!(self.chunk_i < self.array_chunks.len());
        get_opt_value_and_set_indexes!(self, current_data, current_array)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }
}

impl<'a> ExactSizeIterator for ChunkBoolIter<'a> {
    fn len(&self) -> usize {
        self.length
    }
}

impl<'a> IntoIterator for &'a BooleanChunked {
    type Item = Option<bool>;
    type IntoIter = ChunkBoolIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let arrays = self.downcast_chunks();

        let arr = arrays.get(0).map(|v| *v);
        let data = arr.map(|arr| arr.data());
        let current_len = match arr {
            Some(arr) => arr.len(),
            None => 0,
        };

        ChunkBoolIter {
            array_chunks: arrays,
            current_data: data,
            current_array: arr,
            current_len,
            chunk_i: 0,
            array_i: 0,
            length: self.len(),
            n_chunks: self.chunks.len(),
        }
    }
}

fn get_iter_capacity<T, I: Iterator<Item = T>>(iter: &I) -> usize {
    match iter.size_hint() {
        (_lower, Some(upper)) => upper,
        (0, None) => 1024,
        (lower, None) => lower,
    }
}

impl<T> FromIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    fn from_iter<I: IntoIterator<Item = Option<T::Native>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = PrimitiveChunkedBuilder::new("", get_iter_capacity(&iter));

        for opt_val in iter {
            builder.append_option(opt_val).expect("could not append");
        }
        builder.finish()
    }
}

// Xob is only a wrapper needed for specialization
impl<T> FromIterator<T::Native> for Xob<ChunkedArray<T>>
where
    T: ArrowPrimitiveType,
{
    fn from_iter<I: IntoIterator<Item = T::Native>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = PrimitiveChunkedBuilder::new("", get_iter_capacity(&iter));

        for val in iter {
            builder.append_value(val).expect("could not append");
        }
        Xob::new(builder.finish())
    }
}

impl FromIterator<bool> for BooleanChunked {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = PrimitiveChunkedBuilder::new("", get_iter_capacity(&iter));

        for val in iter {
            builder.append_value(val).expect("could not append");
        }
        builder.finish()
    }
}

// FromIterator for Utf8Chunked variants.

impl<'a> FromIterator<&'a str> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = Utf8ChunkedBuilder::new("", get_iter_capacity(&iter));

        for val in iter {
            builder.append_value(val).expect("could not append");
        }
        builder.finish()
    }
}

impl<'a> FromIterator<&'a &'a str> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = &'a &'a str>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = Utf8ChunkedBuilder::new("", get_iter_capacity(&iter));

        for val in iter {
            builder.append_value(val).expect("could not append");
        }
        builder.finish()
    }
}

impl<'a> FromIterator<Option<&'a str>> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = Option<&'a str>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = Utf8ChunkedBuilder::new("", get_iter_capacity(&iter));

        for opt_val in iter {
            match opt_val {
                None => builder.append_null().expect("should not fail"),
                Some(val) => builder.append_value(val).expect("should not fail"),
            }
        }
        builder.finish()
    }
}

impl FromIterator<String> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = Utf8ChunkedBuilder::new("", get_iter_capacity(&iter));

        for val in iter {
            builder
                .append_value(val.as_str())
                .expect("could not append");
        }
        builder.finish()
    }
}

impl FromIterator<Option<String>> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = Option<String>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = Utf8ChunkedBuilder::new("", get_iter_capacity(&iter));

        for opt_val in iter {
            match opt_val {
                None => builder.append_null().expect("should not fail"),
                Some(val) => builder.append_value(val.as_str()).expect("should not fail"),
            }
        }
        builder.finish()
    }
}

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

    #[test]
    fn test_iter_numitersinglechunknullcheck() {
        let a = UInt32Chunked::new_from_opt_slice("a", &[Some(1), None, Some(3)]);
        let mut it = a.into_iter();

        // normal iterator
        assert_eq!(it.next(), Some(Some(1)));
        assert_eq!(it.next(), Some(None));
        assert_eq!(it.next(), Some(Some(3)));
        assert_eq!(it.next(), None);

        // reverse iterator
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(3)));
        assert_eq!(it.next_back(), Some(None));
        assert_eq!(it.next_back(), Some(Some(1)));
        assert_eq!(it.next_back(), None);

        // iterators should not cross
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(3)));
        assert_eq!(it.next(), Some(Some(1)));
        assert_eq!(it.next(), Some(None));
        // should stop here as we took this one from the back
        assert_eq!(it.next(), None);

        // do the same from the right side
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(1)));
        assert_eq!(it.next_back(), Some(Some(3)));
        assert_eq!(it.next_back(), Some(None));
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn test_iter_numitersinglechunk() {
        let a = UInt32Chunked::new_from_slice("a", &[1, 2, 3]);

        // normal iterator
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(1)));
        assert_eq!(it.next(), Some(Some(2)));
        assert_eq!(it.next(), Some(Some(3)));
        assert_eq!(it.next(), None);

        // reverse iterator
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(3)));
        assert_eq!(it.next_back(), Some(Some(2)));
        assert_eq!(it.next_back(), Some(Some(1)));
        assert_eq!(it.next_back(), None);

        // iterators should not cross
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(3)));
        assert_eq!(it.next(), Some(Some(1)));
        assert_eq!(it.next(), Some(Some(2)));
        // should stop here as we took this one from the back
        assert_eq!(it.next(), None);

        // do the same from the right side
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(1)));
        assert_eq!(it.next_back(), Some(Some(3)));
        assert_eq!(it.next_back(), Some(Some(2)));
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn test_iter_numitermanychunk() {
        let mut a = UInt32Chunked::new_from_slice("a", &[1, 2]);
        let a_b = UInt32Chunked::new_from_slice("", &[3]);
        a.append(&a_b);

        // normal iterator
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(1)));
        assert_eq!(it.next(), Some(Some(2)));
        assert_eq!(it.next(), Some(Some(3)));
        assert_eq!(it.next(), None);

        // reverse iterator
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(3)));
        assert_eq!(it.next_back(), Some(Some(2)));
        assert_eq!(it.next_back(), Some(Some(1)));
        assert_eq!(it.next_back(), None);

        // iterators should not cross
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(3)));
        assert_eq!(it.next(), Some(Some(1)));
        assert_eq!(it.next(), Some(Some(2)));
        // should stop here as we took this one from the back
        assert_eq!(it.next(), None);

        // do the same from the right side
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(1)));
        assert_eq!(it.next_back(), Some(Some(3)));
        assert_eq!(it.next_back(), Some(Some(2)));
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn test_iter_numitermanychunknullcheck() {
        let mut a = UInt32Chunked::new_from_opt_slice("a", &[Some(1), None]);
        let a_b = UInt32Chunked::new_from_opt_slice("", &[Some(3)]);
        a.append(&a_b);

        // normal iterator
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(1)));
        assert_eq!(it.next(), Some(None));
        assert_eq!(it.next(), Some(Some(3)));
        assert_eq!(it.next(), None);

        // reverse iterator
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(3)));
        assert_eq!(it.next_back(), Some(None));
        assert_eq!(it.next_back(), Some(Some(1)));
        assert_eq!(it.next_back(), None);

        // iterators should not cross
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(3)));
        assert_eq!(it.next(), Some(Some(1)));
        assert_eq!(it.next(), Some(None));
        // should stop here as we took this one from the back
        assert_eq!(it.next(), None);

        // do the same from the right side
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(1)));
        assert_eq!(it.next_back(), Some(Some(3)));
        assert_eq!(it.next_back(), Some(None));
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn test_iter_utf8itersinglechunknullcheck() {
        let a = Utf8Chunked::new_utf8_from_opt_slice("a", &[Some("a"), None, Some("c")]);

        // normal iterator
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some("a")));
        assert_eq!(it.next(), Some(None));
        assert_eq!(it.next(), Some(Some("c")));
        assert_eq!(it.next(), None);

        // reverse iterator
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some("c")));
        assert_eq!(it.next_back(), Some(None));
        assert_eq!(it.next_back(), Some(Some("a")));
        assert_eq!(it.next_back(), None);

        // iterators should not cross
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some("c")));
        assert_eq!(it.next(), Some(Some("a")));
        assert_eq!(it.next(), Some(None));
        // should stop here as we took this one from the back
        assert_eq!(it.next(), None);

        // do the same from the right side
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some("a")));
        assert_eq!(it.next_back(), Some(Some("c")));
        assert_eq!(it.next_back(), Some(None));
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn test_iter_utf8itersinglechunk() {
        let a = Utf8Chunked::new_utf8_from_slice("a", &["a", "b", "c"]);

        // normal iterator
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some("a")));
        assert_eq!(it.next(), Some(Some("b")));
        assert_eq!(it.next(), Some(Some("c")));
        assert_eq!(it.next(), None);

        // reverse iterator
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some("c")));
        assert_eq!(it.next_back(), Some(Some("b")));
        assert_eq!(it.next_back(), Some(Some("a")));
        assert_eq!(it.next_back(), None);

        // iterators should not cross
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some("c")));
        assert_eq!(it.next(), Some(Some("a")));
        assert_eq!(it.next(), Some(Some("b")));
        // should stop here as we took this one from the back
        assert_eq!(it.next(), None);

        // do the same from the right side
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some("a")));
        assert_eq!(it.next_back(), Some(Some("c")));
        assert_eq!(it.next_back(), Some(Some("b")));
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn test_iter_utf8itermanychunk() {
        let mut a = Utf8Chunked::new_utf8_from_slice("a", &["a", "b"]);
        let a_b = Utf8Chunked::new_utf8_from_slice("", &["c"]);
        a.append(&a_b);

        // normal iterator
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some("a")));
        assert_eq!(it.next(), Some(Some("b")));
        assert_eq!(it.next(), Some(Some("c")));
        assert_eq!(it.next(), None);

        // reverse iterator
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some("c")));
        assert_eq!(it.next_back(), Some(Some("b")));
        assert_eq!(it.next_back(), Some(Some("a")));
        assert_eq!(it.next_back(), None);

        // iterators should not cross
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some("c")));
        assert_eq!(it.next(), Some(Some("a")));
        assert_eq!(it.next(), Some(Some("b")));
        // should stop here as we took this one from the back
        assert_eq!(it.next(), None);

        // do the same from the right side
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some("a")));
        assert_eq!(it.next_back(), Some(Some("c")));
        assert_eq!(it.next_back(), Some(Some("b")));
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn test_iter_utf8itermanychunknullcheck() {
        let mut a = Utf8Chunked::new_utf8_from_opt_slice("a", &[Some("a"), None]);
        let a_b = Utf8Chunked::new_utf8_from_opt_slice("", &[Some("c")]);
        a.append(&a_b);

        // normal iterator
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some("a")));
        assert_eq!(it.next(), Some(None));
        assert_eq!(it.next(), Some(Some("c")));
        assert_eq!(it.next(), None);

        // reverse iterator
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some("c")));
        assert_eq!(it.next_back(), Some(None));
        assert_eq!(it.next_back(), Some(Some("a")));
        assert_eq!(it.next_back(), None);

        // iterators should not cross
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some("c")));
        assert_eq!(it.next(), Some(Some("a")));
        assert_eq!(it.next(), Some(None));
        // should stop here as we took this one from the back
        assert_eq!(it.next(), None);

        // do the same from the right side
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some("a")));
        assert_eq!(it.next_back(), Some(Some("c")));
        assert_eq!(it.next_back(), Some(None));
        assert_eq!(it.next_back(), None);
    }
}
