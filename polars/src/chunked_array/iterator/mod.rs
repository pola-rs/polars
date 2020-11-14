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
impl<'a, T> ExactSizeIterator for NumIterSingleChunkNullCheck<'a, T> where T: PolarsNumericType {}
impl<'a, T> ExactSizeIterator for NumIterSingleChunk<'a, T>
where
    T: PolarsNumericType,
    T::Native: Copy,
{
}
impl<'a, T> ExactSizeIterator for NumIterManyChunkNullCheck<'a, T> where T: PolarsNumericType {}
impl<'a, T> ExactSizeIterator for NumIterManyChunk<'a, T> where T: PolarsNumericType {}
impl<'a> ExactSizeIterator for Utf8IterCont<'a> {}
impl<'a> ExactSizeIterator for Utf8IterSingleChunk<'a> {}
impl<'a> ExactSizeIterator for Utf8IterSingleChunkNullCheck<'a> {}
impl<'a> ExactSizeIterator for Utf8IterManyChunk<'a> {}
impl<'a> ExactSizeIterator for Utf8IterManyChunkNullCheck<'a> {}

/// Trait for ChunkedArrays that don't have null values.
pub trait IntoNoNullIterator {
    type Item;
    type IntoIter: Iterator<Item = Self::Item>;

    fn into_no_null_iter(self) -> Self::IntoIter;
}

impl<'a, T> IntoNoNullIterator for &'a ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = T::Native;
    type IntoIter = Box<dyn Iterator<Item = Self::Item> + 'a>;

    fn into_no_null_iter(self) -> Self::IntoIter {
        match self.chunks.len() {
            1 => Box::new(
                self.downcast_chunks()[0]
                    .value_slice(0, self.len())
                    .iter()
                    .copied(),
            ),
            _ => Box::new(NumIterManyChunk::new(self)),
        }
    }
}

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
/// Both used as iterator with null checks and without. We later map Some on it for the iter
/// with null checks
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

pub enum NumericChunkIterDispatch<'a, T>
where
    T: PolarsNumericType,
{
    SingleChunk(NumIterSingleChunk<'a, T>),
    SingleChunkNullCheck(NumIterSingleChunkNullCheck<'a, T>),
    ManyChunk(NumIterManyChunk<'a, T>),
    ManyChunkNullCheck(NumIterManyChunkNullCheck<'a, T>),
}

impl<'a, T> Iterator for NumericChunkIterDispatch<'a, T>
where
    T: PolarsNumericType,
{
    type Item = Option<T::Native>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            NumericChunkIterDispatch::SingleChunk(a) => a.next(),
            NumericChunkIterDispatch::SingleChunkNullCheck(a) => a.next(),
            NumericChunkIterDispatch::ManyChunk(a) => a.next().map(Some),
            NumericChunkIterDispatch::ManyChunkNullCheck(a) => a.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            NumericChunkIterDispatch::SingleChunk(a) => a.size_hint(),
            NumericChunkIterDispatch::SingleChunkNullCheck(a) => a.size_hint(),
            NumericChunkIterDispatch::ManyChunk(a) => a.size_hint(),
            NumericChunkIterDispatch::ManyChunkNullCheck(a) => a.size_hint(),
        }
    }
}

impl<'a, T> DoubleEndedIterator for NumericChunkIterDispatch<'a, T>
where
    T: PolarsNumericType,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            NumericChunkIterDispatch::SingleChunk(a) => a.next_back(),
            NumericChunkIterDispatch::SingleChunkNullCheck(a) => a.next_back(),
            NumericChunkIterDispatch::ManyChunk(a) => a.next_back().map(Some),
            NumericChunkIterDispatch::ManyChunkNullCheck(a) => a.next_back(),
        }
    }
}

impl<'a, T> ExactSizeIterator for NumericChunkIterDispatch<'a, T> where T: PolarsNumericType {}

impl<'a, T> IntoIterator for &'a ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = Option<T::Native>;
    type IntoIter = NumericChunkIterDispatch<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        match self.cont_slice() {
            Ok(slice) => {
                // Compile could not infer T.
                let a: NumIterSingleChunk<'_, T> = NumIterSingleChunk {
                    iter: slice.iter().copied(),
                };
                NumericChunkIterDispatch::SingleChunk(a)
            }
            Err(_) => {
                let chunks = self.downcast_chunks();
                match chunks.len() {
                    1 => {
                        let arr = chunks[0];
                        let len = arr.len();

                        NumericChunkIterDispatch::SingleChunkNullCheck(
                            NumIterSingleChunkNullCheck {
                                arr,
                                idx: 0,
                                back_idx: len,
                            },
                        )
                    }
                    _ => {
                        if self.null_count() == 0 {
                            NumericChunkIterDispatch::ManyChunk(NumIterManyChunk::new(self))
                        } else {
                            NumericChunkIterDispatch::ManyChunkNullCheck(
                                NumIterManyChunkNullCheck::new(self),
                            )
                        }
                    }
                }
            }
        }
    }
}

/// No null checks and dont return Option<T> but T directly. So this struct is not return by the
/// IntoIterator trait
pub struct Utf8IterCont<'a> {
    current_array: &'a StringArray,
    idx_left: usize,
    idx_right: usize,
}

impl<'a> Utf8IterCont<'a> {
    fn new(ca: &'a Utf8Chunked) -> Self {
        let chunks = ca.downcast_chunks();
        let current_array = chunks[0];
        let idx_left = 0;
        let idx_right = current_array.len();

        Utf8IterCont {
            current_array,
            idx_left,
            idx_right,
        }
    }
}

impl<'a> Iterator for Utf8IterCont<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        // end of iterator or meet reversed iterator in the middle
        if self.idx_left == self.idx_right {
            return None;
        }

        let v = self.current_array.value(self.idx_left);
        self.idx_left += 1;
        Some(v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.current_array.len();
        (len, Some(len))
    }
}

impl<'a> DoubleEndedIterator for Utf8IterCont<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        // end of iterator or meet reversed iterator in the middle
        if self.idx_left == self.idx_right {
            return None;
        }
        self.idx_right -= 1;
        Some(self.current_array.value(self.idx_right))
    }
}

impl<'a> IntoNoNullIterator for &'a Utf8Chunked {
    type Item = &'a str;
    type IntoIter = Utf8IterCont<'a>;

    fn into_no_null_iter(self) -> Self::IntoIter {
        Utf8IterCont::new(self)
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

/// Many chunks with nulls
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

pub enum Utf8ChunkIterDispatch<'a> {
    SingleChunk(Utf8IterSingleChunk<'a>),
    SingleChunkNullCheck(Utf8IterSingleChunkNullCheck<'a>),
    ManyChunk(Utf8IterManyChunk<'a>),
    ManyChunkNullCheck(Utf8IterManyChunkNullCheck<'a>),
}

impl<'a> Iterator for Utf8ChunkIterDispatch<'a> {
    type Item = Option<&'a str>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Utf8ChunkIterDispatch::SingleChunk(a) => a.next(),
            Utf8ChunkIterDispatch::SingleChunkNullCheck(a) => a.next(),
            Utf8ChunkIterDispatch::ManyChunk(a) => a.next(),
            Utf8ChunkIterDispatch::ManyChunkNullCheck(a) => a.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Utf8ChunkIterDispatch::SingleChunk(a) => a.size_hint(),
            Utf8ChunkIterDispatch::SingleChunkNullCheck(a) => a.size_hint(),
            Utf8ChunkIterDispatch::ManyChunk(a) => a.size_hint(),
            Utf8ChunkIterDispatch::ManyChunkNullCheck(a) => a.size_hint(),
        }
    }
}

impl<'a> DoubleEndedIterator for Utf8ChunkIterDispatch<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Utf8ChunkIterDispatch::SingleChunk(a) => a.next_back(),
            Utf8ChunkIterDispatch::SingleChunkNullCheck(a) => a.next_back(),
            Utf8ChunkIterDispatch::ManyChunk(a) => a.next_back(),
            Utf8ChunkIterDispatch::ManyChunkNullCheck(a) => a.next_back(),
        }
    }
}

impl<'a> ExactSizeIterator for Utf8ChunkIterDispatch<'a> {}

impl<'a> IntoIterator for &'a Utf8Chunked {
    type Item = Option<&'a str>;
    type IntoIter = Utf8ChunkIterDispatch<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let chunks = self.downcast_chunks();
        match chunks.len() {
            1 => {
                if self.null_count() == 0 {
                    Utf8ChunkIterDispatch::SingleChunk(Utf8IterSingleChunk::new(self))
                } else {
                    Utf8ChunkIterDispatch::SingleChunkNullCheck(Utf8IterSingleChunkNullCheck::new(
                        self,
                    ))
                }
            }
            _ => {
                if self.null_count() == 0 {
                    Utf8ChunkIterDispatch::ManyChunk(Utf8IterManyChunk::new(self))
                } else {
                    Utf8ChunkIterDispatch::ManyChunkNullCheck(Utf8IterManyChunkNullCheck::new(self))
                }
            }
        }
    }
}

macro_rules! impl_iterator_traits {
    ($ca_type:ident, // ChunkedArray type
     $arrow_array:ident, // Arrow array Type
     $no_null_iter_struct:ident, // Name of Iterator in case of no null checks and return type is T instead of Option<T>
     $single_chunk_ident:ident, // Name of Iterator in case of single chunk
     $single_chunk_null_ident:ident, // Name of Iterator in case of single chunk and nulls
     $many_chunk_ident:ident, // Name of Iterator in case of many chunks
     $many_chunk_null_ident:ident, // Name of Iterator in case of many chunks and null
     $chunkdispatch:ident, // Name of Dispatch struct
     $iter_item:ty, // Item returned by Iterator e.g. Option<bool>
     $iter_item_no_null:ty, // Iterm return by Iterator that doesn't do null checks. e.g. bool
     $return_function: ident, // function that is called upon returning from the iterator with the inner type
                              // So in case of both Option<T> and T function is called with T
                              // Fn(method_name: &str, type: T) -> ?
     ) => {
        impl<'a> ExactSizeIterator for $single_chunk_ident<'a> {}
        impl<'a> ExactSizeIterator for $single_chunk_null_ident<'a> {}
        impl<'a> ExactSizeIterator for $many_chunk_ident<'a> {}
        impl<'a> ExactSizeIterator for $many_chunk_null_ident<'a> {}

        /// No null checks and dont return Option<T> but T directly. So this struct is not return by the
        /// IntoIterator trait
        pub struct $no_null_iter_struct<'a> {
            current_array: &'a $arrow_array,
            idx_left: usize,
            idx_right: usize,
        }

        impl<'a> $no_null_iter_struct<'a> {
            fn new(ca: &'a $ca_type) -> Self {
                let chunks = ca.downcast_chunks();
                let current_array = chunks[0];
                let idx_left = 0;
                let idx_right = current_array.len();

                $no_null_iter_struct {
                    current_array,
                    idx_left,
                    idx_right,
                }
            }
        }

        impl<'a> Iterator for $no_null_iter_struct<'a> {
            type Item = $iter_item_no_null;

            fn next(&mut self) -> Option<Self::Item> {
                // end of iterator or meet reversed iterator in the middle
                if self.idx_left == self.idx_right {
                    return None;
                }

                let v = self.current_array.value(self.idx_left);
                self.idx_left += 1;
                Some($return_function("next", v))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.current_array.len();
                (len, Some(len))
            }
        }

        impl<'a> IntoNoNullIterator for &'a $ca_type {
            type Item = $iter_item_no_null;
            type IntoIter = $no_null_iter_struct<'a>;

            fn into_no_null_iter(self) -> Self::IntoIter {
                $no_null_iter_struct::new(self)
            }
        }

        /// No null checks
        pub struct $single_chunk_ident<'a> {
            current_array: &'a $arrow_array,
            idx_left: usize,
            idx_right: usize,
        }

        impl<'a> $single_chunk_ident<'a> {
            fn new(ca: &'a $ca_type) -> Self {
                let chunks = ca.downcast_chunks();
                let current_array = chunks[0];
                let idx_left = 0;
                let idx_right = current_array.len();

                $single_chunk_ident {
                    current_array,
                    idx_left,
                    idx_right,
                }
            }
        }

        impl<'a> Iterator for $single_chunk_ident<'a> {
            type Item = $iter_item;

            fn next(&mut self) -> Option<Self::Item> {
                // end of iterator or meet reversed iterator in the middle
                if self.idx_left == self.idx_right {
                    return None;
                }

                let v = self.current_array.value(self.idx_left);
                self.idx_left += 1;
                Some(Some($return_function("next", v)))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.current_array.len();
                (len, Some(len))
            }
        }

        impl<'a> DoubleEndedIterator for $single_chunk_ident<'a> {
            fn next_back(&mut self) -> Option<Self::Item> {
                // end of iterator or meet reversed iterator in the middle
                if self.idx_left == self.idx_right {
                    return None;
                }
                self.idx_right -= 1;
                let v = self.current_array.value(self.idx_right);
                Some(Some($return_function("next_back", v)))
            }
        }

        pub struct $single_chunk_null_ident<'a> {
            current_data: ArrayDataRef,
            current_array: &'a $arrow_array,
            idx_left: usize,
            idx_right: usize,
        }

        impl<'a> $single_chunk_null_ident<'a> {
            fn new(ca: &'a $ca_type) -> Self {
                let chunks = ca.downcast_chunks();
                let current_array = chunks[0];
                let current_data = current_array.data();
                let idx_left = 0;
                let idx_right = current_array.len();

                $single_chunk_null_ident {
                    current_data,
                    current_array,
                    idx_left,
                    idx_right,
                }
            }
        }

        impl<'a> Iterator for $single_chunk_null_ident<'a> {
            type Item = $iter_item;

            fn next(&mut self) -> Option<Self::Item> {
                // end of iterator or meet reversed iterator in the middle
                if self.idx_left == self.idx_right {
                    return None;
                }
                let ret;
                if self.current_data.is_null(self.idx_left) {
                    ret = None
                } else {
                    let v = self.current_array.value(self.idx_left);
                    ret = Some($return_function("next", v))
                }
                self.idx_left += 1;
                Some(ret)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.current_array.len();
                (len, Some(len))
            }
        }

        impl<'a> DoubleEndedIterator for $single_chunk_null_ident<'a> {
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
                    Some(Some($return_function("next_back", v)))
                }
            }
        }

        /// Many chunks no nulls
        pub struct $many_chunk_ident<'a> {
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

        impl<'a> $many_chunk_ident<'a> {
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

                $many_chunk_ident {
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

        impl<'a> Iterator for $many_chunk_ident<'a> {
            type Item = $iter_item;

            fn next(&mut self) -> Option<Self::Item> {
                // end of iterator or meet reversed iterator in the middle
                if self.idx_left == self.idx_right {
                    return None;
                }

                // return value
                let v = self.current_array_left.value(self.current_array_idx_left);
                let ret = $return_function("next", v);

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

        impl<'a> DoubleEndedIterator for $many_chunk_ident<'a> {
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
                Some(Some($return_function("next_back", ret)))
            }
        }

        /// Many chunks no nulls
        pub struct $many_chunk_null_ident<'a> {
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

        impl<'a> $many_chunk_null_ident<'a> {
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

                $many_chunk_null_ident {
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

        impl<'a> Iterator for $many_chunk_null_ident<'a> {
            type Item = $iter_item;

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
                    ret = Some($return_function("next", v));
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

        impl<'a> DoubleEndedIterator for $many_chunk_null_ident<'a> {
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
                    Some(Some($return_function("next_back", v)))
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

        pub enum $chunkdispatch<'a> {
            SingleChunk($single_chunk_ident<'a>),
            SingleChunkNullCheck($single_chunk_null_ident<'a>),
            ManyChunk($many_chunk_ident<'a>),
            ManyChunkNullCheck($many_chunk_null_ident<'a>),
        }

        impl<'a> Iterator for $chunkdispatch<'a> {
            type Item = $iter_item;

            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    $chunkdispatch::SingleChunk(a) => a.next(),
                    $chunkdispatch::SingleChunkNullCheck(a) => a.next(),
                    $chunkdispatch::ManyChunk(a) => a.next(),
                    $chunkdispatch::ManyChunkNullCheck(a) => a.next(),
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                match self {
                    $chunkdispatch::SingleChunk(a) => a.size_hint(),
                    $chunkdispatch::SingleChunkNullCheck(a) => a.size_hint(),
                    $chunkdispatch::ManyChunk(a) => a.size_hint(),
                    $chunkdispatch::ManyChunkNullCheck(a) => a.size_hint(),
                }
            }
        }

        impl<'a> DoubleEndedIterator for $chunkdispatch<'a> {
            fn next_back(&mut self) -> Option<Self::Item> {
                match self {
                    $chunkdispatch::SingleChunk(a) => a.next_back(),
                    $chunkdispatch::SingleChunkNullCheck(a) => a.next_back(),
                    $chunkdispatch::ManyChunk(a) => a.next_back(),
                    $chunkdispatch::ManyChunkNullCheck(a) => a.next_back(),
                }
            }
        }

        impl<'a> ExactSizeIterator for $chunkdispatch<'a> {}

        impl<'a> IntoIterator for &'a $ca_type {
            type Item = $iter_item;
            type IntoIter = $chunkdispatch<'a>;

            fn into_iter(self) -> Self::IntoIter {
                let chunks = self.downcast_chunks();
                match chunks.len() {
                    1 => {
                        if self.null_count() == 0 {
                            $chunkdispatch::SingleChunk($single_chunk_ident::new(self))
                        } else {
                            $chunkdispatch::SingleChunkNullCheck($single_chunk_null_ident::new(
                                self,
                            ))
                        }
                    }
                    _ => {
                        if self.null_count() == 0 {
                            $chunkdispatch::ManyChunk($many_chunk_ident::new(self))
                        } else {
                            $chunkdispatch::ManyChunkNullCheck($many_chunk_null_ident::new(self))
                        }
                    }
                }
            }
        }
    };
}

// Used for macro. method_name is ignored
fn return_from_bool_iter(_method_name: &str, v: bool) -> bool {
    v
}

impl_iterator_traits!(
    BooleanChunked,
    BooleanArray,
    BooleanIterCont,
    BooleanIterSingleChunk,
    BooleanIterSingleChunkNullCheck,
    BooleanIterManyChunk,
    BooleanIterManyChunkNullCheck,
    BooleanIterDispatch,
    Option<bool>,
    bool,
    return_from_bool_iter,
);

// used for macro
fn return_from_list_iter(method_name: &str, v: ArrayRef) -> Series {
    (method_name, v).into()
}

impl_iterator_traits!(
    ListChunked,
    ListArray,
    ListIterCont,
    ListIterSingleChunk,
    ListIterSingleChunkNullCheck,
    ListIterManyChunk,
    ListIterManyChunkNullCheck,
    ListIterDispatch,
    Option<Series>,
    Series,
    return_from_list_iter,
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
        let a = Utf8Chunked::new_from_opt_slice("a", &[Some("a"), None, Some("c")]);

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
        let a = Utf8Chunked::new_from_slice("a", &["a", "b", "c"]);

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
        let mut a = Utf8Chunked::new_from_slice("a", &["a", "b"]);
        let a_b = Utf8Chunked::new_from_slice("", &["c"]);
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
        let mut a = Utf8Chunked::new_from_opt_slice("a", &[Some("a"), None]);
        let a_b = Utf8Chunked::new_from_opt_slice("", &[Some("c")]);
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

    #[test]
    fn test_iter_boolitersinglechunknullcheck() {
        let a = BooleanChunked::new_from_opt_slice("", &[Some(true), None, Some(false)]);

        // normal iterator
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(true)));
        assert_eq!(it.next(), Some(None));
        assert_eq!(it.next(), Some(Some(false)));
        assert_eq!(it.next(), None);

        // reverse iterator
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(false)));
        assert_eq!(it.next_back(), Some(None));
        assert_eq!(it.next_back(), Some(Some(true)));
        assert_eq!(it.next_back(), None);

        // iterators should not cross
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(false)));
        assert_eq!(it.next(), Some(Some(true)));
        assert_eq!(it.next(), Some(None));
        // should stop here as we took this one from the back
        assert_eq!(it.next(), None);

        // do the same from the right side
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(true)));
        assert_eq!(it.next_back(), Some(Some(false)));
        assert_eq!(it.next_back(), Some(None));
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn test_iter_boolitersinglechunk() {
        let a = BooleanChunked::new_from_slice("", &[true, true, false]);

        // normal iterator
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(true)));
        assert_eq!(it.next(), Some(Some(true)));
        assert_eq!(it.next(), Some(Some(false)));
        assert_eq!(it.next(), None);

        // reverse iterator
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(false)));
        assert_eq!(it.next_back(), Some(Some(true)));
        assert_eq!(it.next_back(), Some(Some(true)));
        assert_eq!(it.next_back(), None);

        // iterators should not cross
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(false)));
        assert_eq!(it.next(), Some(Some(true)));
        assert_eq!(it.next(), Some(Some(true)));
        // should stop here as we took this one from the back
        assert_eq!(it.next(), None);

        // do the same from the right side
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(true)));
        assert_eq!(it.next_back(), Some(Some(false)));
        assert_eq!(it.next_back(), Some(Some(true)));
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn test_iter_boolitermanychunk() {
        let mut a = BooleanChunked::new_from_slice("", &[true, true]);
        let a_b = BooleanChunked::new_from_slice("", &[false]);
        a.append(&a_b);

        // normal iterator
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(true)));
        assert_eq!(it.next(), Some(Some(true)));
        assert_eq!(it.next(), Some(Some(false)));
        assert_eq!(it.next(), None);

        // reverse iterator
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(false)));
        assert_eq!(it.next_back(), Some(Some(true)));
        assert_eq!(it.next_back(), Some(Some(true)));
        assert_eq!(it.next_back(), None);

        // iterators should not cross
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(false)));
        assert_eq!(it.next(), Some(Some(true)));
        assert_eq!(it.next(), Some(Some(true)));
        // should stop here as we took this one from the back
        assert_eq!(it.next(), None);

        // do the same from the right side
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(true)));
        assert_eq!(it.next_back(), Some(Some(false)));
        assert_eq!(it.next_back(), Some(Some(true)));
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn test_iter_boolitermanychunknullcheck() {
        let mut a = BooleanChunked::new_from_opt_slice("a", &[Some(true), None]);
        let a_b = BooleanChunked::new_from_opt_slice("", &[Some(false)]);
        a.append(&a_b);

        // normal iterator
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(true)));
        assert_eq!(it.next(), Some(None));
        assert_eq!(it.next(), Some(Some(false)));
        assert_eq!(it.next(), None);

        // reverse iterator
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(false)));
        assert_eq!(it.next_back(), Some(None));
        assert_eq!(it.next_back(), Some(Some(true)));
        assert_eq!(it.next_back(), None);

        // iterators should not cross
        let mut it = a.into_iter();
        assert_eq!(it.next_back(), Some(Some(false)));
        assert_eq!(it.next(), Some(Some(true)));
        assert_eq!(it.next(), Some(None));
        // should stop here as we took this one from the back
        assert_eq!(it.next(), None);

        // do the same from the right side
        let mut it = a.into_iter();
        assert_eq!(it.next(), Some(Some(true)));
        assert_eq!(it.next_back(), Some(Some(false)));
        assert_eq!(it.next_back(), Some(None));
        assert_eq!(it.next_back(), None);
    }
}
