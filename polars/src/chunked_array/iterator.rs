use crate::chunked_array::builder::{PrimitiveChunkedBuilder, Utf8ChunkedBuilder};
use crate::prelude::*;
use crate::utils::Xob;
use arrow::array::{Array, ArrayDataRef, BooleanArray, PrimitiveArray, StringArray};
use arrow::datatypes::ArrowPrimitiveType;
use std::iter::Copied;
use std::iter::FromIterator;
use std::slice::Iter;
use unsafe_unwrap::UnsafeUnwrap;

pub trait ExactSizeDoubleEndedIterator: ExactSizeIterator + DoubleEndedIterator {}

impl<'a, T: PolarsNumericType> ExactSizeDoubleEndedIterator for NumIterSingleChunk<'a, T> {}
impl<'a, T: PolarsNumericType> ExactSizeDoubleEndedIterator for NumIterSingleChunkNullCheck<'a, T> {}
impl<'a, T: PolarsNumericType> ExactSizeDoubleEndedIterator for NumIterManyChunk<'a, T> {}
impl<'a, T: PolarsNumericType> ExactSizeDoubleEndedIterator for NumIterManyChunkNullCheck<'a, T> {}

impl<'a, T> ExactSizeIterator for NumIterSingleChunkNullCheck<'a, T> where T: PolarsNumericType {}
impl<'a, T> ExactSizeIterator for NumIterSingleChunk<'a, T>
where
    T: PolarsNumericType,
    T::Native: Copy,
{
}
impl<'a, T> ExactSizeIterator for NumIterManyChunkNullCheck<'a, T> where T: PolarsNumericType {}
impl<'a, T> ExactSizeIterator for NumIterManyChunk<'a, T> where T: PolarsNumericType {}

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

/// Many chunks with null checks
pub struct NumIterManyChunkNullCheck<'a, T>
where
    T: PolarsNumericType,
{
    ca: &'a ChunkedArray<T>,
    chunks: Vec<&'a PrimitiveArray<T>>,
    current_iter: Copied<Iter<'a, T::Native>>,
    // data is faster for null checks
    current_data: ArrayDataRef,
    // if current chunk has null values
    current_null: bool,
    array_i: usize,
    chunk_i: usize,
    current_len: usize,
}

impl<'a, T> NumIterManyChunkNullCheck<'a, T>
where
    T: PolarsNumericType,
{
    fn new(ca: &'a ChunkedArray<T>) -> Self {
        let chunks = ca.downcast_chunks();
        let arr = chunks[0];
        let current_len = arr.len();
        let current_iter = arr.value_slice(0, current_len).iter().copied();
        NumIterManyChunkNullCheck {
            ca,
            chunks,
            current_iter,
            current_data: arr.data(),
            current_null: arr.null_count() != 0,
            current_len,
            array_i: 0,
            chunk_i: 0,
        }
    }
}

impl<'a, T> Iterator for NumIterManyChunkNullCheck<'a, T>
where
    T: PolarsNumericType,
{
    type Item = Option<T::Native>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.array_i == self.current_len {
            // go to the next array in the chunks
            self.chunk_i += 1;
            self.array_i = 0;

            if self.chunk_i < self.chunks.len() {
                let current_chunk = unsafe { self.chunks.get_unchecked(self.chunk_i) };
                // not passed last chunk
                self.current_len = current_chunk.len();
                self.current_iter = current_chunk
                    .value_slice(0, self.current_len)
                    .iter()
                    .copied();
                self.current_null = current_chunk.null_count() != 0;
                self.current_data = current_chunk.data()
            } else {
                // end of iterator
                return None;
            }
        }
        self.array_i += 1;
        let ret = self.current_iter.next();
        if self.current_null {
            if self.current_data.is_null(self.array_i) {
                return Some(None);
            }
        }

        ret.map(Some)
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
        unimplemented!()
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

pub struct ChunkStringIter<'a> {
    array_chunks: Vec<&'a StringArray>,
    current_data: Option<ArrayDataRef>,
    current_array: Option<&'a StringArray>,
    current_len: usize,
    chunk_i: usize,
    array_i: usize,
    length: usize,
    n_chunks: usize,
}

impl<'a> ChunkStringIter<'a> {
    #[inline]
    fn set_indexes(&mut self) {
        set_indexes!(self)
    }

    #[inline]
    fn out_of_bounds(&self) -> bool {
        self.chunk_i >= self.n_chunks
    }
}

impl<'a> Iterator for ChunkStringIter<'a> {
    type Item = Option<&'a str>;

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

impl<'a> ExactSizeIterator for ChunkStringIter<'a> {
    fn len(&self) -> usize {
        self.length
    }
}

impl<'a> IntoIterator for &'a Utf8Chunked {
    type Item = Option<&'a str>;
    type IntoIter = ChunkStringIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let arrays = self.downcast_chunks();

        let arr = arrays.get(0).map(|v| *v);
        let data = arr.map(|arr| arr.data());
        let current_len = match arr {
            Some(arr) => arr.len(),
            None => 0,
        };

        ChunkStringIter {
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

/// Specialized Iterator for ChunkedArray<PolarsNumericType>
pub struct ChunkNumIter<'a, T>
where
    T: PolarsNumericType,
{
    array_chunks: Vec<&'a PrimitiveArray<T>>,
    current_data: Option<ArrayDataRef>,
    current_array: Option<&'a PrimitiveArray<T>>,
    current_len: usize,
    chunk_i: usize,
    array_i: usize,
    length: usize,
    n_chunks: usize,
    opt_iter: Option<Iter<'a, T::Native>>,
}

impl<'a, T> Iterator for ChunkNumIter<'a, T>
where
    T: PolarsNumericType,
{
    type Item = Option<T::Native>;

    fn next(&mut self) -> Option<Self::Item> {
        // Faster path for chunks without null values
        if self.opt_iter.is_some() {
            let iter = unsafe { &mut self.opt_iter.as_mut().unsafe_unwrap() };

            // first get return value
            let ret = iter.next().map(|&v| Some(v));

            // And maybe set indexes. Only when there are multiple chunks
            // This is stateful and may change the iterator in self.opt_slice
            if self.n_chunks > 1 {
                self.set_indexes()
            }
            return ret;
        }

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

impl<'a, T> ExactSizeIterator for ChunkNumIter<'a, T>
where
    T: PolarsNumericType,
{
    fn len(&self) -> usize {
        self.length
    }
}

impl<'a, T> ChunkNumIter<'a, T>
where
    T: PolarsNumericType,
{
    #[inline]
    fn set_indexes(&mut self) {
        self.array_i += 1;
        if self.array_i >= self.current_len {
            // go to next array in the chunks
            self.array_i = 0;
            self.chunk_i += 1;

            if self.chunk_i < self.array_chunks.len() {
                // not yet at last chunk
                let arr = unsafe { *self.array_chunks.get_unchecked(self.chunk_i) };
                self.current_data = Some(arr.data());
                self.current_array = Some(arr);
                self.current_len = arr.len();

                if arr.null_count() == 0 {
                    self.opt_iter = Some(arr.value_slice(0, self.current_len).iter())
                } else {
                    self.opt_iter = None
                }
            }
        }
    }

    #[inline]
    fn out_of_bounds(&self) -> bool {
        self.chunk_i >= self.n_chunks
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
}
