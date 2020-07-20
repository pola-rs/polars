use crate::chunked_array::builder::{PrimitiveChunkedBuilder, Utf8ChunkedBuilder};
use crate::prelude::*;
use crate::utils::Xob;
use arrow::array::{Array, ArrayDataRef, BooleanArray, PrimitiveArray, StringArray};
use arrow::datatypes::ArrowPrimitiveType;
use std::iter::Copied;
use std::iter::FromIterator;
use std::slice::Iter;
use unsafe_unwrap::UnsafeUnwrap;

/// Single chunk with null values
pub struct NumIterSingleChunkNullCheck<'a, T>
where
    T: PolarsNumericType,
{
    arr: &'a PrimitiveArray<T>,
    idx: usize,
}

impl<'a, T> Iterator for NumIterSingleChunkNullCheck<'a, T>
where
    T: PolarsNumericType,
{
    type Item = Option<T::Native>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.arr.len() {
            None
        } else {
            self.idx += 1;
            if self.arr.is_null(self.idx - 1) {
                Some(None)
            } else {
                Some(Some(self.arr.value(self.idx - 1)))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.arr.len();
        (len, Some(len))
    }
}

impl<'a, T> ExactSizeIterator for NumIterSingleChunkNullCheck<'a, T> where T: PolarsNumericType {}

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

impl<'a, T> ExactSizeIterator for NumIterSingleChunk<'a, T>
where
    T: PolarsNumericType,
    T::Native: Copy,
{
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

impl<'a, T> ExactSizeIterator for NumIterManyChunkNullCheck<'a, T> where T: PolarsNumericType {}
/// Many chunks no null checks
pub struct NumIterManyChunk<'a, T>
where
    T: PolarsNumericType,
{
    ca: &'a ChunkedArray<T>,
    chunks: Vec<&'a PrimitiveArray<T>>,
    current_iter: Copied<Iter<'a, T::Native>>,
    array_i: usize,
    chunk_i: usize,
    current_len: usize,
}

impl<'a, T> NumIterManyChunk<'a, T>
where
    T: PolarsNumericType,
{
    fn new(ca: &'a ChunkedArray<T>) -> Self {
        let chunks = ca.downcast_chunks();
        let current_len = chunks[0].len();
        NumIterManyChunk {
            ca,
            current_len,
            current_iter: chunks[0].value_slice(0, current_len).iter().copied(),
            chunks,
            array_i: 0,
            chunk_i: 0,
        }
    }
}

impl<'a, T> Iterator for NumIterManyChunk<'a, T>
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
            } else {
                // end of iterator
                return None;
            }
        }
        self.array_i += 1;
        self.current_iter.next().map(Some)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.ca.len();
        (len, Some(len))
    }
}

impl<'a, T> ExactSizeIterator for NumIterManyChunk<'a, T> where T: PolarsNumericType {}
impl<'a, T> IntoIterator for &'a ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Item = Option<T::Native>;
    type IntoIter = Box<dyn ExactSizeIterator<Item = Option<T::Native>> + 'a>;

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
                    1 => Box::new(NumIterSingleChunkNullCheck {
                        arr: chunks[0],
                        idx: 0,
                    }),
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

pub struct ChunkStringIter<'a> {
    array_chunks: Vec<&'a StringArray>,
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
        self.array_i += 1;
        if self.array_i >= self.current_len {
            // go to next array in the chunks
            self.array_i = 0;
            self.chunk_i += 1;

            if self.chunk_i < self.array_chunks.len() {
                // not yet at last chunk
                let arr = unsafe { *self.array_chunks.get_unchecked(self.chunk_i) };
                self.current_array = Some(arr);
                self.current_len = arr.len();
            }
        }
    }

    #[inline]
    fn out_of_bounds(&self) -> bool {
        self.chunk_i >= self.n_chunks
    }
}

impl<'a> Iterator for ChunkStringIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.out_of_bounds() {
            return None;
        }

        let current_array = unsafe { self.current_array.unsafe_unwrap() };

        debug_assert!(self.chunk_i < self.array_chunks.len());

        let v = current_array.value(self.array_i);
        self.set_indexes();
        Some(v)
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
    type Item = &'a str;
    type IntoIter = ChunkStringIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        let arrays = self.downcast_chunks();

        let arr = arrays.get(0).map(|v| *v);
        let current_len = match arr {
            Some(arr) => arr.len(),
            None => 0,
        };

        ChunkStringIter {
            array_chunks: arrays,
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
        let ret;

        if current_data.is_null(self.array_i) {
            ret = Some(None)
        } else {
            let v = current_array.value(self.array_i);
            ret = Some(Some(v));
        }
        self.set_indexes();
        ret
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
        let ret;

        if current_data.is_null(self.array_i) {
            ret = Some(None)
        } else {
            let v = current_array.value(self.array_i);
            ret = Some(Some(v));
        }
        self.set_indexes();
        ret
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
}
