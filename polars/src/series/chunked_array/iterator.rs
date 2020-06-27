use crate::prelude::*;
use crate::series::chunked_array::builder::{PrimitiveChunkedBuilder, Utf8ChunkedBuilder};
use arrow::array::{Array, ArrayDataRef, BooleanArray, PrimitiveArray, StringArray};
use arrow::datatypes::ArrowPrimitiveType;
use std::iter::FromIterator;
use std::slice::Iter;
use unsafe_unwrap::UnsafeUnwrap;

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
    T: PolarNumericType,
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
    T: PolarNumericType,
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
}

impl<'a, T> ChunkNumIter<'a, T>
where
    T: PolarNumericType,
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

impl<'a, T> IntoIterator for &'a ChunkedArray<T>
where
    T: PolarNumericType,
{
    type Item = Option<T::Native>;
    type IntoIter = ChunkNumIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        let arrays = self.downcast_chunks();

        let arr = arrays.get(0).map(|v| *v);
        let data = arr.map(|arr| arr.data());

        let (current_len, null_count) = match arr {
            Some(arr) => (arr.len(), arr.null_count()),
            None => (0, 0),
        };

        let mut opt_slice = None;

        if null_count == 0 && current_len > 0 {
            opt_slice = arr.map(|arr| arr.value_slice(0, current_len).iter())
        }

        ChunkNumIter {
            array_chunks: arrays,
            current_data: data,
            current_array: arr,
            current_len,
            chunk_i: 0,
            array_i: 0,
            length: self.len(),
            n_chunks: self.chunks.len(),
            opt_iter: opt_slice,
        }
    }
}

impl<T> FromIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    fn from_iter<I: IntoIterator<Item = Option<T::Native>>>(iter: I) -> Self {
        let mut builder = PrimitiveChunkedBuilder::new("", 1024);

        for opt_val in iter {
            builder.append_option(opt_val).expect("could not append");
        }

        builder.finish()
    }
}

impl<'a> FromIterator<&'a str> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        let mut builder = Utf8ChunkedBuilder::new("", 1024);

        for val in iter {
            builder.append_value(val).expect("could not append");
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
