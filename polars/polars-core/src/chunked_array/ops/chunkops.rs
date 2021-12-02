#[cfg(feature = "object")]
use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::prelude::*;
use crate::utils::slice_offsets;
#[cfg(feature = "object")]
use arrow::array::Array;
use arrow::compute::concatenate;
use itertools::Itertools;
#[cfg(feature = "dtype-categorical")]
use std::ops::Deref;

#[inline]
fn slice(
    chunks: &[ArrayRef],
    offset: i64,
    slice_length: usize,
    own_length: usize,
) -> Vec<ArrayRef> {
    let (raw_offset, slice_len) = slice_offsets(offset, slice_length, own_length);

    let mut remaining_length = slice_len;
    let mut remaining_offset = raw_offset;
    let mut new_chunks = Vec::with_capacity(1);

    for chunk in chunks {
        let chunk_len = chunk.len();
        if remaining_offset > 0 && remaining_offset >= chunk_len {
            remaining_offset -= chunk_len;
            continue;
        }
        let take_len;
        if remaining_length + remaining_offset > chunk_len {
            take_len = chunk_len - remaining_offset;
        } else {
            take_len = remaining_length;
        }

        new_chunks.push(chunk.slice(remaining_offset, take_len).into());
        remaining_length -= take_len;
        remaining_offset = 0;
        if remaining_length == 0 {
            break;
        }
    }
    new_chunks
}

impl<T> ChunkOps for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn rechunk(&self) -> Self {
        if self.chunks().len() == 1 {
            self.clone()
        } else {
            let chunks = vec![concatenate::concatenate(
                self.chunks.iter().map(|a| &**a).collect_vec().as_slice(),
            )
            .unwrap()
            .into()];
            ChunkedArray::new_from_chunks(self.name(), chunks)
        }
    }
    #[inline]
    fn slice(&self, offset: i64, length: usize) -> Self {
        self.copy_with_chunks(slice(&self.chunks, offset, length, self.len()))
    }
}

impl ChunkOps for BooleanChunked {
    fn rechunk(&self) -> Self {
        if self.chunks().len() == 1 {
            self.clone()
        } else {
            let chunks = vec![concatenate::concatenate(
                self.chunks.iter().map(|a| &**a).collect_vec().as_slice(),
            )
            .unwrap()
            .into()];
            ChunkedArray::new_from_chunks(self.name(), chunks)
        }
    }
    #[inline]
    fn slice(&self, offset: i64, length: usize) -> Self {
        self.copy_with_chunks(slice(&self.chunks, offset, length, self.len()))
    }
}

impl ChunkOps for Utf8Chunked {
    fn rechunk(&self) -> Self {
        if self.chunks().len() == 1 {
            self.clone()
        } else {
            let chunks = vec![concatenate::concatenate(
                self.chunks.iter().map(|a| &**a).collect_vec().as_slice(),
            )
            .unwrap()
            .into()];
            ChunkedArray::new_from_chunks(self.name(), chunks)
        }
    }
    #[inline]
    fn slice(&self, offset: i64, length: usize) -> Self {
        self.copy_with_chunks(slice(&self.chunks, offset, length, self.len()))
    }
}

#[cfg(feature = "dtype-categorical")]
impl ChunkOps for CategoricalChunked {
    fn rechunk(&self) -> Self
    where
        Self: std::marker::Sized,
    {
        let mut out: CategoricalChunked = self.deref().rechunk().into();
        let cat_map = self.categorical_map.clone();
        out.categorical_map = cat_map;
        out
    }
    #[inline]
    fn slice(&self, offset: i64, length: usize) -> Self {
        let mut out = self.copy_with_chunks(slice(&self.chunks, offset, length, self.len()));
        out.set_fast_unique(false);
        out
    }
}

impl ChunkOps for ListChunked {
    fn rechunk(&self) -> Self {
        if self.chunks.len() == 1 {
            self.clone()
        } else {
            let chunks = vec![concatenate::concatenate(
                self.chunks.iter().map(|a| &**a).collect_vec().as_slice(),
            )
            .unwrap()
            .into()];
            let mut ca = ListChunked::new_from_chunks(self.name(), chunks);
            if self.can_fast_explode() {
                ca.set_fast_explode()
            }
            ca
        }
    }
    #[inline]
    fn slice(&self, offset: i64, length: usize) -> Self {
        self.copy_with_chunks(slice(&self.chunks, offset, length, self.len()))
    }
}

#[cfg(feature = "object")]
impl<T> ChunkOps for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn rechunk(&self) -> Self
    where
        Self: std::marker::Sized,
    {
        if self.chunks.len() == 1 {
            self.clone()
        } else {
            let mut builder = ObjectChunkedBuilder::new(self.name(), self.len());
            let chunks = self.downcast_iter();

            // todo! use iterators once implemented
            // no_null path
            if !self.has_validity() {
                for arr in chunks {
                    for idx in 0..arr.len() {
                        builder.append_value(arr.value(idx).clone())
                    }
                }
            } else {
                for arr in chunks {
                    for idx in 0..arr.len() {
                        if arr.is_valid(idx) {
                            builder.append_value(arr.value(idx).clone())
                        } else {
                            builder.append_null()
                        }
                    }
                }
            }
            builder.finish()
        }
    }
    #[inline]
    fn slice(&self, offset: i64, length: usize) -> Self {
        self.copy_with_chunks(slice(&self.chunks, offset, length, self.len()))
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_categorical_map_after_rechunk() {
        let s = Series::new("", &["foo", "bar", "spam"]);
        let mut a = s.cast(&DataType::Categorical).unwrap();

        a.append(&a.slice(0, 2)).unwrap();
        a.rechunk();
        assert!(a.categorical().unwrap().categorical_map.is_some());
    }
}
