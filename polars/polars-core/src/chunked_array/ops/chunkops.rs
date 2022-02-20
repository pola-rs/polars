#[cfg(feature = "object")]
use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::prelude::*;
use crate::utils::slice_offsets;
#[cfg(feature = "object")]
use arrow::array::Array;
use arrow::compute::concatenate;

#[inline]
fn slice(
    chunks: &[ArrayRef],
    offset: i64,
    slice_length: usize,
    own_length: usize,
) -> Vec<ArrayRef> {
    let mut new_chunks = Vec::with_capacity(1);
    let (raw_offset, slice_len) = slice_offsets(offset, slice_length, own_length);

    let mut remaining_length = slice_len;
    let mut remaining_offset = raw_offset;

    for chunk in chunks {
        let chunk_len = chunk.len();
        if remaining_offset > 0 && remaining_offset >= chunk_len {
            remaining_offset -= chunk_len;
            continue;
        }
        let take_len = if remaining_length + remaining_offset > chunk_len {
            chunk_len - remaining_offset
        } else {
            remaining_length
        };

        debug_assert!(remaining_offset + take_len <= chunk.len());
        unsafe {
            // Safety:
            // this function ensures the slices are in bounds
            new_chunks.push(chunk.slice_unchecked(remaining_offset, take_len).into());
        }
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
                self.chunks
                    .iter()
                    .map(|a| &**a)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .unwrap()
            .into()];
            ChunkedArray::from_chunks(self.name(), chunks)
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
                self.chunks
                    .iter()
                    .map(|a| &**a)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .unwrap()
            .into()];
            ChunkedArray::from_chunks(self.name(), chunks)
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
                self.chunks
                    .iter()
                    .map(|a| &**a)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .unwrap()
            .into()];
            ChunkedArray::from_chunks(self.name(), chunks)
        }
    }
    #[inline]
    fn slice(&self, offset: i64, length: usize) -> Self {
        self.copy_with_chunks(slice(&self.chunks, offset, length, self.len()))
    }
}

impl ChunkOps for ListChunked {
    fn rechunk(&self) -> Self {
        if self.chunks.len() == 1 {
            self.clone()
        } else {
            let chunks = vec![concatenate::concatenate(
                self.chunks
                    .iter()
                    .map(|a| &**a)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .unwrap()
            .into()];
            let mut ca = ListChunked::from_chunks(self.name(), chunks);
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
        let mut a = s.cast(&DataType::Categorical(None)).unwrap();

        a.append(&a.slice(0, 2)).unwrap();
        let a = a.rechunk();
        assert!(a.categorical().unwrap().get_rev_map().len() > 0);
    }
}
