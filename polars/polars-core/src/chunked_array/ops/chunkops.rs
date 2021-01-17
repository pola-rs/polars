use crate::chunked_array::builder::get_list_builder;
#[cfg(feature = "object")]
use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::prelude::*;
#[cfg(feature = "object")]
use arrow::array::Array;
use arrow::compute::concat;
use itertools::Itertools;
#[cfg(feature = "object")]
use std::any::Any;
#[cfg(feature = "object")]
use std::fmt::Debug;

pub trait ChunkOps {
    /// Aggregate to contiguous memory.
    fn rechunk(&self) -> Result<Self>
    where
        Self: std::marker::Sized;
}

impl<T> ChunkOps for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn rechunk(&self) -> Result<Self> {
        if self.chunks().len() == 1 {
            Ok(self.clone())
        } else {
            let chunks = vec![concat(
                &self.chunks.iter().map(|a| &**a).collect_vec().as_slice(),
            )?];
            Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
        }
    }
}

impl ChunkOps for BooleanChunked {
    fn rechunk(&self) -> Result<Self> {
        if self.chunks().len() == 1 {
            Ok(self.clone())
        } else {
            let chunks = vec![concat(
                &self.chunks.iter().map(|a| &**a).collect_vec().as_slice(),
            )?];
            Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
        }
    }
}

impl ChunkOps for Utf8Chunked {
    fn rechunk(&self) -> Result<Self> {
        if self.chunks().len() == 1 {
            Ok(self.clone())
        } else {
            let chunks = vec![concat(
                &self.chunks.iter().map(|a| &**a).collect_vec().as_slice(),
            )?];
            Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
        }
    }
}

impl ChunkOps for CategoricalChunked {
    fn rechunk(&self) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        self.cast::<UInt32Type>()?.rechunk()?.cast()
    }
}

impl ChunkOps for ListChunked {
    fn rechunk(&self) -> Result<Self> {
        if self.chunks.len() == 1 {
            Ok(self.clone())
        } else {
            let mut builder = get_list_builder(&self.dtype(), self.len(), self.name());
            for v in self {
                builder.append_opt_series(v.as_ref())
            }
            Ok(builder.finish())
        }
    }
}

#[cfg(feature = "object")]
impl<T> ChunkOps for ObjectChunked<T>
where
    T: Any + Debug + Clone + Send + Sync + Default,
{
    fn rechunk(&self) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        if self.chunks.len() == 1 {
            Ok(self.clone())
        } else {
            let mut builder = ObjectChunkedBuilder::new(self.name(), self.len());
            let chunks = self.downcast_chunks();

            // todo! use iterators once implemented
            // no_null path
            if self.null_count() == 0 {
                for idx in 0..self.len() {
                    let (chunk_idx, idx) = self.index_to_chunked_index(idx);
                    let arr = unsafe { &**chunks.get_unchecked(chunk_idx) };
                    builder.append_value(arr.value(idx).clone())
                }
            } else {
                for idx in 0..self.len() {
                    let (chunk_idx, idx) = self.index_to_chunked_index(idx);
                    let arr = unsafe { &**chunks.get_unchecked(chunk_idx) };
                    if arr.is_valid(idx) {
                        builder.append_value(arr.value(idx).clone())
                    } else {
                        builder.append_null()
                    }
                }
            }
            Ok(builder.finish())
        }
    }
}
