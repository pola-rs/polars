use crate::chunked_array::builder::get_list_builder;
#[cfg(feature = "object")]
use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::prelude::*;
use arrow::array::{Array, ArrayRef, PrimitiveBuilder, StringBuilder};
use arrow::compute::concat;
#[cfg(feature = "object")]
use std::any::Any;
#[cfg(feature = "object")]
use std::fmt::Debug;
use std::sync::Arc;

pub trait ChunkOps {
    /// Aggregate to chunk id.
    /// A chunk id is a vector of the chunk lengths.
    fn rechunk(&self, chunk_lengths: Option<&[usize]>) -> Result<Self>
    where
        Self: std::marker::Sized;
    /// Only rechunk if lhs and rhs don't match
    fn optional_rechunk<A>(&self, rhs: &ChunkedArray<A>) -> Result<Option<Self>>
    where
        Self: std::marker::Sized;
}

macro_rules! optional_rechunk {
    ($self:tt, $rhs:tt) => {
        if $self.chunk_id != $rhs.chunk_id {
            // we can rechunk ourselves to match
            $self.rechunk(Some(&$rhs.chunk_id)).map(Some)
        } else {
            Ok(None)
        }
    };
}

#[inline]
fn mimic_chunks<T>(arr: &ArrayRef, chunk_lengths: &[usize], name: &str) -> ChunkedArray<T>
where
    T: PolarsDataType,
    ChunkedArray<T>: ChunkOps,
{
    let mut chunks = Vec::with_capacity(chunk_lengths.len());
    let mut offset = 0;
    for chunk_length in chunk_lengths {
        chunks.push(arr.slice(offset, *chunk_length));
        offset += *chunk_length
    }
    ChunkedArray::new_from_chunks(name, chunks)
}

impl<T> ChunkOps for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn rechunk(&self, chunk_lengths: Option<&[usize]>) -> Result<Self> {
        // we aggregate to 1 or chunk_id
        match (self.chunks.len(), chunk_lengths.map(|v| v.len())) {
            // No rechunking needed.
            (1, Some(1)) | (1, None) => Ok(self.clone()),
            // use arrows concat logic
            (_, Some(1)) => {
                let chunks = vec![concat(&self.chunks)?];
                Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
            }
            // Left contains a single chunk. We can cheaply mimic right as arrow slices are zero copy
            (1, Some(_)) => Ok(mimic_chunks(
                &self.chunks[0],
                chunk_lengths.unwrap(),
                self.name(),
            )),
            // Left will be aggregated to match right
            (_, Some(_)) | (_, None) => {
                let default = &[self.len()];
                let chunk_id = chunk_lengths.unwrap_or(default);
                let mut iter = self.into_iter();
                let mut chunks: Vec<Arc<dyn Array>> = Vec::with_capacity(chunk_id.len());

                for &chunk_length in chunk_id {
                    let mut builder = PrimitiveBuilder::<T>::new(chunk_length);

                    for _ in 0..chunk_length {
                        builder.append_option(
                            iter.next()
                                .expect("the first option is the iterator bounds"),
                        )?;
                    }
                    chunks.push(Arc::new(builder.finish()))
                }
                Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
            }
        }
    }

    fn optional_rechunk<A>(&self, rhs: &ChunkedArray<A>) -> Result<Option<Self>> {
        optional_rechunk!(self, rhs)
    }
}

impl ChunkOps for BooleanChunked {
    fn rechunk(&self, chunk_lengths: Option<&[usize]>) -> Result<Self> {
        match (self.chunks.len(), chunk_lengths.map(|v| v.len())) {
            (1, Some(1)) | (1, None) => Ok(self.clone()),
            (_, Some(1)) => {
                let chunks = vec![concat(&self.chunks)?];
                Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
            }
            (1, Some(_)) => Ok(mimic_chunks(
                &self.chunks[0],
                chunk_lengths.unwrap(),
                self.name(),
            )),
            (_, Some(_)) | (_, None) => {
                let default = &[self.len()];
                let chunk_id = chunk_lengths.unwrap_or(default);

                let mut iter = self.into_iter();
                let mut chunks: Vec<Arc<dyn Array>> = Vec::with_capacity(chunk_id.len());

                for &chunk_length in chunk_id {
                    let mut builder = PrimitiveBuilder::<BooleanType>::new(chunk_length);

                    for _ in 0..chunk_length {
                        builder.append_option(
                            iter.next()
                                .expect("the first option is the iterator bounds"),
                        )?;
                    }
                    chunks.push(Arc::new(builder.finish()))
                }
                Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
            }
        }
    }

    fn optional_rechunk<A>(&self, rhs: &ChunkedArray<A>) -> Result<Option<Self>> {
        optional_rechunk!(self, rhs)
    }
}

impl ChunkOps for Utf8Chunked {
    fn rechunk(&self, chunk_lengths: Option<&[usize]>) -> Result<Self> {
        match (self.chunks.len(), chunk_lengths.map(|v| v.len())) {
            (1, Some(1)) | (1, None) => Ok(self.clone()),
            (_, Some(1)) => {
                let chunks = vec![concat(&self.chunks)?];
                Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
            }
            (1, Some(_)) => Ok(mimic_chunks(
                &self.chunks[0],
                chunk_lengths.unwrap(),
                self.name(),
            )),
            (_, Some(_)) | (_, None) => {
                let default = &[self.len()];
                let chunk_id = chunk_lengths.unwrap_or(default);
                let mut iter = self.into_iter();
                let mut chunks: Vec<Arc<dyn Array>> = Vec::with_capacity(chunk_id.len());

                for &chunk_length in chunk_id {
                    let mut builder = StringBuilder::new(chunk_length);

                    for _ in 0..chunk_length {
                        let opt_val = iter.next().expect("first option is iterator bounds");
                        match opt_val {
                            None => builder.append_null().expect("should not fail"),
                            Some(val) => builder.append_value(val).expect("should not fail"),
                        }
                    }
                    chunks.push(Arc::new(builder.finish()))
                }
                Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
            }
        }
    }

    fn optional_rechunk<A>(&self, rhs: &ChunkedArray<A>) -> Result<Option<Self>> {
        optional_rechunk!(self, rhs)
    }
}
impl ChunkOps for ListChunked {
    fn rechunk(&self, chunk_lengths: Option<&[usize]>) -> Result<Self> {
        match (self.chunks.len(), chunk_lengths.map(|v| v.len())) {
            (1, Some(1)) | (1, None) => Ok(self.clone()),
            (1, Some(_)) => Ok(mimic_chunks(
                &self.chunks[0],
                chunk_lengths.unwrap(),
                self.name(),
            )),
            (_, Some(_)) | (_, None) => {
                let default = &[self.len()];
                let chunk_id = chunk_lengths.unwrap_or(default);
                let mut iter = self.into_iter();
                let mut chunks: Vec<Arc<dyn Array>> = Vec::with_capacity(chunk_id.len());

                for &chunk_length in chunk_id {
                    let mut builder = get_list_builder(self.dtype(), chunk_length, self.name());
                    while let Some(v) = iter.next() {
                        builder.append_opt_series(v.as_ref())
                    }
                    let list = builder.finish();
                    // cheap clone of Arc
                    chunks.push(list.chunks[0].clone())
                }
                Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
            }
        }
    }

    fn optional_rechunk<A>(&self, rhs: &ChunkedArray<A>) -> Result<Option<Self>> {
        optional_rechunk!(self, rhs)
    }
}

#[cfg(feature = "object")]
impl<T> ChunkOps for ObjectChunked<T>
where
    T: Any + Debug + Clone + Send + Sync + Default,
{
    fn rechunk(&self, chunk_lengths: Option<&[usize]>) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        match (self.chunks.len(), chunk_lengths.map(|v| v.len())) {
            (1, Some(1)) | (1, None) => Ok(self.clone()),
            (1, Some(_)) => Ok(mimic_chunks(
                &self.chunks[0],
                chunk_lengths.unwrap(),
                self.name(),
            )),
            (_, None) => {
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
            _ => Err(PolarsError::Other(
                "rechunk of ObjectChunked still needs to be implemented".into(),
            )),
        }
    }

    fn optional_rechunk<A>(&self, _rhs: &ChunkedArray<A>) -> Result<Option<Self>>
    where
        Self: std::marker::Sized,
    {
        todo!()
    }
}
