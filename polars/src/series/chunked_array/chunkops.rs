use crate::prelude::*;
use arrow::array::{PrimitiveBuilder, StringBuilder};
use std::sync::Arc;

pub trait ChunkOps {
    fn rechunk(&mut self, chunk_id: &[usize]);
    fn optional_rechunk<A>(&self, rhs: &ChunkedArray<A>) -> Result<Option<Self>>
    where
        Self: std::marker::Sized;
}

macro_rules! optional_rechunk {
    ($self:tt, $rhs:tt) => {
        if $self.chunk_id != $rhs.chunk_id {
            // we can rechunk ourselves to match
            if $rhs.chunks.len() == 1 {
                let mut new = $self.clone();
                new.rechunk(&$rhs.chunk_id);
                Ok(Some(new))
            } else {
                Err(PolarsError::ChunkMisMatch)
            }
        } else {
            Ok(None)
        }
    };
}

impl<T> ChunkOps for ChunkedArray<T>
where
    T: PolarNumericType,
{
    fn rechunk(&mut self, chunk_id: &[usize]) {
        if self.chunks.len() > 1 {
            let mut builder = PrimitiveBuilder::<T>::new(self.len());
            self.into_iter().for_each(|val| {
                builder.append_option(val).expect("Could not append value");
            });
            self.chunks = vec![Arc::new(builder.finish())];
            self.set_chunk_id()
        }
    }

    fn optional_rechunk<A>(&self, rhs: &ChunkedArray<A>) -> Result<Option<Self>> {
        optional_rechunk!(self, rhs)
    }
}

impl ChunkOps for BooleanChunked {
    fn rechunk(&mut self, chunk_id: &[usize]) {
        if self.chunks.len() > 1 {
            let mut builder = PrimitiveBuilder::<BooleanType>::new(self.len());
            self.into_iter()
                .for_each(|val| builder.append_option(val).expect("Could not append value"));
            self.chunks = vec![Arc::new(builder.finish())];
            self.set_chunk_id()
        }
    }

    fn optional_rechunk<A>(&self, rhs: &ChunkedArray<A>) -> Result<Option<Self>> {
        optional_rechunk!(self, rhs)
    }
}

impl ChunkOps for Utf8Chunked {
    fn rechunk(&mut self, chunk_id: &[usize]) {
        if self.chunks.len() > 1 {
            let mut builder = StringBuilder::new(self.len());
            self.into_iter().for_each(|opt_val| match opt_val {
                Some(val) => builder.append_value(val).expect("Could not append value"),
                None => builder.append_null().expect("append null"),
            });

            self.chunks = vec![Arc::new(builder.finish())];
            self.set_chunk_id()
        }
    }

    fn optional_rechunk<A>(&self, rhs: &ChunkedArray<A>) -> Result<Option<Self>> {
        optional_rechunk!(self, rhs)
    }
}
