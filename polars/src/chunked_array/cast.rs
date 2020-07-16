use crate::prelude::*;
use arrow::compute;

pub trait ChunkCast {
    fn cast<N>(&self) -> Result<ChunkedArray<N>>
    where
        ChunkedArray<N>: ChunkOps,
        N: PolarsDataType;
}
impl<T> ChunkCast for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn cast<N>(&self) -> Result<ChunkedArray<N>>
    where
        ChunkedArray<N>: ChunkOps,
        N: PolarsDataType,
    {
        let chunks = self
            .chunks
            .iter()
            .map(|arr| compute::cast(arr, &N::get_data_type()))
            .collect::<arrow::error::Result<Vec<_>>>()?;

        Ok(ChunkedArray::<N>::new_from_chunks(
            self.field.name(),
            chunks,
        ))
    }
}

macro_rules! impl_chunkcast {
    ($ca_type:ident) => {
        impl ChunkCast for $ca_type {
            fn cast<N>(&self) -> Result<ChunkedArray<N>>
            where
                ChunkedArray<N>: ChunkOps,
                N: PolarsDataType,
            {
                let chunks = self
                    .chunks
                    .iter()
                    .map(|arr| compute::cast(arr, &N::get_data_type()))
                    .collect::<arrow::error::Result<Vec<_>>>()?;

                Ok(ChunkedArray::<N>::new_from_chunks(
                    self.field.name(),
                    chunks,
                ))
            }
        }
    };
}

impl_chunkcast!(Utf8Chunked);
impl_chunkcast!(BooleanChunked);
