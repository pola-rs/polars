//! Implementations of the ChunkCast Trait.
use crate::prelude::*;
use arrow::compute;

fn cast_ca<N, T>(ca: &ChunkedArray<T>) -> Result<ChunkedArray<N>>
where
    N: PolarsDataType,
    T: PolarsDataType,
{
    match N::get_data_type() {
        // only i32 can be cast to Date32
        ArrowDataType::Date32(DateUnit::Day) => {
            if T::get_data_type() != ArrowDataType::Int32 {
                let casted_i32 = cast_ca::<Int32Type, _>(ca)?;
                return cast_ca(&casted_i32);
            }
        }
        _ => (),
    };
    let chunks = ca
        .chunks
        .iter()
        .map(|arr| compute::cast(arr, &N::get_data_type()))
        .collect::<arrow::error::Result<Vec<_>>>()?;

    Ok(ChunkedArray::<N>::new_from_chunks(ca.field.name(), chunks))
}

impl<T> ChunkCast for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn cast<N>(&self) -> Result<ChunkedArray<N>>
    where
        N: PolarsDataType,
    {
        cast_ca(self)
    }
}

macro_rules! impl_chunkcast {
    ($ca_type:ident) => {
        impl ChunkCast for $ca_type {
            fn cast<N>(&self) -> Result<ChunkedArray<N>>
            where
                N: PolarsDataType,
            {
                cast_ca(self)
            }
        }
    };
}

impl_chunkcast!(Utf8Chunked);
impl_chunkcast!(BooleanChunked);
impl_chunkcast!(ListChunked);
