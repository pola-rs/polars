//! Implementations of the ChunkCast Trait.
use crate::chunked_array::kernels::cast_numeric_from_dtype;
use crate::prelude::*;
use arrow::compute;
use num::NumCast;

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

    Ok(ChunkedArray::new_from_chunks(ca.field.name(), chunks))
}

macro_rules! cast_from_dtype {
    ($self: expr, $kernel:expr, $dtype: ident) => {{
        let chunks = $self
            .downcast_chunks()
            .into_iter()
            .map(|arr| $kernel(arr, ArrowDataType::$dtype))
            .collect();

        Ok(ChunkedArray::new_from_chunks($self.field.name(), chunks))
    }};
}

impl<T> ChunkCast for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
{
    fn cast<N>(&self) -> Result<ChunkedArray<N>>
    where
        N: PolarsDataType,
    {
        // Duration cast is not implemented in Arrow
        match T::get_data_type() {
            // underlying type: i64
            ArrowDataType::Duration(_) => match N::get_data_type() {
                ArrowDataType::UInt64 => {
                    // todo! check if this is safe as underlying type is i64 and not u64
                    return cast_from_dtype!(self, cast_numeric_from_dtype, UInt64);
                }
                ArrowDataType::Int64 => {
                    return cast_from_dtype!(self, cast_numeric_from_dtype, Int64)
                }
                _ => (),
            },
            _ => {}
        }

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
