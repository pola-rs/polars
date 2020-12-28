//! Implementations of the ChunkCast Trait.
use crate::chunked_array::kernels::{cast_numeric_from_dtype, transmute_array_from_dtype};
use crate::prelude::*;
use arrow::compute;
use num::NumCast;

fn cast_ca<N, T>(ca: &ChunkedArray<T>) -> Result<ChunkedArray<N>>
where
    N: PolarsDataType,
    T: PolarsDataType,
{
    if N::get_data_type() == T::get_data_type() {
        return unsafe {
            let ca = std::mem::transmute(ca.clone());
            Ok(ca)
        };
    };

    // only i32 can be cast to Date32
    if let ArrowDataType::Date32(DateUnit::Day) = N::get_data_type() {
        if T::get_data_type() != ArrowDataType::Int32 {
            let casted_i32 = cast_ca::<Int32Type, _>(ca)?;
            return cast_ca(&casted_i32);
        }
    }
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
        match T::get_data_type() {
            // Duration cast is not implemented in Arrow
            ArrowDataType::Duration(_) => {
                // underlying type: i64
                match N::get_data_type() {
                    ArrowDataType::UInt64 => {
                        cast_from_dtype!(self, cast_numeric_from_dtype, UInt64)
                    }
                    // the underlying datatype is i64 so we transmute array
                    ArrowDataType::Int64 => unsafe {
                        cast_from_dtype!(self, transmute_array_from_dtype, Int64)
                    },
                    ArrowDataType::Float32 => {
                        cast_from_dtype!(self, cast_numeric_from_dtype, Float32)
                    }
                    ArrowDataType::Float64 => {
                        cast_from_dtype!(self, cast_numeric_from_dtype, Float64)
                    }
                    _ => cast_ca(self),
                }
            }
            ArrowDataType::Date32(_) => {
                match N::get_data_type() {
                    // underlying type: i32
                    ArrowDataType::Int32 => cast_ca(self),
                    ArrowDataType::Int64 => {
                        cast_from_dtype!(self, cast_numeric_from_dtype, Int64)
                    }
                    ArrowDataType::Float32 => {
                        cast_from_dtype!(self, cast_numeric_from_dtype, Float32)
                    }
                    ArrowDataType::Float64 => {
                        cast_from_dtype!(self, cast_numeric_from_dtype, Float64)
                    }
                    ArrowDataType::Utf8 => {
                        let ca: ChunkedArray<N> = unsafe {
                            std::mem::transmute(self.cast::<Date32Type>().unwrap().str_fmt("%F"))
                        };
                        Ok(ca)
                    }
                    _ => cast_ca(self),
                }
            }
            ArrowDataType::Date64(_) => {
                match N::get_data_type() {
                    // underlying type: i32
                    ArrowDataType::Int32 => {
                        cast_from_dtype!(self, cast_numeric_from_dtype, Int32)
                    }
                    ArrowDataType::Int64 => cast_ca(self),
                    ArrowDataType::Float32 => {
                        cast_from_dtype!(self, cast_numeric_from_dtype, Float32)
                    }
                    ArrowDataType::Float64 => {
                        cast_from_dtype!(self, cast_numeric_from_dtype, Float64)
                    }
                    ArrowDataType::Utf8 => {
                        let ca: ChunkedArray<N> = unsafe {
                            std::mem::transmute(self.cast::<Date64Type>().unwrap().str_fmt("%+"))
                        };
                        Ok(ca)
                    }
                    _ => cast_ca(self),
                }
            }
            _ => cast_ca(self),
        }
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
