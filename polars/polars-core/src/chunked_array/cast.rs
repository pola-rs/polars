//! Implementations of the ChunkCast Trait.
use crate::chunked_array::builder::CategoricalChunkedBuilder;
use crate::chunked_array::kernels::{cast_numeric_from_dtype, transmute_array_from_dtype};
use crate::prelude::*;
use crate::use_string_cache;
use arrow::compute::cast;
use num::{NumCast, ToPrimitive};

fn cast_ca<N, T>(ca: &ChunkedArray<T>) -> Result<ChunkedArray<N>>
where
    N: PolarsDataType,
    T: PolarsDataType,
{
    if N::get_dtype() == T::get_dtype() {
        // convince the compiler that N and T are the same type
        return unsafe {
            let ca = std::mem::transmute(ca.clone());
            Ok(ca)
        };
    };

    // // only i32 can be cast to Date32
    // if let DataType::Date32 = N::get_dtype() {
    //     if T::get_dtype() != ArrowDataType::Int32 {
    //         let casted_i32 = cast_ca::<Int32Type, _>(ca)?;
    //         return cast_ca(&casted_i32);
    //     }
    // }
    let chunks = ca
        .chunks
        .iter()
        .map(|arr| cast(arr, &N::get_dtype().to_arrow()))
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

impl ChunkCast for CategoricalChunked {
    fn cast<N>(&self) -> Result<ChunkedArray<N>>
    where
        N: PolarsDataType,
    {
        match N::get_dtype() {
            DataType::Utf8 => {
                let mapping = &**self.categorical_map.as_ref().expect("should be set");

                let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len(), self.len() * 5);

                let f = |idx: u32| mapping.get(&idx).unwrap();

                if self.null_count() == 0 {
                    self.into_no_null_iter()
                        .for_each(|idx| builder.append_value(f(idx)));
                } else {
                    self.into_iter().for_each(|opt_idx| {
                        builder.append_option(opt_idx.map(f));
                    });
                }

                let ca = builder.finish();
                let ca = unsafe { std::mem::transmute(ca) };
                Ok(ca)
            }
            DataType::UInt32 => {
                let ca: ChunkedArray<N> = unsafe { std::mem::transmute(self.clone()) };
                Ok(ca)
            }
            _ => cast_ca(self),
        }
    }
}

impl<T> ChunkCast for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast + ToPrimitive,
{
    fn cast<N>(&self) -> Result<ChunkedArray<N>>
    where
        N: PolarsDataType,
    {
        use DataType::*;
        let ca = match (T::get_dtype(), N::get_dtype()) {
            (UInt32, Categorical) => {
                let ca: ChunkedArray<N> = unsafe { std::mem::transmute(self.clone()) };
                return Ok(ca);
            }
            // underlying type: i64
            (Duration(_), UInt64) => cast_from_dtype!(self, cast_numeric_from_dtype, UInt64),
            // the underlying datatype is i64 so we transmute array
            (Duration(_), Int64) => unsafe {
                cast_from_dtype!(self, transmute_array_from_dtype, Int64)
            },
            (Duration(_), Float32) | (Date32, Float32) | (Date64, Float32) => {
                cast_from_dtype!(self, cast_numeric_from_dtype, Float32)
            }
            (Duration(_), Float64) | (Date32, Float64) | (Date64, Float64) => {
                cast_from_dtype!(self, cast_numeric_from_dtype, Float64)
            }
            _ => cast_ca(self),
        };
        ca.map(|mut ca| {
            ca.field = Arc::new(Field::new(ca.name(), N::get_dtype()));
            ca
        })
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

impl ChunkCast for Utf8Chunked {
    fn cast<N>(&self) -> Result<ChunkedArray<N>>
    where
        N: PolarsDataType,
    {
        match N::get_dtype() {
            DataType::Categorical => {
                let mut builder = CategoricalChunkedBuilder::new(self.name(), self.len());

                if use_string_cache() || self.null_count() != 0 {
                    builder.append_values(self.into_iter());
                } else if self.null_count() == 0 {
                    self.into_no_null_iter()
                        .for_each(|v| builder.append_value(v));
                } else {
                    self.into_iter()
                        .for_each(|opt_v| builder.append_option(opt_v));
                }

                let ca = builder.finish();
                let ca = unsafe { std::mem::transmute(ca) };
                Ok(ca)
            }
            _ => cast_ca(self),
        }
    }
}

impl_chunkcast!(BooleanChunked);
impl_chunkcast!(ListChunked);
