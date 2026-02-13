use polars_core::prelude::*;

pub fn reinterpret(s: &Series, dtype: DataType) -> PolarsResult<Series> {
    Ok(match (s.dtype(), dtype) {
        (DataType::UInt8, DataType::UInt8) => s.clone(),
        (DataType::UInt16, DataType::UInt16) => s.clone(),
        (DataType::UInt32, DataType::UInt32) => s.clone(),
        (DataType::UInt64, DataType::UInt64) => s.clone(),
        (DataType::UInt128, DataType::UInt128) => s.clone(),

        (DataType::Int8, DataType::Int8) => s.clone(),
        (DataType::Int16, DataType::Int16) => s.clone(),
        (DataType::Int32, DataType::Int32) => s.clone(),
        (DataType::Int64, DataType::Int64) => s.clone(),
        (DataType::Int128, DataType::Int128) => s.clone(),

        (DataType::Float16, DataType::Float16) => s.clone(),
        (DataType::Float32, DataType::Float32) => s.clone(),
        (DataType::Float64, DataType::Float64) => s.clone(),

        #[cfg(all(feature = "dtype-u8", feature = "dtype-i8"))]
        (DataType::UInt8, DataType::Int8) => s.u8().unwrap().reinterpret_signed(),
        #[cfg(all(feature = "dtype-u16", feature = "dtype-i16"))]
        (DataType::UInt16, DataType::Int16) => s.u16().unwrap().reinterpret_signed(),
        (DataType::UInt32, DataType::Int32) => s.u32().unwrap().reinterpret_signed(),
        (DataType::UInt64, DataType::Int64) => s.u64().unwrap().reinterpret_signed(),
        #[cfg(feature = "dtype-i128")]
        (DataType::UInt128, DataType::Int128) => s.u128().unwrap().reinterpret_signed(),

        #[cfg(feature = "dtype-f16")]
        (DataType::UInt16, DataType::Float16) => s.u16().unwrap().reinterpret_float(),
        (DataType::UInt32, DataType::Float32) => s.u32().unwrap().reinterpret_float(),
        (DataType::UInt64, DataType::Float64) => s.u64().unwrap().reinterpret_float(),

        #[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
        (DataType::Int8, DataType::UInt8) => s.i8().unwrap().reinterpret_unsigned(),
        #[cfg(all(feature = "dtype-i16", feature = "dtype-u16"))]
        (DataType::Int16, DataType::UInt16) => s.i16().unwrap().reinterpret_unsigned(),
        (DataType::Int32, DataType::UInt32) => s.i32().unwrap().reinterpret_unsigned(),
        (DataType::Int64, DataType::UInt64) => s.i64().unwrap().reinterpret_unsigned(),
        #[cfg(feature = "dtype-i128")]
        (DataType::Int128, DataType::UInt128) => s.i128().unwrap().reinterpret_unsigned(),

        #[cfg(feature = "dtype-f16")]
        (DataType::Int16, DataType::Float16) => s.i16().unwrap().reinterpret_float(),
        (DataType::Int32, DataType::Float32) => s.i32().unwrap().reinterpret_float(),
        (DataType::Int64, DataType::Float64) => s.i64().unwrap().reinterpret_float(),

        #[cfg(feature = "dtype-f16")]
        (DataType::Float16, DataType::UInt16) => s.f16().unwrap().reinterpret_unsigned(),
        (DataType::Float32, DataType::UInt32) => s.f32().unwrap().reinterpret_unsigned(),
        (DataType::Float64, DataType::UInt64) => s.f64().unwrap().reinterpret_unsigned(),

        #[cfg(feature = "dtype-f16")]
        (DataType::Float16, DataType::Int16) => s.f16().unwrap().reinterpret_signed(),
        (DataType::Float32, DataType::Int32) => s.f32().unwrap().reinterpret_signed(),
        (DataType::Float64, DataType::Int64) => s.f64().unwrap().reinterpret_signed(),

        _ => polars_bail!(
            ComputeError:
            "reinterpret is only allowed for numeric types of the same size (for example: 32-bits integer to 32-bits float), use cast otherwise"
        ),
    })
}
