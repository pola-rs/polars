use polars_ops::chunked_array::peaks::{peak_max as pmax, peak_min as pmin};

use super::*;

pub(super) fn peak_min(s: &Series) -> PolarsResult<Series> {
    let s = match s.dtype() {
        DataType::Boolean => unimplemented!(), // TODO
        DataType::UInt8 => pmin(s.u8()?).into_series(),
        DataType::UInt16 => pmin(s.u16()?).into_series(),
        DataType::UInt32 => pmin(s.u32()?).into_series(),
        DataType::UInt64 => pmin(s.u64()?).into_series(),
        DataType::Int8 => pmin(s.i8()?).into_series(),
        DataType::Int16 => pmin(s.i16()?).into_series(),
        DataType::Int32 => pmin(s.i32()?).into_series(),
        DataType::Int64 => pmin(s.i64()?).into_series(),
        DataType::Float32 => pmin(s.f32()?).into_series(),
        DataType::Float64 => pmin(s.f64()?).into_series(),
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => pmin(s.decimal()?).into_series(),
        #[cfg(feature = "dtype-date")]
        DataType::Date => pmin(s.date()?).into_series(),
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(_, _) => pmin(s.datetime()?).into_series(),
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(_) => pmin(s.duration()?).into_series(),
        #[cfg(feature = "dtype-time")]
        DataType::Time => pmin(s.time()?).into_series(),
        _ => unimplemented!(),
    };
    Ok(s)
}

pub(super) fn peak_max(s: &Series) -> PolarsResult<Series> {
    let s = match s.dtype() {
        DataType::Boolean => unimplemented!(), // TODO
        DataType::UInt8 => pmax(s.u8()?).into_series(),
        DataType::UInt16 => pmax(s.u16()?).into_series(),
        DataType::UInt32 => pmax(s.u32()?).into_series(),
        DataType::UInt64 => pmax(s.u64()?).into_series(),
        DataType::Int8 => pmax(s.i8()?).into_series(),
        DataType::Int16 => pmax(s.i16()?).into_series(),
        DataType::Int32 => pmax(s.i32()?).into_series(),
        DataType::Int64 => pmax(s.i64()?).into_series(),
        DataType::Float32 => pmax(s.f32()?).into_series(),
        DataType::Float64 => pmax(s.f64()?).into_series(),
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => pmax(s.decimal()?).into_series(),
        #[cfg(feature = "dtype-date")]
        DataType::Date => pmax(s.date()?).into_series(),
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(_, _) => pmax(s.datetime()?).into_series(),
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(_) => pmax(s.duration()?).into_series(),
        #[cfg(feature = "dtype-time")]
        DataType::Time => pmax(s.time()?).into_series(),
        _ => unimplemented!(),
    };
    Ok(s)
}
