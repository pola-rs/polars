use polars_core::prelude::*;

pub fn reinterpret(s: &Series, signed: bool) -> PolarsResult<Series> {
    Ok(match (s.dtype(), signed) {
        (DataType::UInt64, true) => s.u64().unwrap().reinterpret_signed().into_series(),
        (DataType::UInt64, false) => s.clone(),
        (DataType::Int64, false) => s.i64().unwrap().reinterpret_unsigned().into_series(),
        (DataType::Int64, true) => s.clone(),
        (DataType::UInt32, true) => s.u32().unwrap().reinterpret_signed().into_series(),
        (DataType::UInt32, false) => s.clone(),
        (DataType::Int32, false) => s.i32().unwrap().reinterpret_unsigned().into_series(),
        (DataType::Int32, true) => s.clone(),
        _ => polars_bail!(
            ComputeError:
            "reinterpret is only allowed for 64-bit/32-bit integers types, use cast otherwise"
        ),
    })
}
