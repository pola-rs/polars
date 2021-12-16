use polars::prelude::*;

pub fn str_to_polarstype(s: &str) -> DataType {
    match s {
        "<class 'polars.datatypes.UInt8'>" => DataType::UInt8,
        "<class 'polars.datatypes.UInt16'>" => DataType::UInt16,
        "<class 'polars.datatypes.UInt32'>" => DataType::UInt32,
        "<class 'polars.datatypes.UInt64'>" => DataType::UInt64,
        "<class 'polars.datatypes.Int8'>" => DataType::Int8,
        "<class 'polars.datatypes.Int16'>" => DataType::Int16,
        "<class 'polars.datatypes.Int32'>" => DataType::Int32,
        "<class 'polars.datatypes.Int64'>" => DataType::Int64,
        "<class 'polars.datatypes.Float32'>" => DataType::Float32,
        "<class 'polars.datatypes.Float64'>" => DataType::Float64,
        "<class 'polars.datatypes.Boolean'>" => DataType::Boolean,
        "<class 'polars.datatypes.Utf8'>" => DataType::Utf8,
        "<class 'polars.datatypes.Date'>" => DataType::Date,
        "<class 'polars.datatypes.Datetime'>" => DataType::Datetime,
        "<class 'polars.datatypes.Time'>" => DataType::Time,
        "<class 'polars.datatypes.List'>" => DataType::List(DataType::Null.into()),
        "<class 'polars.datatypes.Categorical'>" => DataType::Categorical,
        "<class 'polars.datatypes.Object'>" => DataType::Object("object"),
        tp => panic!("Type {} not implemented in str_to_polarstype", tp),
    }
}

pub fn reinterpret(s: &Series, signed: bool) -> polars::prelude::Result<Series> {
    match (s.dtype(), signed) {
        (DataType::UInt64, true) => {
            let ca = s.u64().unwrap();
            Ok(ca.reinterpret_signed().into_series())
        }
        (DataType::UInt64, false) => Ok(s.clone()),
        (DataType::Int64, false) => {
            let ca = s.i64().unwrap();
            Ok(ca.reinterpret_unsigned().into_series())
        }
        (DataType::Int64, true) => Ok(s.clone()),
        _ => Err(PolarsError::ComputeError(
            "reinterpret is only allowed for 64bit integers dtype, use cast otherwise".into(),
        )),
    }
}
