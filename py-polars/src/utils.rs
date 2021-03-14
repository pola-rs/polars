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
        "<class 'polars.datatypes.Date32'>" => DataType::Date32,
        "<class 'polars.datatypes.Date64'>" => DataType::Date64,
        "<class 'polars.datatypes.List'>" => DataType::List(ArrowDataType::Null),
        "<class 'polars.datatypes.Categorical'>" => DataType::Categorical,
        tp => panic!("Type {} not implemented in str_to_polarstype", tp),
    }
}
