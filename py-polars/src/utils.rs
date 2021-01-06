use polars::prelude::*;

pub fn str_to_arrow_type(s: &str) -> DataType {
    match s {
        "<class 'pypolars.datatypes.UInt8'>" => DataType::UInt8,
        "<class 'pypolars.datatypes.UInt16'>" => DataType::UInt16,
        "<class 'pypolars.datatypes.UInt32'>" => DataType::UInt32,
        "<class 'pypolars.datatypes.UInt64'>" => DataType::UInt64,
        "<class 'pypolars.datatypes.Int8'>" => DataType::Int8,
        "<class 'pypolars.datatypes.Int16'>" => DataType::Int16,
        "<class 'pypolars.datatypes.Int32'>" => DataType::Int32,
        "<class 'pypolars.datatypes.Int64'>" => DataType::Int64,
        "<class 'pypolars.datatypes.Float32'>" => DataType::Float32,
        "<class 'pypolars.datatypes.Float64'>" => DataType::Float64,
        "<class 'pypolars.datatypes.Boolean'>" => DataType::Boolean,
        "<class 'pypolars.datatypes.Utf8'>" => DataType::Utf8,
        "<class 'pypolars.datatypes.Date32'>" => DataType::Date32,
        "<class 'pypolars.datatypes.Date64'>" => DataType::Date64,
        "<class 'pypolars.datatypes.List'>" => DataType::List(ArrowDataType::Null),
        tp => panic!(format!("Type {} not implemented in str_to_arrow_type", tp)),
    }
}
