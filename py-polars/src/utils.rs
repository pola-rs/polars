use polars::prelude::*;

pub fn str_to_arrow_type(s: &str) -> ArrowDataType {
    match s {
        "<class 'pypolars.datatypes.UInt8'>" => ArrowDataType::UInt8,
        "<class 'pypolars.datatypes.UInt16'>" => ArrowDataType::UInt16,
        "<class 'pypolars.datatypes.UInt32'>" => ArrowDataType::UInt32,
        "<class 'pypolars.datatypes.UInt64'>" => ArrowDataType::UInt64,
        "<class 'pypolars.datatypes.Int8'>" => ArrowDataType::Int8,
        "<class 'pypolars.datatypes.Int16'>" => ArrowDataType::Int16,
        "<class 'pypolars.datatypes.Int32'>" => ArrowDataType::Int32,
        "<class 'pypolars.datatypes.Int64'>" => ArrowDataType::Int64,
        "<class 'pypolars.datatypes.Float32'>" => ArrowDataType::Float32,
        "<class 'pypolars.datatypes.Float64'>" => ArrowDataType::Float64,
        "<class 'pypolars.datatypes.Boolean'>" => ArrowDataType::Boolean,
        "<class 'pypolars.datatypes.Utf8'>" => ArrowDataType::Utf8,
        "<class 'pypolars.datatypes.Date32'>" => ArrowDataType::Date32(DateUnit::Day),
        "<class 'pypolars.datatypes.Date64'>" => ArrowDataType::Date64(DateUnit::Millisecond),
        "<class 'pypolars.datatypes.List'>" => ArrowDataType::List(Box::new(ArrowDataType::Null)),
        tp => panic!(format!("Type {} not implemented in str_to_arrow_type", tp)),
    }
}
