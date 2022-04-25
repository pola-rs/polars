use polars::prelude::{DataType, TimeUnit};

pub fn str_to_polarstype(s: &str) -> DataType {
  match s {
    "UInt8" => DataType::UInt8,
    "UInt16" => DataType::UInt16,
    "UInt32" => DataType::UInt32,
    "UInt64" => DataType::UInt64,
    "Int8" => DataType::Int8,
    "Int16" => DataType::Int16,
    "Int32" => DataType::Int32,
    "Int64" => DataType::Int64,
    "Float32" => DataType::Float32,
    "Float64" => DataType::Float64,
    "Boolean" => DataType::Boolean,
    "Utf8" => DataType::Utf8,
    "Date" => DataType::Date,
    "Datetime" => DataType::Datetime(TimeUnit::Milliseconds, None),
    "Duration" => DataType::Duration(TimeUnit::Milliseconds),
    "Time" => DataType::Time,
    "List" => DataType::List(DataType::Null.into()),
    "Categorical" => DataType::Categorical(None),
    "Object" => DataType::Object("object"),
    tp => panic!("Type {} not implemented in str_to_polarstype", tp),
  }
}
