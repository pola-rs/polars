use polars::datatypes::ArrowDataType;

// Don't change the order of these!
#[repr(u8)]
pub enum DataType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Bool,
    Utf8,
    LargeList,
    Date32,
    Date64,
}

impl From<&ArrowDataType> for DataType {
    fn from(dt: &ArrowDataType) -> Self {
        use DataType::*;
        match dt {
            ArrowDataType::Int8 => Int8,
            ArrowDataType::Int16 => Int16,
            ArrowDataType::Int32 => Int32,
            ArrowDataType::Int64 => Int64,
            ArrowDataType::UInt8 => UInt8,
            ArrowDataType::UInt16 => UInt16,
            ArrowDataType::UInt32 => UInt32,
            ArrowDataType::UInt64 => UInt64,
            ArrowDataType::Float32 => Float32,
            ArrowDataType::Float64 => Float64,
            ArrowDataType::Boolean => Bool,
            ArrowDataType::Utf8 => Utf8,
            ArrowDataType::LargeList(_) => LargeList,
            ArrowDataType::Date32(_) => Date32,
            ArrowDataType::Date64(_) => Date64,
            dt => panic!(format!("datatype: {:?} not supported", dt)),
        }
    }
}
