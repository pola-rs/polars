use polars::prelude::*;

// Don't change the order of these!
#[repr(u8)]
pub enum PyDataType {
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
    List,
    Date32,
    Date64,
    Time64Nanosecond,
    DurationNanosecond,
    DurationMillisecond,
    Object,
    Categorical,
}

impl From<&DataType> for PyDataType {
    fn from(dt: &DataType) -> Self {
        use PyDataType::*;
        match dt {
            DataType::Int8 => Int8,
            DataType::Int16 => Int16,
            DataType::Int32 => Int32,
            DataType::Int64 => Int64,
            DataType::UInt8 => UInt8,
            DataType::UInt16 => UInt16,
            DataType::UInt32 => UInt32,
            DataType::UInt64 => UInt64,
            DataType::Float32 => Float32,
            DataType::Float64 => Float64,
            DataType::Boolean => Bool,
            DataType::Utf8 => Utf8,
            DataType::List(_) => List,
            DataType::Date32 => Date32,
            DataType::Date64 => Date64,
            DataType::Time64(TimeUnit::Nanosecond) => Time64Nanosecond,
            DataType::Duration(TimeUnit::Nanosecond) => DurationNanosecond,
            DataType::Duration(TimeUnit::Millisecond) => DurationMillisecond,
            DataType::Object(_) => Object,
            DataType::Categorical => Categorical,
            dt => panic!("datatype: {:?} not supported", dt),
        }
    }
}

pub trait PyPolarsPrimitiveType: PolarsPrimitiveType {}
impl PyPolarsPrimitiveType for UInt8Type {}
impl PyPolarsPrimitiveType for UInt16Type {}
impl PyPolarsPrimitiveType for UInt32Type {}
impl PyPolarsPrimitiveType for UInt64Type {}
impl PyPolarsPrimitiveType for Int8Type {}
impl PyPolarsPrimitiveType for Int16Type {}
impl PyPolarsPrimitiveType for Int32Type {}
impl PyPolarsPrimitiveType for Int64Type {}
impl PyPolarsPrimitiveType for Float32Type {}
impl PyPolarsPrimitiveType for Float64Type {}
impl PyPolarsPrimitiveType for Date32Type {}
impl PyPolarsPrimitiveType for Date64Type {}
impl PyPolarsPrimitiveType for Time64NanosecondType {}
impl PyPolarsPrimitiveType for DurationNanosecondType {}
impl PyPolarsPrimitiveType for DurationMillisecondType {}
