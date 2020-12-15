use polars::prelude::*;

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
    List,
    Date32,
    Date64,
    Time64Nanosecond,
    DurationNanosecond,
    DurationMillisecond,
    Object,
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
            ArrowDataType::List(_) => List,
            ArrowDataType::Date32(_) => Date32,
            ArrowDataType::Date64(_) => Date64,
            ArrowDataType::Time64(TimeUnit::Nanosecond) => Time64Nanosecond,
            ArrowDataType::Duration(TimeUnit::Nanosecond) => DurationNanosecond,
            ArrowDataType::Duration(TimeUnit::Millisecond) => DurationMillisecond,
            ArrowDataType::Binary => Object,
            dt => panic!(format!("datatype: {:?} not supported", dt)),
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
impl PyPolarsPrimitiveType for BooleanType {}
