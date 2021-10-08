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
    Date,
    Datetime,
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
            DataType::Date => Date,
            DataType::Datetime => Datetime,
            DataType::Object(_) => Object,
            DataType::Categorical => Categorical,
            dt => panic!("datatype: {:?} not supported", dt),
        }
    }
}

pub trait PyPolarsNumericType: PolarsNumericType {}
impl PyPolarsNumericType for UInt8Type {}
impl PyPolarsNumericType for UInt16Type {}
impl PyPolarsNumericType for UInt32Type {}
impl PyPolarsNumericType for UInt64Type {}
impl PyPolarsNumericType for Int8Type {}
impl PyPolarsNumericType for Int16Type {}
impl PyPolarsNumericType for Int32Type {}
impl PyPolarsNumericType for Int64Type {}
impl PyPolarsNumericType for Float32Type {}
impl PyPolarsNumericType for Float64Type {}
