use polars::prelude::*;
use pyo3::{FromPyObject, PyAny, PyResult};

#[cfg(feature = "object")]
use crate::object::OBJECT_NAME;
use crate::Wrap;

// Don't change the order of these!
#[repr(u8)]
pub(crate) enum PyDataType {
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
    Datetime(TimeUnit, Option<TimeZone>),
    Duration(TimeUnit),
    Time,
    #[cfg(feature = "object")]
    Object,
    Categorical,
    Struct,
    Binary,
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
            DataType::Decimal128(_) => todo!(),
            DataType::Boolean => Bool,
            DataType::Utf8 => Utf8,
            DataType::Binary => Binary,
            DataType::List(_) => List,
            DataType::Date => Date,
            DataType::Datetime(tu, tz) => Datetime(*tu, tz.clone()),
            DataType::Duration(tu) => Duration(*tu),
            DataType::Time => Time,
            #[cfg(feature = "object")]
            DataType::Object(_) => Object,
            DataType::Categorical(_) => Categorical,
            DataType::Struct(_) => Struct,
            DataType::Null | DataType::Unknown => {
                panic!("null or unknown not expected here")
            }
        }
    }
}

impl From<DataType> for PyDataType {
    fn from(dt: DataType) -> Self {
        (&dt).into()
    }
}

impl From<PyDataType> for DataType {
    fn from(pdt: PyDataType) -> DataType {
        use DataType::*;
        match pdt {
            PyDataType::Int8 => Int8,
            PyDataType::Int16 => Int16,
            PyDataType::Int32 => Int32,
            PyDataType::Int64 => Int64,
            PyDataType::UInt8 => UInt8,
            PyDataType::UInt16 => UInt16,
            PyDataType::UInt32 => UInt32,
            PyDataType::UInt64 => UInt64,
            PyDataType::Float32 => Float32,
            PyDataType::Float64 => Float64,
            PyDataType::Bool => Boolean,
            PyDataType::Utf8 => Utf8,
            PyDataType::Binary => Binary,
            PyDataType::List => List(DataType::Null.into()),
            PyDataType::Date => Date,
            PyDataType::Datetime(tu, tz) => Datetime(tu, tz),
            PyDataType::Duration(tu) => Duration(tu),
            PyDataType::Time => Time,
            #[cfg(feature = "object")]
            PyDataType::Object => Object(OBJECT_NAME),
            PyDataType::Categorical => Categorical(None),
            PyDataType::Struct => Struct(vec![]),
        }
    }
}

impl FromPyObject<'_> for PyDataType {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let dt = ob.extract::<Wrap<DataType>>()?;
        Ok(dt.0.into())
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
