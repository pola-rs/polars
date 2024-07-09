use polars::prelude::*;
use polars_core::utils::arrow::array::Utf8ViewArray;
use pyo3::prelude::*;

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
    String,
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
    Decimal(Option<usize>, usize),
    Array(usize),
    Enum(Utf8ViewArray),
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
            DataType::Decimal(p, s) => Decimal(*p, s.expect("unexpected null decimal scale")),
            DataType::Boolean => Bool,
            DataType::String => String,
            DataType::Binary => Binary,
            DataType::Array(_, width) => Array(*width),
            DataType::List(_) => List,
            DataType::Date => Date,
            DataType::Datetime(tu, tz) => Datetime(*tu, tz.clone()),
            DataType::Duration(tu) => Duration(*tu),
            DataType::Time => Time,
            #[cfg(feature = "object")]
            DataType::Object(_, _) => Object,
            DataType::Categorical(_, _) => Categorical,
            DataType::Enum(rev_map, _) => Enum(rev_map.as_ref().unwrap().get_categories().clone()),
            DataType::Struct(_) => Struct,
            DataType::Null | DataType::Unknown(_) | DataType::BinaryOffset => {
                panic!("null or unknown not expected here")
            },
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
            PyDataType::String => String,
            PyDataType::Binary => Binary,
            PyDataType::List => List(DataType::Null.into()),
            PyDataType::Date => Date,
            PyDataType::Datetime(tu, tz) => Datetime(tu, tz),
            PyDataType::Duration(tu) => Duration(tu),
            PyDataType::Time => Time,
            #[cfg(feature = "object")]
            PyDataType::Object => Object(OBJECT_NAME, None),
            PyDataType::Categorical => Categorical(None, Default::default()),
            PyDataType::Enum(categories) => create_enum_data_type(categories),
            PyDataType::Struct => Struct(vec![]),
            PyDataType::Decimal(p, s) => Decimal(p, Some(s)),
            PyDataType::Array(width) => Array(DataType::Null.into(), width),
        }
    }
}

impl<'py> FromPyObject<'py> for PyDataType {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let dt = ob.extract::<Wrap<DataType>>()?;
        Ok(dt.0.into())
    }
}
