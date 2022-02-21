use crate::conversion::wrap::Wrap;
use crate::prelude::FromJsUnknown;
use napi::JsNumber;
use napi::JsUnknown;
use polars::prelude::*;
#[repr(u32)]
pub enum JsDataType {
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
    Time,
    Object,
    Categorical,
}

impl JsDataType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "Int8" => JsDataType::Int8,
            "Int16" => JsDataType::Int16,
            "Int32" => JsDataType::Int32,
            "Int64" => JsDataType::Int64,
            "UInt8" => JsDataType::UInt8,
            "UInt16" => JsDataType::UInt16,
            "UInt32" => JsDataType::UInt32,
            "UInt64" => JsDataType::UInt64,
            "Float32" => JsDataType::Float32,
            "Float64" => JsDataType::Float64,
            "Bool" => JsDataType::Bool,
            "Utf8" => JsDataType::Utf8,
            "List" => JsDataType::List,
            "Date" => JsDataType::Date,
            "Datetime" => JsDataType::Datetime,
            "Time" => JsDataType::Time,
            "Object" => JsDataType::Object,
            "Categorical" => JsDataType::Categorical,
            _ => panic!("not a valid dtype"),
        }
    }
    pub fn to_string(self) -> String {
        match self {
            JsDataType::Int8 => "Int8",
            JsDataType::Int16 => "Int16",
            JsDataType::Int32 => "Int32",
            JsDataType::Int64 => "Int64",
            JsDataType::UInt8 => "UInt8",
            JsDataType::UInt16 => "UInt16",
            JsDataType::UInt32 => "UInt32",
            JsDataType::UInt64 => "UInt64",
            JsDataType::Float32 => "Float32",
            JsDataType::Float64 => "Float64",
            JsDataType::Bool => "Bool",
            JsDataType::Utf8 => "Utf8",
            JsDataType::List => "List",
            JsDataType::Date => "Date",
            JsDataType::Datetime => "Datetime",
            JsDataType::Time => "Time",
            JsDataType::Object => "Object",
            JsDataType::Categorical => "Categorical",
        }
        .to_owned()
    }
    pub fn to_str(&self) -> &str {
        match self {
            JsDataType::Int8 => "Int8",
            JsDataType::Int16 => "Int16",
            JsDataType::Int32 => "Int32",
            JsDataType::Int64 => "Int64",
            JsDataType::UInt8 => "UInt8",
            JsDataType::UInt16 => "UInt16",
            JsDataType::UInt32 => "UInt32",
            JsDataType::UInt64 => "UInt64",
            JsDataType::Float32 => "Float32",
            JsDataType::Float64 => "Float64",
            JsDataType::Bool => "Bool",
            JsDataType::Utf8 => "Utf8",
            JsDataType::List => "List",
            JsDataType::Date => "Date",
            JsDataType::Datetime => "Datetime",
            JsDataType::Time => "Time",
            JsDataType::Object => "Object",
            JsDataType::Categorical => "Categorical",
        }
    }
}

impl From<&DataType> for JsDataType {
    fn from(dt: &DataType) -> Self {
        use JsDataType::*;
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
            DataType::Datetime(_, _) => Datetime,
            DataType::Time => Time,
            DataType::Object(_) => Object,
            DataType::Categorical => Categorical,
            _ => panic!("null or unknown not expected here"),
        }
    }
}

impl From<napi::TypedArrayType> for JsDataType {
    fn from(dt: napi::TypedArrayType) -> Self {
        use napi::TypedArrayType::*;
        match dt {
            Int8 => JsDataType::Int8,
            Uint8 => JsDataType::UInt8,
            Uint8Clamped => JsDataType::UInt8,
            Int16 => JsDataType::Int16,
            Uint16 => JsDataType::UInt16,
            Int32 => JsDataType::Int32,
            Uint32 => JsDataType::UInt32,
            Float32 => JsDataType::Float32,
            Float64 => JsDataType::Float64,
            BigInt64 => JsDataType::Int64,
            BigUint64 => JsDataType::UInt64,
            _ => panic!("unknown datatype"),
        }
    }
}
impl From<napi::ValueType> for Wrap<DataType> {
    fn from(dt: napi::ValueType) -> Self {
        use napi::ValueType::*;
        match dt {
            Undefined | Null | Unknown => Wrap(DataType::Null),
            Boolean => Wrap(DataType::Boolean),
            Number => Wrap(DataType::Float64),
            String => Wrap(DataType::Utf8),
            Bigint => Wrap(DataType::UInt64),
            _ => panic!("unknown data type"),
        }
    }
}

// impl From<&napi::JsUnknown> for Result<Wrap<DataType>> {
//     fn from(val: &napi::JsUnknown) -> Self {
//         use napi::ValueType::*;
//         match val.get_type()? {
//             Undefined | Null  => Ok(Wrap(DataType::Null)),
//             Boolean => Ok(Wrap(DataType::Boolean)),
//             Number => Ok(Wrap(DataType::Float64)),
//             String => Ok(Wrap(DataType::Utf8)),
//             Bigint => Ok(Wrap(DataType::UInt64)),
//             Object => {
//                 if val.is_date()? {
//                     Ok(Wrap(DataType::Datetime(TimeUnit::Milliseconds, None)))
//                 } else if val.is_array()? {
//                     let i0: Wrap<DataType> = val.get_element::<JsUnknown>(0)?.into();
//                     let i1: Wrap<DataType> = val.get_element::<JsUnknown>(1)?.into();
                    
//                     let v = Vec::<DataType>::from_js(val)
//                         .map(|list| DataType::List(Box::new(coerce_data_type(&list))))
//                 } else {
//                     Ok(DataType::Utf8)
//                 }
//             }
//             _ => panic!("not supported"),
//         }
//     }
// }

impl From<DataType> for JsDataType {
    fn from(dt: DataType) -> Self {
        (&dt).into()
    }
}

#[allow(clippy::from_over_into)]
impl Into<DataType> for JsDataType {
    fn into(self) -> DataType {
        use DataType::*;
        match self {
            JsDataType::Int8 => Int8,
            JsDataType::Int16 => Int16,
            JsDataType::Int32 => Int32,
            JsDataType::Int64 => Int64,
            JsDataType::UInt8 => UInt8,
            JsDataType::UInt16 => UInt16,
            JsDataType::UInt32 => UInt32,
            JsDataType::UInt64 => UInt64,
            JsDataType::Float32 => Float32,
            JsDataType::Float64 => Float64,
            JsDataType::Bool => Boolean,
            JsDataType::Utf8 => Utf8,
            JsDataType::List => List(DataType::Null.into()),
            JsDataType::Date => Date,
            JsDataType::Datetime => Datetime(TimeUnit::Milliseconds, None),
            JsDataType::Time => Time,
            JsDataType::Object => Object("object"),
            JsDataType::Categorical => Categorical,
        }
    }
}

pub trait JsPolarsNumericType: PolarsNumericType {}
impl JsPolarsNumericType for UInt8Type {}
impl JsPolarsNumericType for UInt16Type {}
impl JsPolarsNumericType for UInt32Type {}
impl JsPolarsNumericType for UInt64Type {}
impl JsPolarsNumericType for Int8Type {}
impl JsPolarsNumericType for Int16Type {}
impl JsPolarsNumericType for Int32Type {}
impl JsPolarsNumericType for Int64Type {}
impl JsPolarsNumericType for Float32Type {}
impl JsPolarsNumericType for Float64Type {}

use crate::prelude::JsResult;

impl FromJsUnknown for JsDataType {
    fn from_js(val: JsUnknown) -> JsResult<Self> {
        match val.get_type()? {
            napi::ValueType::Number => {
                let n: JsNumber = unsafe { val.cast() };
                let val = n.get_uint32()?;
                Ok(num_to_polarstype(val).into())
            }
            napi::ValueType::String => {
                let s: napi::JsString = unsafe { val.cast() };
                let val = s.into_utf8().unwrap();
                let val = val.as_slice();
                let val = std::str::from_utf8(val).unwrap();

                Ok(JsDataType::from_str(val))
            }
            _ => panic!("not a valid dtype"),
        }
    }
}
// Don't change the order of these!
pub fn num_to_polarstype(n: u32) -> DataType {
    match n {
        0 => DataType::Int8,
        1 => DataType::Int16,
        2 => DataType::Int32,
        3 => DataType::Int64,
        4 => DataType::UInt8,
        5 => DataType::UInt16,
        6 => DataType::UInt32,
        7 => DataType::UInt64,
        8 => DataType::Float32,
        9 => DataType::Float64,
        10 => DataType::Boolean,
        11 => DataType::Utf8,
        12 => DataType::List(DataType::Null.into()),
        13 => DataType::Date,
        14 => DataType::Datetime(TimeUnit::Milliseconds, None),
        15 => DataType::Time,
        16 => DataType::Object("object"),
        17 => DataType::Categorical,
        tp => panic!("Type {} not implemented in num_to_polarstype", tp),
    }
}
