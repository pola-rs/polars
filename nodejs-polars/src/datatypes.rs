use crate::prelude::*;
use crate::series::JsSeries;
#[napi(js_name = "DataType")]
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
    Struct,
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
            "Struct" => JsDataType::Struct,
            _ => panic!("not a valid dtype"),
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
            DataType::Categorical(_) => Categorical,
            DataType::Struct(_) => Struct,
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

#[derive(Debug, Clone)]
pub enum JsAnyValue {
    Null,
    Boolean(bool),
    Utf8(String),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Date(i32),
    Datetime(i64, TimeUnit, Option<TimeZone>),
    Duration(i64, TimeUnit),
    Time(i64),
    List(Series),
    Struct(Vec<JsAnyValue>),
}

impl<'a> FromNapiValue for JsAnyValue {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let ty = type_of!(env, napi_val)?;
        let val = match ty {
            ValueType::Boolean => JsAnyValue::Boolean(bool::from_napi_value(env, napi_val)?),
            ValueType::Number => JsAnyValue::Float64(f64::from_napi_value(env, napi_val)?),
            ValueType::String => JsAnyValue::Utf8(String::from_napi_value(env, napi_val)?),
            ValueType::BigInt => JsAnyValue::UInt64(Wrap::<u64>::from_napi_value(env, napi_val)?.0),
            ValueType::Object => {
                if let Ok(s) = <&JsSeries>::from_napi_value(env, napi_val) {
                    JsAnyValue::List(s.series.clone())
                } else if let Ok(d) = napi::JsDate::from_napi_value(env, napi_val) {
                    let d = d.value_of()?;
                    let dt = d as i64;
                    JsAnyValue::Datetime(dt, TimeUnit::Milliseconds, None)
                } else {
                    return Err(Error::new(
                        Status::InvalidArg,
                        "Unknown JS variables cannot be represented as a JsAnyValue".to_owned(),
                    ));
                }
            }
            ValueType::Null | ValueType::Undefined => JsAnyValue::Null,
            _ => {
                return Err(Error::new(
                    Status::InvalidArg,
                    "Unknown JS variables cannot be represented as a JsAnyValue".to_owned(),
                ))
            }
        };
        Ok(val)
    }
}

impl<'a> FromNapiValue for Wrap<AnyValue<'a>> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> JsResult<Self> {
        let ty = type_of!(env, napi_val)?;
        let val = match ty {
            ValueType::Boolean => AnyValue::Boolean(bool::from_napi_value(env, napi_val)?),
            ValueType::Number => AnyValue::Float64(f64::from_napi_value(env, napi_val)?),
            ValueType::String => {
                let s = String::from_napi_value(env, napi_val)?;
                AnyValue::Utf8(Box::leak::<'_>(s.into_boxed_str()))
            }
            ValueType::BigInt => AnyValue::UInt64(Wrap::<u64>::from_napi_value(env, napi_val)?.0),
            ValueType::Object => {
                if let Ok(vals) = Vec::<Wrap<AnyValue>>::from_napi_value(env, napi_val) {
                    let vals = std::mem::transmute::<_, Vec<AnyValue>>(vals);
                    let s = Series::new("", vals);
                    AnyValue::List(s)
                } else if let Ok(s) = <&JsSeries>::from_napi_value(env, napi_val) {
                    AnyValue::List(s.series.clone())
                } else if let Ok(d) = napi::JsDate::from_napi_value(env, napi_val) {
                    let d = d.value_of()?;
                    let dt = d as i64;
                    AnyValue::Datetime(dt, TimeUnit::Milliseconds, &None)
                } else {
                    return Err(Error::new(
                        Status::InvalidArg,
                        "Unknown JS variables cannot be represented as a JsAnyValue".to_owned(),
                    ));
                }
            }
            ValueType::Null | ValueType::Undefined => AnyValue::Null,
            _ => {
                return Err(Error::new(
                    Status::InvalidArg,
                    "Unknown JS variables cannot be represented as a JsAnyValue".to_owned(),
                ))
            }
        };

        Ok(val.into())
    }
}

impl From<napi::ValueType> for Wrap<DataType> {
    fn from(dt: napi::ValueType) -> Self {
        use napi::ValueType::*;
        match dt {
            Undefined | Null | Unknown => Wrap(DataType::Null),
            Boolean => Wrap(DataType::Boolean),
            Number => Wrap(DataType::Float64),
            BigInt => Wrap(DataType::UInt64),
            _ => Wrap(DataType::Utf8),
        }
    }
}

impl ToNapiValue for JsAnyValue {
    unsafe fn to_napi_value(env: sys::napi_env, val: Self) -> Result<sys::napi_value> {
        match val {
            JsAnyValue::Null => Null::to_napi_value(env, Null),
            JsAnyValue::Boolean(b) => bool::to_napi_value(env, b),
            JsAnyValue::Int8(n) => i32::to_napi_value(env, n as i32),
            JsAnyValue::Int16(n) => i32::to_napi_value(env, n as i32),
            JsAnyValue::Int32(n) => i32::to_napi_value(env, n),
            JsAnyValue::Int64(n) => i64::to_napi_value(env, n),
            JsAnyValue::UInt8(n) => u32::to_napi_value(env, n as u32),
            JsAnyValue::UInt16(n) => u32::to_napi_value(env, n as u32),
            JsAnyValue::UInt32(n) => u32::to_napi_value(env, n),
            JsAnyValue::UInt64(n) => u64::to_napi_value(env, n),
            JsAnyValue::Float32(n) => f64::to_napi_value(env, n as f64),
            JsAnyValue::Float64(n) => f64::to_napi_value(env, n),
            JsAnyValue::Utf8(s) => String::to_napi_value(env, s),
            JsAnyValue::Date(v) => {
                let mut ptr = std::ptr::null_mut();

                check_status!(
                    napi::sys::napi_create_date(env, v as f64, &mut ptr),
                    "Failed to convert rust type `AnyValue::Date` into napi value",
                )?;

                Ok(ptr)
            }
            JsAnyValue::Datetime(v, _, _) => {
                let mut ptr = std::ptr::null_mut();

                check_status!(
                    napi::sys::napi_create_date(env, v as f64, &mut ptr),
                    "Failed to convert rust type `AnyValue::Date` into napi value",
                )?;

                Ok(ptr)
            }
            JsAnyValue::Duration(v, _) => i64::to_napi_value(env, v),
            JsAnyValue::Time(v) => i64::to_napi_value(env, v),
            JsAnyValue::List(ser) => JsSeries::to_napi_value(env, ser.into()),
            JsAnyValue::Struct(vals) => {
                let vals = std::mem::transmute::<_, Vec<JsAnyValue>>(vals);
                Vec::<JsAnyValue>::to_napi_value(env, vals)
            }
        }
    }
}

impl<'a> From<JsAnyValue> for AnyValue<'a> {
    fn from(av: JsAnyValue) -> Self {
        match av {
            JsAnyValue::Null => AnyValue::Null,
            JsAnyValue::Boolean(v) => AnyValue::Boolean(v),
            JsAnyValue::Utf8(v) => AnyValue::Utf8(Box::leak::<'_>(v.into_boxed_str())),
            JsAnyValue::UInt8(v) => AnyValue::UInt8(v),
            JsAnyValue::UInt16(v) => AnyValue::UInt16(v),
            JsAnyValue::UInt32(v) => AnyValue::UInt32(v),
            JsAnyValue::UInt64(v) => AnyValue::UInt64(v),
            JsAnyValue::Int8(v) => AnyValue::Int8(v),
            JsAnyValue::Int16(v) => AnyValue::Int16(v),
            JsAnyValue::Int32(v) => AnyValue::Int32(v),
            JsAnyValue::Int64(v) => AnyValue::Int64(v),
            JsAnyValue::Float32(v) => AnyValue::Float32(v),
            JsAnyValue::Float64(v) => AnyValue::Float64(v),
            JsAnyValue::Date(v) => AnyValue::Date(v),
            JsAnyValue::Datetime(v, w, _) => AnyValue::Datetime(v, w, &None),
            JsAnyValue::Duration(v, _) => AnyValue::Duration(v, TimeUnit::Milliseconds),
            JsAnyValue::Time(v) => AnyValue::Time(v),
            JsAnyValue::List(v) => AnyValue::List(v),
            _ => todo!(), // JsAnyValue::Struct(v) => AnyValue::Struct(v),
        }
    }
}

impl From<AnyValue<'_>> for JsAnyValue {
    fn from(av: AnyValue) -> Self {
        match av {
            AnyValue::Null => JsAnyValue::Null,
            AnyValue::Boolean(v) => JsAnyValue::Boolean(v),
            AnyValue::Utf8(v) => JsAnyValue::Utf8(v.to_owned()),
            AnyValue::UInt8(v) => JsAnyValue::UInt8(v),
            AnyValue::UInt16(v) => JsAnyValue::UInt16(v),
            AnyValue::UInt32(v) => JsAnyValue::UInt32(v),
            AnyValue::UInt64(v) => JsAnyValue::UInt64(v),
            AnyValue::Int8(v) => JsAnyValue::Int8(v),
            AnyValue::Int16(v) => JsAnyValue::Int16(v),
            AnyValue::Int32(v) => JsAnyValue::Int32(v),
            AnyValue::Int64(v) => JsAnyValue::Int64(v),
            AnyValue::Float32(v) => JsAnyValue::Float32(v),
            AnyValue::Float64(v) => JsAnyValue::Float64(v),
            AnyValue::Date(v) => JsAnyValue::Date(v),
            AnyValue::Datetime(v, w, _) => JsAnyValue::Datetime(v, w, None),
            AnyValue::Duration(v, _) => JsAnyValue::Duration(v, TimeUnit::Milliseconds),
            AnyValue::Time(v) => JsAnyValue::Time(v),
            AnyValue::List(v) => JsAnyValue::List(v),
            _ => todo!(), // JsAnyValue::Struct(v) => AnyValue::Struct(v),
        }
    }
}

impl From<&JsAnyValue> for DataType {
    fn from(av: &JsAnyValue) -> Self {
        match av {
            JsAnyValue::Null => DataType::Null,
            JsAnyValue::Boolean(_) => DataType::Boolean,
            JsAnyValue::Utf8(_) => DataType::Utf8,
            JsAnyValue::UInt8(_) => DataType::UInt8,
            JsAnyValue::UInt16(_) => DataType::UInt16,
            JsAnyValue::UInt32(_) => DataType::UInt32,
            JsAnyValue::UInt64(_) => DataType::UInt64,
            JsAnyValue::Int8(_) => DataType::Int8,
            JsAnyValue::Int16(_) => DataType::Int16,
            JsAnyValue::Int32(_) => DataType::Int32,
            JsAnyValue::Int64(_) => DataType::Int64,
            JsAnyValue::Float32(_) => DataType::Float32,
            JsAnyValue::Float64(_) => DataType::Float64,
            JsAnyValue::Date(_) => DataType::Date,
            JsAnyValue::Datetime(_, _, _) => DataType::Datetime(TimeUnit::Milliseconds, None),
            JsAnyValue::Duration(_, _) => DataType::Duration(TimeUnit::Milliseconds),
            JsAnyValue::Time(_) => DataType::Time,
            _ => todo!(), // JsAnyValue::Struct(v) => AnyValue::Struct(v),
        }
    }
}

macro_rules! impl_av_into {
    ($type:ty, $pattern:pat => $extracted_value:expr) => {
        impl TryInto<$type> for JsAnyValue {
            type Error = napi::Error;
            fn try_into(self) -> napi::Result<$type> {
                match self {
                    $pattern => $extracted_value,
                    _ => Err(napi::Error::from_reason(
                        "invalid primitive cast".to_owned(),
                    )),
                }
            }
        }
    };
}

impl_av_into!(&'static str, JsAnyValue::Utf8(v) =>  Ok(Box::leak::<'_>(v.into_boxed_str())));
impl_av_into!(String, JsAnyValue::Utf8(v) => Ok(v));
impl_av_into!(bool, JsAnyValue::Boolean(v) => Ok(v));
impl_av_into!(u8, JsAnyValue::UInt8(v) => Ok(v));
impl_av_into!(u16,JsAnyValue::UInt16(v) => Ok(v));
impl_av_into!(u32,JsAnyValue::UInt32(v) => Ok(v));
impl_av_into!(u64,JsAnyValue::UInt64(v) => Ok(v));
impl_av_into!(i8,JsAnyValue::Int8(v) => Ok(v));
impl_av_into!(i16,JsAnyValue::Int16(v) => Ok(v));
impl_av_into!(i32,JsAnyValue::Int32(v) => Ok(v));
impl_av_into!(i64,JsAnyValue::Int64(v) => Ok(v));
impl_av_into!(f32,JsAnyValue::Float32(v) => Ok(v));
impl_av_into!(f64,JsAnyValue::Float64(v) => Ok(v));

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
            JsDataType::Categorical => Categorical(None),
            JsDataType::Struct => Struct(vec![]),
        }
    }
}
