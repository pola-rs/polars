use crate::conversion::prelude::*;
use crate::error::JsPolarsEr;
use neon::prelude::*;
use neon::types::JsDate;
use polars::prelude::*;

use std::collections::HashMap;

// Don't change the order of these!
#[repr(u8)]
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

#[derive(Debug, Clone, PartialEq)]
pub enum JsAnyValue {
  Null,
  Boolean(bool),
  Utf8(String),
  Float64(f64),
  Date(i32),
  Datetime(i64),
  Time(i64),
  List(Vec<JsAnyValue>),
  Object(HashMap<String, JsAnyValue>),
}
impl<'a> FromJsValue<'a> for JsAnyValue {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let jsv: WrappedValue = jsv.into();
    if jsv.is_a::<JsBoolean>(cx) {
      let v = jsv.extract::<bool>(cx)?;
      Ok(JsAnyValue::Boolean(v).into())
    } else if jsv.is_a::<JsString>(cx) {
      let v = jsv.extract::<String>(cx)?;
      Ok(JsAnyValue::Utf8(v).into())
    } else if jsv.is_a::<JsNumber>(cx) {
      let v = jsv.extract::<f64>(cx)?;
      Ok(JsAnyValue::Float64(v).into())
    } else if jsv.is_a::<JsUndefined>(cx) {
      Ok(JsAnyValue::Null.into())
    } else if jsv.is_a::<JsNull>(cx) {
      Ok(JsAnyValue::Null.into())
    } else if jsv.is_a::<JsDate>(cx) {
      let js_date: Handle<JsDate> = jsv.0.downcast_or_throw(cx)?;
      let v = js_date.value(cx);
      Ok(JsAnyValue::Datetime(v as i64).into())
    } else if jsv.is_a::<JsArray>(cx) {
      let v = jsv.extract::<Vec<JsAnyValue>>(cx)?;
      Ok(JsAnyValue::List(v))
    } else if jsv.is_a::<JsObject>(cx) {
      todo!()
    } else if jsv.is_a::<JsBuffer>(cx) {
      let js_buff: Handle<JsBuffer> = jsv.0.downcast_or_throw::<JsBuffer, _>(cx)?;
      let buff: Vec<u8> = cx.borrow(&js_buff, |data| data.as_slice::<u8>().to_vec());
      match String::from_utf8(buff) {
        Ok(s) => Ok(JsAnyValue::Utf8(s).into()),
        Err(_) => Err(JsPolarsEr::Other("unknown buffer type".to_string()).into()),
      }
    } else {
      let unknown = jsv.0.to_string(cx)?.value(cx);
      let err = JsPolarsEr::Other(format!("row type not supported {:?}", unknown));
      Err(err.into())
    }
  }
}
// impl<'a> From<AnyValue<'a>> for &JsAnyValue {
//     fn from(av: AnyValue) -> Self {
//         use JsAnyValue::*;
//         match av {
//             AnyValue::Boolean(v) => &Boolean(v),
//             AnyValue::Utf8(v) => &Utf8(String::from(v)),
//             AnyValue::Float64(v) => &Float64(v),
//             AnyValue::Date(v) => &Date(v),
//             AnyValue::Datetime(v) => &Datetime(v),
//             AnyValue::Time(v) => &Time(v),
//             _ => &Null,
//         }
//     }
// }
impl<'a> Into<AnyValue<'a>> for &'a JsAnyValue {
  fn into(self) -> AnyValue<'a> {
    use JsAnyValue::*;
    match self {
      Boolean(v) => AnyValue::Boolean(*v),
      Utf8(v) => AnyValue::Utf8(v),
      Float64(v) => AnyValue::Float64(*v),
      Date(v) => AnyValue::Date(*v),
      Datetime(v) => AnyValue::Datetime(*v),
      Time(v) => AnyValue::Time(*v),
      _ => AnyValue::Null,
    }
  }
}
// impl<'a> From<AnyValue<'a>> for &JsAnyValue {
//     fn from(av: AnyValue) -> Self {
//         use JsAnyValue::*;
//         match av {
//             AnyValue::Boolean(v) => &Boolean(v),
//             AnyValue::Utf8(v) => &Utf8(String::from(v)),
//             AnyValue::Float64(v) => &Float64(v),
//             AnyValue::Date(v) => &Date(v),
//             AnyValue::Datetime(v) => &Datetime(v),
//             AnyValue::Time(v) => &Time(v),
//             _ => &Null,
//         }
//     }
// }

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
      DataType::Datetime => Datetime,
      DataType::Time => Time,
      DataType::Object(_) => Object,
      DataType::Categorical => Categorical,
      DataType::Null => {
        panic!("null not expected here")
      }
    }
  }
}

impl From<DataType> for JsDataType {
  fn from(dt: DataType) -> Self {
    (&dt).into()
  }
}

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
      JsDataType::Datetime => Datetime,
      JsDataType::Time => Time,
      JsDataType::Object => Object("object"),
      JsDataType::Categorical => Categorical,
    }
  }
}

impl<'a> ToJsValue<'a, JsNumber> for JsDataType {
  fn into_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsNumber> {
    cx.number(self as u8)
  }
}
impl FromJsValue<'_> for JsDataType {
  fn from_js(cx: &mut FunctionContext<'_>, jsv: Handle<'_, JsValue>) -> NeonResult<Self> {
    let js_str = jsv.downcast_or_throw::<JsString, _>(cx)?.value(cx);
    let s = js_str.as_str();
    Ok(str_to_polarstype(s).into())
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
    "Datetime" => DataType::Datetime,
    "Time" => DataType::Time,
    "List" => DataType::List(DataType::Null.into()),
    "Categorical" => DataType::Categorical,
    "Object" => DataType::Object("object"),
    tp => panic!("Type {} not implemented in str_to_polarstype", tp),
  }
}
