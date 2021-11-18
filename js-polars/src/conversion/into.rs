use crate::conversion::jstypes::*;
use crate::conversion::prelude::*;
use neon::prelude::*;
use polars::datatypes::*;
/// wrap self into a JsBox
pub trait ToJsBox<'a>: Send + 'static + Sized {
  fn into_js_box(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsBox<Self>>;
}

impl<'a, T> ToJsBox<'a> for T
where
  T: Send + 'static + Sized + Finalize,
{
  fn into_js_box(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsBox<Self>> {
    cx.boxed(self)
  }
}

/// convert self to a js primitive type
pub trait ToJsValue<'a>: Send + Sized {
  type V: Value;
  fn into_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, Self::V>;
}

pub trait ToJsValueB<'a, V: Value>: Send + Sized {
  fn into_jsb(&self, cx: &mut FunctionContext<'a>) -> Handle<'a, V>;
}

impl<'a> ToJsValue<'a> for &'a str {
  type V = JsString;
  fn into_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsString> {
    cx.string(self)
  }
}
pub trait ToJsValueArray<'a>: Send + Sized {
  fn into_js_array(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsArray>;
}
impl<'a, T> ToJsValueArray<'a> for Vec<T>
where 
  T: ToJsValue<'a>,
{
  fn into_js_array(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsArray> {
    let js_arr = cx.empty_array();
    for (idx, val) in self.into_iter().enumerate() {
      let jsv = val.into_js(cx);
      js_arr
        .set(cx, idx as u32, jsv)
        .expect("Error converting from 'Vec<T>' to JsArray");
    }
    js_arr
  }
}

impl<'a> ToJsValue<'a> for Wrap<AnyValue<'_>> {
  type V = JsValue;
  fn into_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsValue> {
    match self.0 {
      AnyValue::UInt8(v) => cx.number(v as f64).upcast(),
      AnyValue::UInt16(v) => cx.number(v as f64).upcast(),
      AnyValue::UInt32(v) => cx.number(v as f64).upcast(),
      AnyValue::UInt64(v) => cx.number(v as f64).upcast(),
      AnyValue::Int8(v) => cx.number(v as f64).upcast(),
      AnyValue::Int16(v) => cx.number(v as f64).upcast(),
      AnyValue::Int32(v) => cx.number(v as f64).upcast(),
      AnyValue::Int64(v) => cx.number(v as f64).upcast(),
      AnyValue::Float32(v) => cx.number(v as f64).upcast(),
      AnyValue::Float64(v) => cx.number(v as f64).upcast(),
      AnyValue::Null => cx.null().upcast(),
      AnyValue::Boolean(v) => cx.boolean(v).upcast(),
      AnyValue::Utf8(v) => cx.string(v).upcast(),
      AnyValue::Categorical(idx, rev) => {
        let s = rev.get(idx);
        cx.string(s).upcast()
      }
      AnyValue::Date(v) => cx.date(v).unwrap().upcast(),
      AnyValue::Datetime(v) => cx.number(v as f64).upcast(),
      AnyValue::Time(v) => cx.number(v as f64).upcast(),
      AnyValue::List(_) => {
        // need to implement JsSeries::to_array first
        todo!()
      }
      AnyValue::Object(v) => {
        let s = format!("{}", v);
        cx.string(s).upcast()
      }
    }
  }
}
