use neon::prelude::*;
use polars::datatypes::*;
use crate::conversion::prelude::*;

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
pub trait ToJsValue<'a, V: Value>: Send + Sized {
  fn into_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, V>;
}

impl<'a> ToJsValue<'a, JsString> for &'a str {
  fn into_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsString> {
    cx.string(self)
  }
}


impl<'a> ToJsValue<'a, JsValue> for Wrap<AnyValue<'a>> {
  fn into_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsValue> {
    match self.0 {
      AnyValue::UInt8(v) => v.into_js(cx).upcast(),
      AnyValue::UInt16(v) => v.into_js(cx).upcast(),
      AnyValue::UInt32(v) => v.into_js(cx).upcast(),
      AnyValue::UInt64(v) => v.into_js(cx).upcast(),
      AnyValue::Int8(v) => v.into_js(cx).upcast(),
      AnyValue::Int16(v) => v.into_js(cx).upcast(),
      AnyValue::Int32(v) => v.into_js(cx).upcast(),
      AnyValue::Int64(v) => v.into_js(cx).upcast(),
      AnyValue::Float32(v) => v.into_js(cx).upcast(),
      AnyValue::Float64(v) => v.into_js(cx).upcast(),
      AnyValue::Null => cx.null().upcast(),
      AnyValue::Boolean(v) => v.into_js(cx).upcast(),
      AnyValue::Utf8(v) => v.into_js(cx).upcast(),
      AnyValue::Categorical(idx, rev) => {
        let s = rev.get(idx);
        s.into_js(cx).upcast()
      }
      AnyValue::Date(v) => v.into_js(cx).upcast(),
      AnyValue::Datetime(v) => v.into_js(cx).upcast(),
      AnyValue::Time(v) => v.into_js(cx).upcast(),
      AnyValue::List(_) => {
        todo!()
      }
      AnyValue::Object(v) => {
        let s = format!("{}", v);
        s.into_js(cx).upcast()
      }
    }
  }
}