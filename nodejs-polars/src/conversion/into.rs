use crate::dataframe::JsDataFrame;
use crate::prelude::Wrap;
use crate::series::JsSeries;
use napi::{CallContext, JsBigint, JsExternal, JsNumber, JsString, JsUnknown, Result};
use polars_core::prelude::AnyValue;
pub trait IntoJs<T>: Send + Sized {
    fn into_js(self, cx: &CallContext) -> T {
        self.try_into_js(cx).expect("Use 'try_into_js' instead")
    }
    fn try_into_js(self, cx: &CallContext) -> Result<T>;
}

impl IntoJs<JsExternal> for JsSeries {
    fn try_into_js(self, cx: &CallContext) -> Result<JsExternal> {
        cx.env.create_external(self, None)
    }
}

impl IntoJs<JsExternal> for JsDataFrame {
    fn try_into_js(self, cx: &CallContext) -> Result<JsExternal> {
        cx.env.create_external(self, None)
    }
}

impl IntoJs<JsString> for &str {
    fn try_into_js(self, cx: &CallContext) -> Result<JsString> {
        cx.env.create_string(self)
    }
}
impl IntoJs<JsString> for String {
    fn try_into_js(self, cx: &CallContext) -> Result<JsString> {
        cx.env.create_string(&self)
    }
}
impl IntoJs<JsNumber> for usize {
    fn try_into_js(self, cx: &CallContext) -> Result<JsNumber> {
        cx.env.create_uint32(self as u32)
    }
}

impl IntoJs<JsNumber> for i8 {
    fn try_into_js(self, cx: &CallContext) -> Result<JsNumber> {
        cx.env.create_uint32(self as u32)
    }
}

impl IntoJs<JsNumber> for u8 {
    fn try_into_js(self, cx: &CallContext) -> Result<JsNumber> {
        cx.env.create_uint32(self as u32)
    }
}

impl IntoJs<JsNumber> for i16 {
    fn try_into_js(self, cx: &CallContext) -> Result<JsNumber> {
        cx.env.create_uint32(self as u32)
    }
}

impl IntoJs<JsNumber> for u16 {
    fn try_into_js(self, cx: &CallContext) -> Result<JsNumber> {
        cx.env.create_uint32(self as u32)
    }
}
impl IntoJs<JsNumber> for i32 {
    fn try_into_js(self, cx: &CallContext) -> Result<JsNumber> {
        cx.env.create_int32(self)
    }
}
impl IntoJs<JsNumber> for u32 {
    fn try_into_js(self, cx: &CallContext) -> Result<JsNumber> {
        cx.env.create_uint32(self)
    }
}
impl IntoJs<JsBigint> for i64 {
    fn try_into_js(self, cx: &CallContext) -> Result<JsBigint> {
        cx.env.create_bigint_from_i64(self)
    }
}
impl IntoJs<JsBigint> for u64 {
    fn try_into_js(self, cx: &CallContext) -> Result<JsBigint> {
        cx.env.create_bigint_from_u64(self)
    }
}
impl IntoJs<JsNumber> for f32 {
    fn try_into_js(self, cx: &CallContext) -> Result<JsNumber> {
        cx.env.create_double(self as f64)
    }
}

impl IntoJs<JsNumber> for f64 {
    fn try_into_js(self, cx: &CallContext) -> Result<JsNumber> {
        cx.env.create_double(self)
    }
}

impl IntoJs<JsUnknown> for Wrap<AnyValue<'_>> {
    fn try_into_js(self, cx: &CallContext) -> Result<JsUnknown> {
        match self.0 {
            AnyValue::Boolean(v) => cx.env.get_boolean(v).map(|v| v.into_unknown()),
            AnyValue::Utf8(v) => cx.env.create_string(v).map(|v| v.into_unknown()),
            AnyValue::UInt8(v) => cx.env.create_uint32(v as u32).map(|v| v.into_unknown()),
            AnyValue::UInt16(v) => cx.env.create_uint32(v as u32).map(|v| v.into_unknown()),
            AnyValue::UInt32(v) => cx.env.create_uint32(v).map(|v| v.into_unknown()),
            AnyValue::UInt64(v) => cx.env.create_bigint_from_u64(v).map(|v| v.into_unknown())?,
            AnyValue::Int8(v) => cx.env.create_int32(v as i32).map(|v| v.into_unknown()),
            AnyValue::Int16(v) => cx.env.create_int32(v as i32).map(|v| v.into_unknown()),
            AnyValue::Int32(v) => cx.env.create_int32(v).map(|v| v.into_unknown()),
            AnyValue::Int64(v) => cx.env.create_bigint_from_i64(v).map(|v| v.into_unknown())?,
            AnyValue::Float32(v) => cx.env.create_double(v as f64).map(|v| v.into_unknown()),
            AnyValue::Float64(v) => cx.env.create_double(v).map(|v| v.into_unknown()),
            AnyValue::Date(v) => cx.env.create_date((v/ 1000000) as f64).map(|v| v.into_unknown()),
            AnyValue::Datetime(v) => cx.env.create_date((v / 1000000) as f64).map(|v| v.into_unknown()),
            AnyValue::List(v) => cx.env.to_js_value(&v).map(|v| v.into_unknown()),
            _ => cx.env.get_null().map(|v| v.into_unknown()),
        }
    }
}
