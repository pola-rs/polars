use crate::dataframe::JsDataFrame;
use crate::prelude::JsPolarsEr;
use crate::prelude::Wrap;
use crate::series::JsSeries;
use napi::{CallContext, JsBigint, JsExternal, JsNumber, JsObject, JsString, JsUnknown, Result};
use polars::datatypes::DataType;
use polars_core::prelude::*;
pub trait IntoJs<T>: Send + Sized {
    fn into_js(self, cx: &CallContext) -> T {
        self.try_into_js(cx).expect("Use 'try_into_js' instead")
    }
    fn try_into_js(self, cx: &CallContext) -> Result<T>;
}

pub trait IntoJsRef<T>: Send + Sized {
    fn into_js_ref(&self, cx: &CallContext) -> T {
        self.try_into_js_ref(cx).expect("Use 'try_into_js_ref' instead")
    }
    fn try_into_js_ref(&self, cx: &CallContext) -> Result<T>;
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
impl IntoJs<napi::JsBoolean> for bool {
    fn try_into_js(self, cx: &CallContext) -> Result<napi::JsBoolean> {
        cx.env.get_boolean(self)
    }
}

macro_rules! into_js_chunked {
    ($name:ident) => {
        impl IntoJsRef<JsObject> for $name {
            fn try_into_js_ref(&self, cx: &CallContext) -> Result<JsObject> {
                let mut arr = cx.env.create_array()?;
                for (i, val) in self.into_iter().enumerate() {
                    match val {
                        Some(v) => arr.set_element(i as u32, v.into_js(&cx))?,
                        None => arr.set_element(i as u32, cx.env.get_null()?)?,
                    };
                }
                Ok(arr)
            }
        }
    };
}

impl IntoJsRef<JsObject> for DatetimeChunked {
    fn try_into_js_ref(&self, cx: &CallContext) -> Result<JsObject> {
        let mut arr = cx.env.create_array()?;
        for (i, val) in self.into_iter().enumerate() {
            match val {
                Some(v) => {
                    arr.set_element(i as u32, cx.env.create_date(v as f64)?)?
                },
                None => arr.set_element(i as u32, cx.env.get_null()?)?,
            };
        }
        Ok(arr)
    }
}
impl IntoJsRef<JsObject> for DateChunked {
    fn try_into_js_ref(&self, cx: &CallContext) -> Result<JsObject> {
        let mut arr = cx.env.create_array()?;
        for (i, val) in self.into_iter().enumerate() {
            match val {
                Some(v) => {
                    arr.set_element(i as u32, cx.env.create_date(v as f64)?)?
                },
                None => arr.set_element(i as u32, cx.env.get_null()?)?,
            };
        }
        Ok(arr)
    }
}



into_js_chunked!(Int8Chunked);
into_js_chunked!(Int16Chunked);
into_js_chunked!(Int32Chunked);
into_js_chunked!(Int64Chunked);
into_js_chunked!(UInt8Chunked);
into_js_chunked!(UInt16Chunked);
into_js_chunked!(UInt32Chunked);
into_js_chunked!(UInt64Chunked);
into_js_chunked!(Float32Chunked);
into_js_chunked!(Float64Chunked);
into_js_chunked!(BooleanChunked);
into_js_chunked!(Utf8Chunked);



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
            AnyValue::Date(v) => cx.env.create_date(v as f64).map(|v| v.into_unknown()),
            AnyValue::Datetime(v, _, _) => cx.env.create_date(v as f64).map(|v| v.into_unknown()),
            AnyValue::List(v) => {
                let v = match *v.dtype() {
                    DataType::Null => {
                        let mut arr = cx.env.create_array()?;
                        let null_count = v.null_count();
                        for i in 0..null_count {
                            arr.set_element(i as u32, cx.env.get_null()?)?;
                        }
                        Ok(arr)
                    }
                    DataType::Boolean => v.bool().map_err(JsPolarsEr::from)?.try_into_js_ref(&cx),
                    DataType::Utf8 => v.utf8().map_err(JsPolarsEr::from)?.try_into_js_ref(&cx),
                    DataType::Float32 => v.f32().map_err(JsPolarsEr::from)?.try_into_js_ref(&cx),
                    DataType::Float64 => v.f64().map_err(JsPolarsEr::from)?.try_into_js_ref(&cx),
                    DataType::UInt8 => v.u8().map_err(JsPolarsEr::from)?.try_into_js_ref(&cx),
                    DataType::UInt16 => v.u16().map_err(JsPolarsEr::from)?.try_into_js_ref(&cx),
                    DataType::UInt32 => v.u32().map_err(JsPolarsEr::from)?.try_into_js_ref(&cx),
                    DataType::UInt64 => v.u64().map_err(JsPolarsEr::from)?.try_into_js_ref(&cx),
                    DataType::Int8 => v.i8().map_err(JsPolarsEr::from)?.try_into_js_ref(&cx),
                    DataType::Int16 => v.i16().map_err(JsPolarsEr::from)?.try_into_js_ref(&cx),
                    DataType::Int32 => v.i32().map_err(JsPolarsEr::from)?.try_into_js_ref(&cx),
                    DataType::Int64 => v.i64().map_err(JsPolarsEr::from)?.try_into_js_ref(&cx),
                    DataType::Date => v.date().map_err(JsPolarsEr::from)?.try_into_js_ref(&cx),
                    DataType::Datetime(_, _) => v.datetime().map_err(JsPolarsEr::from)?.try_into_js_ref(&cx),

                    _ => panic!("unsupported"),
                };

                v.map(|val| val.into_unknown())
                
            }
            _ => cx.env.get_null().map(|v| v.into_unknown()),
        }
    }
}
