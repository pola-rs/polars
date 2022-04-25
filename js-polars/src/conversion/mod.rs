pub mod extern_iterator;
pub mod extern_struct;
pub mod from;
pub mod prelude;

use crate::JsResult;
use polars::prelude::*;
use polars::series::ops::NullBehavior;
use wasm_bindgen::convert::{FromWasmAbi, IntoWasmAbi};
use wasm_bindgen::prelude::*;
use wasm_bindgen::{describe::WasmDescribe, JsCast, JsError};

#[repr(transparent)]
pub struct Wrap<T>(pub T);

impl<T> Clone for Wrap<T>
where
  T: Clone,
{
  fn clone(&self) -> Self {
    Wrap(self.0.clone())
  }
}
impl<T> From<T> for Wrap<T> {
  fn from(t: T) -> Self {
    Wrap(t)
  }
}

impl WasmDescribe for Wrap<QuantileInterpolOptions> {
  fn describe() {
    wasm_bindgen::describe::inform(wasm_bindgen::describe::STRING)
  }
}

impl FromWasmAbi for Wrap<QuantileInterpolOptions> {
  type Abi = <Vec<u8> as FromWasmAbi>::Abi;

  #[inline]
  unsafe fn from_abi(js: Self::Abi) -> Self {
    let s = String::from_utf8_unchecked(<Vec<u8>>::from_abi(js));
    let interpol = match s.as_ref() {
      "nearest" => QuantileInterpolOptions::Nearest,
      "lower" => QuantileInterpolOptions::Lower,
      "higher" => QuantileInterpolOptions::Higher,
      "midpoint" => QuantileInterpolOptions::Midpoint,
      "linear" => QuantileInterpolOptions::Linear,
      _ => panic!("not supported"),
    };
    interpol.into()
  }
}

impl WasmDescribe for Wrap<AnyValue<'_>> {
  fn describe() {
    wasm_bindgen::describe::inform(wasm_bindgen::describe::ENUM)
  }
}

impl FromWasmAbi for Wrap<AnyValue<'_>> {
  type Abi = u32;

  unsafe fn from_abi(js: u32) -> Self {
    let jsv = JsValue::from_abi(js);
    todo!()
    // Wrap(AnyValue::from_js(jsv))
  }
}

pub trait FromJsValue: Sized + Send {
  fn from_js(obj: JsValue) -> Self;
}

impl FromJsValue for AnyValue<'_> {
  fn from_js(jsv: JsValue) -> Self {
    if jsv.is_null() || jsv.is_undefined() {
      return AnyValue::Null;
    }
    let ty = jsv.js_typeof().as_string().unwrap();

    match ty.as_ref() {
      "boolean" => {
        let b: bool = js_sys::Boolean::unchecked_from_js(jsv).into();
        AnyValue::Boolean(b)
      }
      "number" => {
        let n: f64 = js_sys::Number::unchecked_from_js(jsv).into();
        AnyValue::Float64(n)
      }
      "string" => {
        let s: String = js_sys::JsString::unchecked_from_js(jsv).into();
        AnyValue::Utf8(Box::leak::<'_>(s.into_boxed_str()))
      }
      "bigint" => {
        let num = jsv.as_string().unwrap().parse::<u64>().unwrap();
        AnyValue::UInt64(num)
      }
      _ => {
        if js_sys::Date::is_type_of(&jsv) {
          let js_date = js_sys::Date::unchecked_from_js(jsv);
          let ms = js_date.get_milliseconds();

          AnyValue::Datetime(ms as i64, TimeUnit::Milliseconds, &None)
        } else if js_sys::Array::is_array(&jsv) {
          todo!()
        } else {
          todo!()
        }
      }
    }
  }
}

impl From<Wrap<AnyValue<'_>>> for JsValue {
  fn from(av: Wrap<AnyValue<'_>>) -> Self {
    match av.0 {
      AnyValue::UInt8(v) => v.into(),
      AnyValue::UInt16(v) => v.into(),
      AnyValue::UInt32(v) => v.into(),
      AnyValue::UInt64(v) => v.into(),
      AnyValue::Int8(v) => v.into(),
      AnyValue::Int16(v) => v.into(),
      AnyValue::Int32(v) => v.into(),
      AnyValue::Int64(v) => v.into(),
      AnyValue::Float32(v) => v.into(),
      AnyValue::Float64(v) => v.into(),
      AnyValue::Null => JsValue::null(),
      AnyValue::Boolean(v) => v.into(),
      AnyValue::Utf8(v) => v.into(),
      AnyValue::Categorical(idx, rev) => {
        let s = rev.get(idx);
        s.into()
      }
      AnyValue::Date(v) => {
        let jsv: JsValue = v.into();
        js_sys::Date::new(&jsv).into()
      }
      AnyValue::Datetime(v, _, _) => {
        let jsv: JsValue = v.into();
        let dt = js_sys::Date::new(&jsv);
        dt.into()
      }
      AnyValue::Time(v) => v.into(),
      _ => todo!(),
    }
  }
}

pub(crate) fn str_to_null_behavior(null_behavior: &str) -> JsResult<NullBehavior> {
  let null_behavior = match null_behavior {
    "drop" => NullBehavior::Drop,
    "ignore" => NullBehavior::Ignore,
    _ => return Err(JsError::new("use one of 'drop', 'ignore'").into()),
  };
  Ok(null_behavior)
}

pub(crate) fn str_to_rankmethod(method: &str) -> JsResult<RankMethod> {
  let method = match method {
    "min" => RankMethod::Min,
    "max" => RankMethod::Max,
    "average" => RankMethod::Average,
    "dense" => RankMethod::Dense,
    "ordinal" => RankMethod::Ordinal,
    // "random" => RankMethod::Random,
    _ => return Err(JsError::new("use one of 'avg, min, max, dense, ordinal'").into()),
  };
  Ok(method)
}
