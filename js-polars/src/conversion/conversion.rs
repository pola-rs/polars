use crate::conversion::prelude::*;
use crate::datatypes::*;
use crate::error::JsPolarsEr;
use polars::series::ops::NullBehavior;

// use neon::object::PropertyKey;
use neon::prelude::*;
// use neon::types::JsDate;
// use polars::datatypes::*;
// use polars::frame::row::Row;
// use polars::prelude::Utf8Chunked;
use polars::prelude::*;
use std::hash::{Hash, Hasher};
// use std::path::PathBuf;

use std::fmt::{Display, Formatter};

use polars::chunked_array::object::*;

#[derive(Debug)]
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
impl<T> Finalize for Wrap<T> {}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectValue(pub JsAnyValue);

impl Display for ObjectValue {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    write!(f, "{:#?}", self.0)
  }
}
impl Default for ObjectValue {
  fn default() -> Self {
    ObjectValue(JsAnyValue::Null)
  }
}
impl Eq for ObjectValue {}
impl Hash for ObjectValue {
  fn hash<H: Hasher>(&self, state: &mut H) {
    use JsAnyValue::*;
    let h = self.0.clone();
    match h {
      Boolean(v) => match v {
        true => state.write_u8(1),
        false => state.write_u8(0),
      },
      Utf8(v) => state.write(v.as_bytes()),
      Float64(v) => state.write_i64(v as i64),
      Date(v) => state.write_i32(v),
      Datetime(v) => state.write_i64(v),
      Time(v) => state.write_i64(v),
      List(v) => state.write_usize(v.len()),
      _ => {}
    }
    state.finish();
  }
}
impl PolarsObject for ObjectValue {
  fn type_name() -> &'static str {
    "object"
  }
}
impl<'a> FromJsValue<'a> for ObjectValue {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let v: WrappedValue = jsv.into();
    let value = v.extract::<JsAnyValue>(cx)?;
    Ok(ObjectValue(value))
  }
}

impl<'a> BoxedFromJsObject<'a> for Handle<'a, JsObject> {
  fn extract_boxed<V: FromJsBox<'a>>(
    &self,
    cx: &mut FunctionContext<'a>,
    key: &'a str,
  ) -> JsResult<'a, JsBox<V>> {
    let jsv = self.get(cx, key).unwrap();
    V::extract_boxed(cx, jsv)
  }
}
/// # Safety
///
/// The caller is responsible for checking that val is Object otherwise UB
impl From<&dyn PolarsObjectSafe> for &ObjectValue {
  fn from(val: &dyn PolarsObjectSafe) -> Self {
    unsafe { &*(val as *const dyn PolarsObjectSafe as *const ObjectValue) }
  }
}

/// Get the first arg from the js function.
///
/// All Js methods only pass in 1 arg as an object
pub(crate) fn get_params<'a>(cx: &mut FunctionContext<'a>) -> NeonResult<WrappedObject<'a>> {
  let obj: Handle<'a, JsObject> = cx
    .argument::<JsObject>(0)
    .map_err(|e| JsPolarsEr::Other(format!("Internal Error {}", e)))?;
  Ok(obj.into())
}

pub(crate) fn str_to_null_behavior(null_behavior: String) -> NeonResult<NullBehavior> {
  let null_behavior = match null_behavior.as_str() {
    "drop" => NullBehavior::Drop,
    "ignore" => NullBehavior::Ignore,
    _ => return Err(JsPolarsEr::Other("use one of 'drop', 'ignore'".to_string()).into()),
  };
  Ok(null_behavior)
}

pub(crate) fn str_to_rankmethod(method: String) -> NeonResult<RankMethod> {
  let method = match method.as_str() {
    "min" => RankMethod::Min,
    "max" => RankMethod::Max,
    "average" => RankMethod::Average,
    "dense" => RankMethod::Dense,
    "ordinal" => RankMethod::Ordinal,
    "random" => RankMethod::Random,
    _ => {
      return Err(
        JsPolarsEr::Other("use one of 'avg, min, max, dense, ordinal'".to_string()).into(),
      )
    }
  };
  Ok(method)
}

pub(crate) fn parse_strategy(strat: String) -> FillNullStrategy {
  match strat.as_str() {
    "backward" => FillNullStrategy::Backward,
    "forward" => FillNullStrategy::Forward,
    "min" => FillNullStrategy::Min,
    "max" => FillNullStrategy::Max,
    "mean" => FillNullStrategy::Mean,
    "zero" => FillNullStrategy::Zero,
    "one" => FillNullStrategy::One,
    s => panic!("Strategy {} not supported", s),
  }
}

macro_rules! impl_conversions {
  ($jstype:ty, $jsmethod:ident, $type:ty) => {
    impl<'a> ToJsValue<'a> for $type {
      type V = $jstype;
      fn into_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, $jstype> {
        cx.$jsmethod(self)
      }
    }
    impl<'a> FromJsValue<'a> for $type {
      fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
        let jsv: Handle<$jstype> = jsv.downcast_or_throw::<$jstype, _>(cx)?;
        Ok(jsv.value(cx) as $type)
      }
    }

  };
}
macro_rules! impl_casted_conversions {
  ($jstype:ty, $jsmethod:ident, $type:ty, $cast:ty) => {
    impl<'a> ToJsValue<'a> for $type {
      type V = $jstype;
      fn into_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, $jstype> {
        cx.$jsmethod(self as $cast)
      }
    }

    impl<'a> FromJsValue<'a> for $type {
      fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
        let js_num: Handle<$jstype> = jsv.downcast_or_throw::<$jstype, _>(cx)?;
        Ok(js_num.value(cx) as $type)
      }
    }
  };
}
impl_conversions!(JsString, string, String);
impl_conversions!(JsNumber, number, f32);
impl_conversions!(JsNumber, number, f64);
impl_conversions!(JsNumber, number, i8);
impl_conversions!(JsNumber, number, i16);
impl_conversions!(JsNumber, number, i32);
impl_conversions!(JsNumber, number, u8);
impl_conversions!(JsNumber, number, u16);
impl_conversions!(JsNumber, number, u32);
impl_conversions!(JsBoolean, boolean, bool);
impl_casted_conversions!(JsNumber, number, i64, f64);
impl_casted_conversions!(JsNumber, number, u64, f64);
impl_casted_conversions!(JsNumber, number, usize, f64);
