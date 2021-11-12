use crate::dataframe::JsDataFrame;
use crate::datatypes::*;
use crate::error::JsPolarsEr;
use crate::series::JsSeries;

use neon::object::PropertyKey;
use neon::prelude::*;
use neon::types::JsDate;
use polars::datatypes::*;
use polars::frame::row::Row;
use polars::prelude::*;
use polars::prelude::{Series, Utf8Chunked};
use std::path::PathBuf;

// ##### Boxed
pub trait ToJsBox<'a>: Send + 'static + Sized {
  fn to_js_box(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsBox<Self>>;
}

impl<'a, T> ToJsBox<'a> for T
where
  T: Send + 'static + Sized + Finalize,
{
  fn to_js_box(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsBox<Self>> {
    cx.boxed(self)
  }
}

pub trait ToJsValue<'a, V: Value>: Send + 'static + Sized {
  fn to_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, V>;
}

impl<'a> ToJsValue<'a, JsNumber> for usize {
  fn to_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsNumber> {
    cx.number(self as f64)
  }
}

impl<'a> ToJsValue<'a, JsString> for String {
  fn to_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsString> {
    cx.string(self)
  }
}

impl<'a> ToJsValue<'a, JsBoolean> for bool {
  fn to_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsBoolean> {
    cx.boolean(self)
  }
}

impl<'a> FromJsValue<'a> for String {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let js_string: Handle<JsString> = jsv.downcast_or_throw::<JsString, _>(cx)?;
    Ok(js_string.value(cx))
  }
}

impl<'a> FromJsValue<'a> for &'a str {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let js_string: Handle<JsString> = jsv.downcast_or_throw::<JsString, _>(cx)?;
    let v = js_string.value(cx);
    Ok(Box::leak::<'a>(v.into_boxed_str()))
  }
}
impl<'a> FromJsValue<'a> for bool {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let js_bool: Handle<JsBoolean> = jsv.downcast_or_throw::<JsBoolean, _>(cx)?;
    Ok(js_bool.value(cx))
  }
}

impl<'a> FromJsValue<'a> for PathBuf {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let js_string: Handle<JsString> = jsv.downcast_or_throw::<JsString, _>(cx)?;
    let s = js_string.value(cx);
    Ok(PathBuf::from(s))
  }
}
impl<'a> FromJsValue<'a> for Vec<AnyValue<'a>> {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let js_arr: Handle<JsArray> = jsv.downcast_or_throw::<JsArray, _>(cx)?;
    let mut rows: Vec<AnyValue> = Vec::with_capacity(js_arr.len(cx) as usize);
    for (i, _) in js_arr.to_vec(cx).iter().enumerate() {
      let jsv = js_arr.get(cx, i as u32).unwrap();
      rows.push(Wrap::<AnyValue>::from_js(cx, jsv)?.0)
    }
    Ok(rows)
  }
}

fn get_js_arr<'a>(
  cx: &mut FunctionContext<'a>,
  jsv: Handle<'a, JsValue>,
) -> NeonResult<(Handle<'a, JsArray>, usize)> {
  let js_arr: Handle<neon::prelude::JsArray> = jsv.downcast_or_throw::<JsArray, _>(cx)?;
  let size = js_arr.len(cx) as usize;
  Ok((js_arr, size))
}

impl<'a, T> FromJsValue<'a> for Wrap<ChunkedArray<T>>
where
  T: JsPolarsNumericType,
  T::Native: FromJsValue<'a>,
{
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let (js_arr, len) = get_js_arr(cx, jsv)?;
    let mut builder = PrimitiveChunkedBuilder::new("", len);

    for item in js_arr.to_vec(cx)?.iter() {
      let i = *item;
      let wv: WrappedValue = i.into();
      match wv.get_as::<T::Native>(cx) {
        Ok(val) => builder.append_value(val),
        Err(_) => builder.append_null(),
      }
    }
    Ok(Wrap(builder.finish()))
  }
}

impl<'a> FromJsValue<'a> for Wrap<Utf8Chunked> {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let (js_arr, len) = get_js_arr(cx, jsv)?;
    let mut builder = Utf8ChunkedBuilder::new("", len, len * 25);

    for item in js_arr.to_vec(cx)?.iter() {
      match <&'a str>::from_js(cx, *item) {
        Ok(val) => builder.append_value(val),
        Err(_) => builder.append_null(),
      }
    }
    Ok(Wrap(builder.finish()))
  }
}



/// # Safety
/// i cant seem to find a workaround for getting the string values.
/// I can do it with a box.leak, i think there has to be a better way around this
impl<'a> FromJsValue<'a> for Wrap<AnyValue<'a>> {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    if jsv.is_a::<JsBoolean, _>(cx) {
      let js_bool: Handle<JsBoolean> = jsv.downcast_or_throw::<JsBoolean, _>(cx)?;
      let b = js_bool.value(cx);
      Ok(AnyValue::Boolean(b).into())
    } else if jsv.is_a::<JsString, _>(cx) {
      let js_string: Handle<JsString> = jsv.downcast_or_throw::<JsString, _>(cx)?;
      let v = js_string.value(cx);
      let ss = Box::leak::<'a>(v.into_boxed_str());
      Ok(AnyValue::Utf8(ss).into())
    } else if jsv.is_a::<JsNumber, _>(cx) {
      let num: Handle<JsNumber> = jsv.downcast_or_throw::<JsNumber, _>(cx)?;
      let n = num.value(cx);
      Ok(AnyValue::Float64(n).into())
    } else if jsv.is_a::<JsUndefined, _>(cx) {
      Ok(AnyValue::Null.into())
    } else if jsv.is_a::<JsNull, _>(cx) {
      Ok(AnyValue::Null.into())
    } else if jsv.is_a::<JsDate, _>(cx) {
      let js_date: Handle<JsDate> = jsv.downcast_or_throw::<JsDate, _>(cx)?;
      let v = js_date.value(cx) as i64;
      Ok(AnyValue::Datetime(v).into())
    } else if jsv.is_a::<JsArray, _>(cx) {
      let js_arr: Handle<JsArray> = jsv.downcast_or_throw::<JsArray, _>(cx)?;
      let as_str = js_arr.to_string(cx)?.value(cx);
      let ss = Box::leak::<'a>(as_str.into_boxed_str());
      Ok(AnyValue::Utf8(ss).into())
    } else if jsv.is_a::<JsBuffer, _>(cx) {
      let js_buff: Handle<JsBuffer> = jsv.downcast_or_throw::<JsBuffer, _>(cx)?;
      let buff: Vec<u8> = cx.borrow(&js_buff, |data| data.as_slice::<u8>().to_vec());
      match String::from_utf8(buff) {
        Ok(s) => {
          let str_from_buff = Box::leak::<'a>(s.into_boxed_str());
          Ok(AnyValue::Utf8(str_from_buff).into())
        }
        Err(_) => Err(JsPolarsEr::Other("unknown buffer type".to_string()).into()),
      }
    } else if jsv.is_a::<JsObject, _>(cx) {
      Ok(AnyValue::Utf8("obj").into())
    } else {
      let unknown = jsv.to_string(cx)?.value(cx);
      let err = JsPolarsEr::Other(format!("row type not supported {:?}", unknown));
      Err(err.into())
    }
  }
}
/// FromJsObject and FromJsValue are used to provide some utility functions around the native neon bindings
pub trait FromJsValue<'a>: Sized + Send + 'a {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self>;
}

/// Wrap `Handle<JsObject>` to provide some much needed utility methods to it
pub struct WrappedObject<'a>(pub Handle<'a, JsObject>);
impl<'a> From<Handle<'a, JsObject>> for WrappedObject<'a> {
  fn from(h: Handle<'a, JsObject>) -> Self {
    Self(h)
  }
}

impl<'a> WrappedObject<'a> {
  pub fn get_as<V: FromJsValue<'a>, K: PropertyKey>(
    &self,
    cx: &mut FunctionContext<'a>,
    key: K,
  ) -> NeonResult<V> {
    let jsv = self.0.get(cx, key)?;
    V::from_js(cx, jsv)
  }
  pub fn get_arr<K: PropertyKey>(
    &self,
    cx: &mut FunctionContext<'a>,
    key: K,
  ) -> NeonResult<(Handle<'a, JsArray>, usize)> {
    let jsv = self.0.get(cx, key)?;
    get_js_arr(cx, jsv)
  }
}

pub trait FromJsObject<'a, K: PropertyKey>: Sized {
  fn get_as<V: FromJsValue<'a>>(self, cx: &mut FunctionContext<'a>, key: K) -> NeonResult<V> {
    let jsv = get_obj_key::<JsValue, _>(cx, key)?;
    V::from_js(cx, jsv)
  }
}

impl<'a, K> FromJsObject<'a, K> for Handle<'a, JsObject> where K: PropertyKey {}

/// Utility to easily wrap and unwrap a js box
pub trait FromJsBox<'a>: Send + Sized {
  fn extract_boxed(
    cx: &mut FunctionContext<'a>,
    jsv: Handle<'a, JsValue>,
  ) -> JsResult<'a, JsBox<Self>>;
}

impl<'a, T> FromJsBox<'a> for T
where
  T: Send + Sized,
{
  fn extract_boxed(
    cx: &mut FunctionContext<'a>,
    jsv: Handle<'a, JsValue>,
  ) -> JsResult<'a, JsBox<Self>> {
    let jsbox: Handle<'a, JsBox<T>> = jsv.downcast_or_throw::<JsBox<T>, _>(cx)?;
    Ok(jsbox)
  }
}

/// Uses the 'FromJsBox' to extract a jsboxed value from an object
pub trait BoxedFromJsObject<'a>: Sized {
  fn extract_boxed<V: FromJsBox<'a>>(
    &self,
    cx: &mut FunctionContext<'a>,
    key: &'a str,
  ) -> JsResult<'a, JsBox<V>>;
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

impl<'a> BoxedFromJsObject<'a> for WrappedObject<'a> {
  fn extract_boxed<V: FromJsBox<'a>>(
    &self,
    cx: &mut FunctionContext<'a>,
    key: &'a str,
  ) -> JsResult<'a, JsBox<V>> {
    let jsv = self.0.get(cx, key).unwrap();
    V::extract_boxed(cx, jsv)
  }
}

pub(crate) fn get_array_param<'a>(
  cx: &mut FunctionContext<'a>,
  key: &str,
) -> JsResult<'a, JsArray> {
  let js_array = get_obj_key::<JsArray, _>(cx, key)?.downcast_or_throw::<JsArray, _>(cx)?;
  Ok(js_array)
}

pub(crate) fn get_obj_key<'a, T: Value, K: PropertyKey>(
  cx: &mut FunctionContext<'a>,
  key: K,
) -> JsResult<'a, T> {
  let obj: Handle<'a, JsObject> = cx
    .argument::<JsObject>(0)
    .map_err(|e| JsPolarsEr::Other(format!("Internal Error {}", e)))?;
  let jsv: Handle<'a, T> = obj.get(cx, key).unwrap().downcast_or_throw::<T, _>(cx)?;
  Ok(jsv)
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
pub trait FromJsValueOld<'a>: Sized + Send + 'a {
  fn from_js(jsv: Handle<'a, JsValue>, cx: &mut FunctionContext<'a>) -> NeonResult<Self>;
}

pub(crate) fn objs_to_rows<'a>(
  cx: &mut FunctionContext<'a>,
  records: &JsArray,
) -> NeonResult<(Vec<Row<'a>>, Vec<String>)> {
  let mut rows: Vec<Row> = Vec::with_capacity(records.len(cx) as usize);
  let values = records.to_vec(cx)?;
  let first = &values[0];
  let first_obj: Handle<JsObject> = first.downcast_or_throw::<JsObject, _>(cx)?;
  let names: Vec<String> = first_obj
    .get_own_property_names(cx)?
    .to_vec(cx)?
    .iter()
    .map(|key| key.downcast_or_throw::<JsString, _>(cx).unwrap().value(cx))
    .collect();

  for item in values.iter() {
    let obj: Handle<JsObject> = item.downcast_or_throw::<JsObject, _>(cx)?;
    let keys: Handle<JsArray> = obj.get_own_property_names(cx)?;

    let keys_iter = keys.to_vec(cx)?.into_iter();
    let mut row = Vec::with_capacity(keys.len(cx) as usize);

    for k in keys_iter {
      let obj: WrappedObject = obj.into();
      let val = obj.get_as::<Wrap<AnyValue>, _>(cx, k).unwrap().0;
      row.push(val)
    }

    rows.push(Row(row));
  }

  Ok((rows, names))
}

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

macro_rules! impl_numerics {
  ($type:ty) => {
    impl<'a> FromJsValue<'a> for $type {
      fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
        let js_num: Handle<JsNumber> = jsv.downcast_or_throw::<JsNumber, _>(cx)?;
        Ok(js_num.value(cx) as $type)
      }
    }
    impl<'a> FromJsValue<'a> for Option<$type> {
      fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
        let js_num: Handle<JsNumber> = jsv.downcast_or_throw::<JsNumber, _>(cx)?;
        Ok(Some(js_num.value(cx) as $type))
      }
    }
  };
}

impl_numerics!(f32);
impl_numerics!(f64);
impl_numerics!(i16);
impl_numerics!(i32);
impl_numerics!(i64);
impl_numerics!(i8);
impl_numerics!(u16);
impl_numerics!(u32);
impl_numerics!(u64);
impl_numerics!(u8);
impl_numerics!(usize);


/// Wrap `Handle<JsObject>` to provide some much needed utility methods to it
pub struct WrappedValue<'a>(pub Handle<'a, JsValue>);
impl<'a> From<Handle<'a, JsValue>> for WrappedValue<'a> {
  fn from(h: Handle<'a, JsValue>) -> Self {
    Self(h)
  }
}

impl<'a> WrappedValue<'a> {
  pub fn get_as<V: FromJsValue<'a>>(
    self,
    cx: &mut FunctionContext<'a>
  ) -> NeonResult<V> {
    V::from_js(cx, self.0)
  }
}