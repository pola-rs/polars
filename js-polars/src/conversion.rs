// use crate::dataframe::JsDataFrame;
use crate::datatypes::*;
use crate::error::JsPolarsEr;

use neon::object::PropertyKey;
use neon::prelude::*;
use neon::types::JsDate;
use polars::datatypes::*;
use polars::frame::row::Row;
use polars::prelude::Utf8Chunked;
use polars::prelude::*;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

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

// ##### Boxed
pub trait ToJsBox<'a>: Send + 'static + Sized {
  fn into_js_box(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsBox<Self>>;
}

pub trait ToJsValue<'a, V: Value>: Send + Sized {
  fn into_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, V>;
}

/// FromJsObject and FromJsValue are used to provide some utility functions around the native neon bindings
pub trait FromJsValue<'a>: Sized + Send + 'a {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self>;
}

pub trait FromJsObject<'a, K: PropertyKey>: Sized {
  fn get_as<V: FromJsValue<'a>>(self, cx: &mut FunctionContext<'a>, key: K) -> NeonResult<V> {
    let jsv = get_obj_key::<JsValue, _>(cx, key)?;
    V::from_js(cx, jsv)
  }
}

/// Utility to easily wrap and unwrap a js box
pub trait FromJsBox<'a>: Send + Sized {
  fn extract_boxed(
    cx: &mut FunctionContext<'a>,
    jsv: Handle<'a, JsValue>,
  ) -> JsResult<'a, JsBox<Self>>;
}
/// Uses the 'FromJsBox' to extract a jsboxed value from an object
pub trait BoxedFromJsObject<'a>: Sized {
  fn extract_boxed<V: FromJsBox<'a>>(
    &self,
    cx: &mut FunctionContext<'a>,
    key: &'a str,
  ) -> JsResult<'a, JsBox<V>>;
}
fn get_js_arr<'a>(
  cx: &mut FunctionContext<'a>,
  jsv: Handle<'a, JsValue>,
) -> NeonResult<(Handle<'a, JsArray>, usize)> {
  let js_arr: Handle<neon::prelude::JsArray> = jsv.downcast_or_throw::<JsArray, _>(cx)?;
  let size = js_arr.len(cx) as usize;
  Ok((js_arr, size))
}
impl<'a, T> ToJsBox<'a> for T
where
  T: Send + 'static + Sized + Finalize,
{
  fn into_js_box(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsBox<Self>> {
    cx.boxed(self)
  }
}

impl<'a> FromJsValue<'a> for &'a str {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let js_string: Handle<JsString> = jsv.downcast_or_throw::<JsString, _>(cx)?;
    let s: String = js_string.value(cx);
    Ok(Box::leak::<'a>(s.into_boxed_str()))
  }
}

impl<'a> FromJsValue<'a> for PathBuf {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let js_string: Handle<JsString> = jsv.downcast_or_throw::<JsString, _>(cx)?;
    let s = js_string.value(cx);
    Ok(PathBuf::from(s))
  }
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
      match wv.extract::<T::Native>(cx) {
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
impl<'a, V> FromJsValue<'a> for Vec<V>
where
  V: FromJsValue<'a>,
{
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let arr: Handle<JsArray> = jsv.downcast_or_throw::<JsArray, _>(cx)?;
    Ok(
      arr
        .to_vec(cx)?
        .iter()
        .map(|v| {
          let wv: WrappedValue = (*v).into();
          wv.extract::<V>(cx)
            .expect("all items in series must be of same type")
        })
        .collect(),
    )
  }
}

impl<'a> FromJsValue<'a> for ObjectValue {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let v: WrappedValue = jsv.into();
    let value = v.extract::<JsAnyValue>(cx)?;
    Ok(ObjectValue(value))
  }
}
/// # Safety
/// i cant seem to find a workaround for getting the string values.
/// I can do it with a box.leak, i think there has to be a better way around this
impl<'a> FromJsValue<'a> for Wrap<AnyValue<'a>> {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let jsv: WrappedValue = jsv.into();
    if jsv.is_a::<JsBoolean>(cx) {
      let v = jsv.extract::<bool>(cx)?;
      Ok(AnyValue::Boolean(v).into())
    } else if jsv.is_a::<JsString>(cx) {
      let v = jsv.extract::<&'a str>(cx)?;
      Ok(AnyValue::Utf8(v).into())
    } else if jsv.is_a::<JsNumber>(cx) {
      let v = jsv.extract::<f64>(cx)?;
      Ok(AnyValue::Float64(v).into())
    } else if jsv.is_a::<JsUndefined>(cx) {
      Ok(AnyValue::Null.into())
    } else if jsv.is_a::<JsNull>(cx) {
      Ok(AnyValue::Null.into())
    } else if jsv.is_a::<JsDate>(cx) {
      let js_date: Handle<JsDate> = jsv.0.downcast_or_throw(cx)?;
      let v = js_date.value(cx);
      Ok(AnyValue::Datetime(v as i64).into())
    } else if jsv.is_a::<JsArray>(cx) {
      todo!()
    } else if jsv.is_a::<JsObject>(cx) {
      todo!()
    } else if jsv.is_a::<JsBuffer>(cx) {
      let js_buff: Handle<JsBuffer> = jsv.0.downcast_or_throw::<JsBuffer, _>(cx)?;
      let buff: Vec<u8> = cx.borrow(&js_buff, |data| data.as_slice::<u8>().to_vec());
      match String::from_utf8(buff) {
        Ok(s) => {
          let str_from_buff = Box::leak::<'a>(s.into_boxed_str());
          Ok(AnyValue::Utf8(str_from_buff).into())
        }
        Err(_) => Err(JsPolarsEr::Other("unknown buffer type".to_string()).into()),
      }
    } else {
      let unknown = jsv.0.to_string(cx)?.value(cx);
      let err = JsPolarsEr::Other(format!("row type not supported {:?}", unknown));
      Err(err.into())
    }
  }
}

impl<'s> FromJsValue<'s> for Wrap<Row<'s>> {
  fn from_js(cx: &mut FunctionContext<'s>, jsv: Handle<'s, JsValue>) -> NeonResult<Self> {
    let jsv: WrappedValue = jsv.into();
    let vals = jsv.extract::<Vec<Wrap<AnyValue<'s>>>>(cx)?;
    let vals: Vec<AnyValue> = unsafe { std::mem::transmute(vals) };
    Ok(Wrap(Row(vals)))
  }
}

/// Wrap `Handle<JsObject>` to provide some much needed utility methods to it
#[derive(Clone)]
pub struct WrappedObject<'a>(pub Handle<'a, JsObject>);
impl<'a> From<Handle<'a, JsObject>> for WrappedObject<'a> {
  fn from(h: Handle<'a, JsObject>) -> Self {
    Self(h)
  }
}

// Any + Debug + Send + Sync + Display
pub trait PlRequiredTypes: PolarsObjectSafe {}

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

  pub fn get_arr_values<K: PropertyKey>(
    &self,
    cx: &mut FunctionContext<'a>,
    key: K,
  ) -> NeonResult<Vec<WrappedValue<'a>>> {
    let jsv: WrappedValue = self.0.get(cx, key)?.into();
    jsv.to_array(cx)
  }
  pub fn into_vec<V: FromJsValue<'a>, K: PropertyKey>(
    &self,
    cx: &mut FunctionContext<'a>,
    key: K,
  ) -> NeonResult<Vec<V>> {
    let jsv: WrappedValue = self.0.get(cx, key)?.into();
    jsv.extract::<Vec<V>>(cx)
  }
  pub fn keys(&self, cx: &mut FunctionContext<'a>) -> NeonResult<Vec<String>> {
    let keys: Handle<JsValue> = self.0.get_own_property_names(cx)?.upcast();
    let keys: WrappedValue = keys.into();
    let keys = keys.extract::<Vec<String>>(cx)?;
    Ok(keys)
  }

  pub fn values(&self, cx: &mut FunctionContext<'a>) -> NeonResult<Vec<WrappedValue<'a>>> {
    let keys: Handle<JsValue> = self.0.get_own_property_names(cx)?.upcast();
    let keys: WrappedValue = keys.into();
    let keys = keys.extract::<Vec<String>>(cx)?;
    let mut values: Vec<WrappedValue> = Vec::with_capacity(keys.len());
    for key in keys {
      let value: WrappedValue = self.0.get(cx, key.as_str())?.into();
      values.push(value);
    }
    Ok(values)
  }

  pub fn to_pairs(
    &self,
    cx: &mut FunctionContext<'a>,
  ) -> NeonResult<Vec<(String, WrappedValue<'a>)>> {
    let keys: Handle<JsValue> = self.0.get_own_property_names(cx)?.upcast();
    let keys: WrappedValue = keys.into();
    let keys = keys.extract::<Vec<String>>(cx)?;
    let mut values: Vec<(String, WrappedValue)> = Vec::with_capacity(keys.len());
    for key in keys {
      let value: WrappedValue = self.0.get(cx, key.as_str())?.into();
      values.push((key, value));
    }
    Ok(values)
  }
}

impl<'a, K> FromJsObject<'a, K> for Handle<'a, JsObject> where K: PropertyKey {}
/// Wrap `Handle<JsObject>` to provide some much needed utility methods to it
#[derive(Clone)]
pub struct WrappedValue<'a>(pub Handle<'a, JsValue>);
impl<'a> From<Handle<'a, JsValue>> for WrappedValue<'a> {
  fn from(h: Handle<'a, JsValue>) -> Self {
    Self(h)
  }
}

impl<'a> WrappedValue<'a> {
  pub fn extract<V: FromJsValue<'a>>(&self, cx: &mut FunctionContext<'a>) -> NeonResult<V> {
    V::from_js(cx, self.0)
  }
  pub fn is_a<'b, U: Value>(&self, cx: &mut FunctionContext<'a>) -> bool {
    self.0.is_a::<U, _>(cx)
  }
  pub fn dtype(&self, cx: &mut FunctionContext<'a>) -> JsDataType {
    if self.is_a::<JsBoolean>(cx) {
      JsDataType::Bool
    } else if self.is_a::<JsString>(cx) {
      JsDataType::Utf8
    } else if self.is_a::<JsNumber>(cx) {
      JsDataType::Float64
    } else if self.is_a::<JsDate>(cx) {
      JsDataType::Datetime
    } else if self.is_a::<JsArray>(cx) {
      JsDataType::List
    } else if self.is_a::<JsObject>(cx) {
      JsDataType::Object
    } else if self.is_a::<JsBuffer>(cx) {
      JsDataType::Utf8
    } else {
      unimplemented!()
    }
  }
  pub fn to_array(&self, cx: &mut FunctionContext<'a>) -> NeonResult<Vec<WrappedValue<'a>>> {
    let js_arr: Vec<Handle<JsValue>> = self.0.downcast_or_throw::<JsArray, _>(cx)?.to_vec(cx)?;
    let values: Vec<WrappedValue> = js_arr.iter().map(|v| WrappedValue(*v)).collect();
    Ok(values)
  }
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

// pub(crate) fn get_array_param<'a>(
//   cx: &mut FunctionContext<'a>,
//   key: &str,
// ) -> JsResult<'a, JsArray> {
//   let js_array = get_obj_key::<JsArray, _>(cx, key)?.downcast_or_throw::<JsArray, _>(cx)?;
//   Ok(js_array)
// }

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

impl<'a> ToJsValue<'a, JsString> for &'a str {
  fn into_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsString> {
    cx.string(self)
  }
}

macro_rules! impl_conversions {
  ($jstype:ty, $jsmethod:ident, $type:ty) => {
    impl<'a> ToJsValue<'a, $jstype> for $type {
      fn into_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, $jstype> {
        cx.$jsmethod(self)
      }
    }
    impl<'a> FromJsValue<'a> for $type {
      fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
        let js_num: Handle<$jstype> = jsv.downcast_or_throw::<$jstype, _>(cx)?;
        Ok(js_num.value(cx) as $type)
      }
    }
    impl<'a> FromJsValue<'a> for Option<$type> {
      fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
        let js_num: Handle<$jstype> = jsv.downcast_or_throw::<$jstype, _>(cx)?;
        Ok(Some(js_num.value(cx) as $type))
      }
    }
  };
}
macro_rules! impl_casted_conversions {
  ($jstype:ty, $jsmethod:ident, $type:ty, $cast:ty) => {
    impl<'a> ToJsValue<'a, $jstype> for $type {
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
    impl<'a> FromJsValue<'a> for Option<$type> {
      fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
        let js_num: Handle<$jstype> = jsv.downcast_or_throw::<$jstype, _>(cx)?;
        Ok(Some(js_num.value(cx) as $type))
      }
    }
  };
}
// impl<'a> ToJsValue<'a, JsBoolean> for bool {
//   fn into_js(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsBoolean> {
//     cx.boolean(self as bool)
//   }
// }
impl_conversions!(JsString, string, String);
// impl_conversions!(JsString, string, &'a str);

impl_conversions!(JsNumber, number, f32);
impl_conversions!(JsNumber, number, f64);
impl_conversions!(JsNumber, number, i8);
impl_conversions!(JsNumber, number, i16);
impl_conversions!(JsNumber, number, i32);
impl_casted_conversions!(JsNumber, number, i64, f64);
impl_conversions!(JsNumber, number, u8);
impl_conversions!(JsNumber, number, u16);
impl_conversions!(JsNumber, number, u32);
impl_casted_conversions!(JsNumber, number, u64, f64);
impl_casted_conversions!(JsNumber, number, usize, f64);
impl_conversions!(JsBoolean, boolean, bool);
