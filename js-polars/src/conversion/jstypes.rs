use crate::datatypes::*;
use crate::error::JsPolarsEr;

use neon::object::PropertyKey;
use neon::prelude::*;
use neon::types::JsDate;
use crate::conversion::prelude::*;
use std::collections::HashMap;
/// Wrap `Handle<JsObject>` to provide some much needed utility methods to it
#[derive(Clone)]
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
    let js_arr: Handle<neon::prelude::JsArray> = jsv.downcast_or_throw::<JsArray, _>(cx)?;
    let size = js_arr.len(cx) as usize;
    Ok((js_arr, size))
  }

  pub fn into_hash_map(&self, cx: &mut FunctionContext<'a>) -> NeonResult<HashMap<String, WrappedValue<'a>>> {
    let mut hmap: HashMap<String, WrappedValue<'a>> = HashMap::new();
    let keys: Handle<JsValue> = self.0.get_own_property_names(cx)?.upcast();
    let keys: WrappedValue = keys.into();
    let keys = keys.extract::<Vec<String>>(cx)?;
    for key in keys {
      let value: WrappedValue = self.0.get(cx, key.as_str())?.into();
      hmap.insert(key, value);
    }
    Ok(hmap)
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
