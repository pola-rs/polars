use crate::errors::JsPolarsEr;
use neon::prelude::*;
use neon::types::JsDate;
use polars::datatypes::*;
use polars::frame::row::Row;
use polars::prelude::*;

pub trait IntoJs<'a>: Send + 'static + Sized {
  fn into_js_box(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsBox<Self>>;
}

impl<'a, T> IntoJs<'a> for T
where
  T: Send + 'static + Sized + Finalize,
{
  fn into_js_box(self, cx: &mut FunctionContext<'a>) -> Handle<'a, JsBox<Self>> {
    cx.boxed(self)
  }
}
pub trait IntoJsValue<'a>: Send + 'static + Sized {
  type Output: Value;
  fn into_js_value(self, cx: &mut FunctionContext<'a>) -> Handle<'a, Self::Output>;
}

impl<'a> IntoJsValue<'a> for usize {
  type Output = JsNumber;
  fn into_js_value(self, cx: &mut FunctionContext<'a>) -> Handle<'a, Self::Output> {
    cx.number(self as f64)
  }
}
impl<'a> IntoJsValue<'a> for String {
  type Output = JsString;
  fn into_js_value(self, cx: &mut FunctionContext<'a>) -> Handle<'a, Self::Output> {
    cx.string(self)
  }
}

impl<'a> IntoJsValue<'a> for bool {
  type Output = JsBoolean;
  fn into_js_value(self, cx: &mut FunctionContext<'a>) -> Handle<'a, Self::Output> {
    cx.boolean(self)
  }
}

pub trait FromContextBox<'a>: Send + 'static + Sized {
  fn extract_boxed(cx: &mut FunctionContext<'a>, key: &'a str) -> JsResult<'a, JsBox<Self>>;
}

impl<'a, T> FromContextBox<'a> for T
where
  T: Send + 'static + Sized,
{
  fn extract_boxed(cx: &mut FunctionContext<'a>, key: &'a str) -> JsResult<'a, JsBox<Self>> {
    get_obj_key::<JsBox<T>>(cx, key)
  }
}

pub trait FromContext<'a>: Send + 'static + Sized {
  fn extract_val(cx: &mut FunctionContext<'a>, key: &'a str) -> NeonResult<Self>;
}

impl<'a> FromContext<'a> for f64 {
  fn extract_val(cx: &mut FunctionContext<'a>, key: &'a str) -> NeonResult<Self> {
    let js_num = get_obj_key::<JsNumber>(cx, key)?;
    Ok(js_num.value(cx) as f64)
  }
}

pub trait FromNamedParameter<'a, T: Sized> {
  fn from_named_parameter(&mut self, key: &'a str) -> NeonResult<T>;
}

impl<'a> FromNamedParameter<'a, String> for FunctionContext<'a> {
  fn from_named_parameter(&mut self, key: &'a str) -> NeonResult<String> {
    Ok(
      get_obj_key::<JsString>(self, key)?
        .downcast_or_throw::<JsString, _>(self)?
        .value(self),
    )
  }
}

impl<'a> FromNamedParameter<'a, f64> for FunctionContext<'a> {
  fn from_named_parameter(&mut self, key: &'a str) -> NeonResult<f64> {
    Ok(
      get_obj_key::<JsNumber>(self, key)?
        .downcast_or_throw::<JsNumber, _>(self)?
        .value(self),
    )
  }
}

impl<'a> FromNamedParameter<'a, Option<usize>> for FunctionContext<'a> {
  fn from_named_parameter(&mut self, key: &'a str) -> NeonResult<Option<usize>> {
    Ok(Some(
      get_obj_key::<JsNumber>(self, key)?
        .downcast_or_throw::<JsNumber, _>(self)?
        .value(self) as usize,
    ))
  }
}

pub(crate) fn get_array_param<'a>(
  cx: &mut FunctionContext<'a>,
  key: &str,
) -> JsResult<'a, JsArray> {
  let js_array = get_obj_key::<JsArray>(cx, key)?.downcast_or_throw::<JsArray, _>(cx)?;
  Ok(js_array)
}

pub(crate) fn get_obj_key<'a, T: Value>(
  cx: &mut FunctionContext<'a>,
  key: &str,
) -> JsResult<'a, T> {
  let obj: Handle<'a, JsObject> = cx
    .argument::<JsObject>(0)
    .map_err(|e| JsPolarsEr::Other(format!("Internal Error {}", e)))?;
  let jsv: Handle<'a, T> = obj.get(cx, key).unwrap().downcast_or_throw::<T, _>(cx)?;
  Ok(jsv)
}

pub trait FromJsValue<'a>: Sized + Send + 'a {
  fn from_js(jsv: Handle<'a, JsValue>, cx: &mut FunctionContext<'a>) -> NeonResult<Self>;
}

impl<'a> FromJsValue<'a> for Wrap<Utf8Chunked> {
  fn from_js(jsv: Handle<'a, JsValue>, cx: &mut FunctionContext<'a>) -> NeonResult<Self> {
    let js_str: Handle<JsString> = jsv.downcast_or_throw::<JsString, _>(cx)?;
    let v: String = js_str.value(cx);
    let len = v.len();
    let mut builder = Utf8ChunkedBuilder::new("", len, len * 25);
    for res in v.chars() {
      builder.append_value(res.to_string())
    }
    Ok(Wrap(builder.finish()))
  }
}

impl<'a> FromJsValue<'a> for Wrap<AnyValue<'a>> {
  fn from_js(jsv: Handle<'a, JsValue>, cx: &mut FunctionContext<'a>) -> NeonResult<Self> {
    if jsv.is_a::<JsBoolean, _>(cx) {
      let js_bool: Handle<JsBoolean> = jsv.downcast_or_throw::<JsBoolean, _>(cx)?;
      let b = js_bool.value(cx);
      Ok(AnyValue::Boolean(b).into())
    } else if jsv.is_a::<JsString, _>(cx) {
      unimplemented!()
    } else if jsv.is_a::<JsNumber, _>(cx) {
      let num: Handle<JsNumber> = jsv.downcast_or_throw::<JsNumber, _>(cx)?;
      let n = num.value(cx);
      Ok(AnyValue::Float64(n).into())
    } else if jsv.is_a::<JsUndefined, _>(cx) {
      Ok(AnyValue::Null.into())
    } else if jsv.is_a::<JsNull, _>(cx) {
      Ok(AnyValue::Null.into())
    } else if jsv.is_a::<JsDate, _>(cx) {
      let js_date: Handle<JsNumber> = jsv.downcast_or_throw::<JsNumber, _>(cx)?;
      let v = js_date.value(cx) as i64;
      Ok(AnyValue::Datetime(v * 1000).into())
    } else {
      let unknown = jsv.to_string(cx)?.value(cx);
      let err = JsPolarsEr::Other(format!("row type not supported {:?}", unknown));

      Err(err.into())
    }
  }
}

pub(crate) fn objs_to_rows<'a>(
  cx: &mut FunctionContext<'a>,
  records: &JsArray,
) -> NeonResult<(Vec<Row<'a>>, Vec<String>)> {
  let mut rows: Vec<Row> = Vec::with_capacity(records.len(cx) as usize);
  let values = records.to_vec(cx)?;
  let first = &values[0];
  let obj: Handle<JsObject> = first.downcast_or_throw::<JsObject, _>(cx)?;
  let names: Vec<String> = obj
    .get_own_property_names(cx)?
    .to_vec(cx)?
    .iter()
    .map(|key| key.downcast_or_throw::<JsString, _>(cx).unwrap().value(cx))
    .collect();

  for d in values.iter() {
    let obj: Handle<JsObject> = d.downcast_or_throw::<JsObject, _>(cx)?;
    let keys: Handle<JsArray> = obj.get_own_property_names(cx)?;
    let keys_iter = keys.to_vec(cx)?.into_iter();
    let mut row = Vec::with_capacity(keys.len(cx) as usize);

    for k in keys_iter {
      let val = obj.get(cx, k).unwrap();
      let val = Wrap::<AnyValue>::from_js(val, cx)?.0;
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
