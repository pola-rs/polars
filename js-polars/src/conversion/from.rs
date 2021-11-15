use crate::conversion::prelude::*;
use crate::datatypes::*;
use crate::error::JsPolarsEr;
use neon::object::PropertyKey;
use neon::prelude::*;
use neon::types::JsDate;
use polars::datatypes::*;
use polars::frame::row::Row;
use polars::prelude::Utf8Chunked;
use polars::prelude::*;
use std::path::PathBuf;

/// converts js value into a rust value
/// # Example
///
/// ```rust
/// impl<'a> FromJsValue<'a> for String {
///    fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
///      let v = jsv.downcast_or_throw::<JsString, _>(cx)?;
///      let s = v.value(cx);
///      Ok(s)
///    }
/// }
/// fn do_something(cx: &mut FunctionContext<'a>,jsv: Handle<'a, JsValue>) -> NeonResult<String> {
///   let s = String::from_js(cx, jsv);
/// }
/// ```
pub trait FromJsValue<'a>: Sized + Send + 'a {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self>;
}

/// extractes a property from a javascript object as type of `V`.
/// # Example
///
/// ```rust
/// fn do_something(cx: &mut FunctionContext<'a>,obj: Handle<'a, JsObject>) -> NeonResult<String> {
/// 
///   let obj = cx.empty_object();
///   let expected = String::from("value");
///   let js_str = cx.string(expected);
/// 
///   obj.set(cx, "name", js_str)?;
/// 
///   let obj: WrappedObject = obj.into();
///   let actual: String = obj.get_as::<String>(cx, "value")
/// 
///   assert_eq!(actual, expected);
/// }
/// ```
pub trait FromJsObject<'a, K: PropertyKey>: Sized {
  fn get_as<V: FromJsValue<'a>>(self, cx: &mut FunctionContext<'a>, key: K) -> NeonResult<V>; 
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


// impl<'a, K> FromJsObject<'a, K> for Handle<'a, JsObject> where K: PropertyKey {}


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
    let jsv: WrappedValue = jsv.into();
    let arr = jsv.to_array(cx)?;

    let mut builder = PrimitiveChunkedBuilder::new("", arr.len());

    for item in arr.iter() {  
      match item.extract::<T::Native>(cx) {
        Ok(val) => builder.append_value(val),
        Err(_) => builder.append_null(),
      }
    }
    Ok(Wrap(builder.finish()))
  }
}

impl<'a> FromJsValue<'a> for Wrap<Utf8Chunked> {
  fn from_js(cx: &mut FunctionContext<'a>, jsv: Handle<'a, JsValue>) -> NeonResult<Self> {
    let jsv: WrappedValue = jsv.into();
    let arr = jsv.to_array(cx)?;
    let len = arr.len();
    let mut builder = Utf8ChunkedBuilder::new("", len, len * 25);

    for value in arr.iter() {
      match  value.extract::<&'a str>(cx) {
        Ok(val) => builder.append_value(val),
        Err(_) => builder.append_null(),
      }
    }
    Ok(Wrap(builder.finish()))
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
