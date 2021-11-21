use crate::conversion::wrap::*;
use crate::error::JsPolarsEr;
use crate::prelude::*;
use napi::JsBoolean;
use napi::JsNumber;
use napi::ValueType;
use napi::{JsObject, JsString, JsBigint, JsUnknown, Result};
use polars::prelude::*;
pub trait FromJsUnknown: Sized + Send {
  fn from_js(obj: JsUnknown) -> Result<Self>;
}

impl FromJsUnknown for String {
  fn from_js(val: JsUnknown) -> Result<Self> {
    let s: JsString = val.try_into()?;
    s.into_utf8()?.into_owned()
  }
}

impl<V> FromJsUnknown for Vec<V>
where
  V: FromJsUnknown,
{
  fn from_js(val: JsUnknown) -> Result<Self> {
    let obj: JsObject = match val.get_type()? {
      ValueType::Object => unsafe { val.cast() },
      _ => return Err(JsPolarsEr::Other("Invalid cast".to_owned()).into()),
    };
    let len = obj.get_array_length()?;
    let mut arr: Self = Vec::with_capacity(len as usize);
    for i in 0..len {
      let item: WrappedValue = obj.get_element::<JsUnknown>(i)?.into();
      let item = item.extract::<V>()?;
      arr.push(item);
    }
    Ok(arr)
  }
}

impl FromJsUnknown for Wrap<Utf8Chunked> {
  fn from_js(val: JsUnknown) -> Result<Self> {
    if val.is_array()? {
      let obj: JsObject = unsafe { val.cast() };
      let len = obj.get_array_length()?;
      let u_len = len as usize;
      let mut builder = Utf8ChunkedBuilder::new("", u_len, u_len * 25);
      for i in 0..len {
        let item: WrappedValue = obj.get_element::<JsUnknown>(i)?.into();
        match item.extract::<String>() {
          Ok(val) => builder.append_value(val),
          Err(_) => builder.append_null(),
        }
      }
      Ok(Wrap(builder.finish()))
    } else {
      Err(JsPolarsEr::Other("incorrect value type".to_owned()).into())
    }
  }
}

impl<'a> FromJsUnknown for &'a str {
  fn from_js(val: JsUnknown) -> Result<Self> {
    let s: JsString = val.try_into()?;
    let s = s.into_utf8()?.into_owned()?;
    Ok(Box::leak::<'a>(s.into_boxed_str()))
  }
}

impl FromJsUnknown for bool {
  fn from_js(val: JsUnknown) -> Result<Self> {
    let s: JsBoolean = val.try_into()?;
    s.try_into()
  }
}

impl FromJsUnknown for f64 {
  fn from_js(val: JsUnknown) -> Result<Self> {
    let s: JsNumber = val.try_into()?;
    s.try_into()
  }
}

impl FromJsUnknown for i64 {
  fn from_js(val: JsUnknown) -> Result<Self> {
    match val.get_type()? {
      ValueType::Bigint => {
        let big: JsBigint = unsafe {val.cast()};
        big.try_into()
      }
      ValueType::Number => {
        let s: JsNumber = val.try_into()?;
        s.try_into()
      }
      dt => Err(JsPolarsEr::Other(format!("cannot cast {} to u64", dt)).into())
    }
  }
}

impl FromJsUnknown for u64 {
  fn from_js(val: JsUnknown) -> Result<Self> {
    match val.get_type()? {
      ValueType::Bigint => {
        let big: JsBigint = unsafe {val.cast()};
        big.try_into()
      }
      ValueType::Number => {
        let s: JsNumber = val.try_into()?;
        Ok(s.get_int64()? as u64)
      }
      dt => Err(JsPolarsEr::Other(format!("cannot cast {} to u64", dt)).into())
    }
  }
}
impl FromJsUnknown for u32 {
  fn from_js(val: JsUnknown) -> Result<Self> {
    let s: JsNumber = val.try_into()?;
    s.get_uint32()
  }
}
impl FromJsUnknown for f32 {
  fn from_js(val: JsUnknown) -> Result<Self> {
    let s: JsNumber = val.try_into()?;
    s.get_double().map(|s| s as f32)
  }
}

impl FromJsUnknown for usize {
  fn from_js(val: JsUnknown) -> Result<Self> {
    let s: JsNumber = val.try_into()?;
    Ok(s.get_uint32()? as usize)
  }
}
impl FromJsUnknown for u8 {
  fn from_js(val: JsUnknown) -> Result<Self> {
    let s: JsNumber = val.try_into()?;
    Ok(s.get_uint32()? as u8)
  }
}
impl FromJsUnknown for u16 {
  fn from_js(val: JsUnknown) -> Result<Self> {
    let s: JsNumber = val.try_into()?;
    Ok(s.get_uint32()? as u16)
  }
}
impl FromJsUnknown for i8 {
  fn from_js(val: JsUnknown) -> Result<Self> {
    let s: JsNumber = val.try_into()?;
    Ok(s.get_int32()? as i8)
  }
}
impl FromJsUnknown for i16 {
  fn from_js(val: JsUnknown) -> Result<Self> {
    let s: JsNumber = val.try_into()?;
    Ok(s.get_int32()? as i16)
  }
}

impl FromJsUnknown for i32 {
  fn from_js(val: JsUnknown) -> Result<Self> {
    let s: JsNumber = val.try_into()?;
    s.try_into()
  }
}

impl<V> FromJsUnknown for Option<V>
where
  V: FromJsUnknown,
{
  fn from_js(val: JsUnknown) -> Result<Self> {
    let v = V::from_js(val);
    match v {
      Ok(v) => Ok(Some(v)),
      Err(_) => Ok(None),
    }
  }
}
