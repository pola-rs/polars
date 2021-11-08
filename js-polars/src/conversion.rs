use crate::dataframe::JsDataFrame;
use crate::errors::JsPolarsEr;
use neon::prelude::*;

pub(crate) fn get_df<'a>(
  cx: &mut FunctionContext<'a>,
) -> Result<Handle<'a, BoxedDataFrame>, JsPolarsEr> {
  let obj: Handle<'a, JsObject> = cx
    .argument::<JsObject>(0)
    .map_err(|e| JsPolarsEr::Other(format!("Internal Error {}", e)))?;

  let jsv: Handle<'a, JsValue> = obj.get(cx, "_df").unwrap();
  let handle: Handle<'a, BoxedDataFrame> = jsv
    .downcast::<BoxedDataFrame, FunctionContext>(cx)
    .map_err(|e| JsPolarsEr::Other(format!("Internal Error {}", e)))?;
  Ok(handle)
}

pub(crate) fn get_string_param<'a>(
  cx: &mut FunctionContext<'a>,
  key: &str,
) -> Result<String, JsPolarsEr> {
  let obj: Handle<'a, JsObject> = cx
    .argument::<JsObject>(0)
    .map_err(|e| JsPolarsEr::Other(format!("Internal Error {}", e)))?;

  let jsv: Handle<'a, JsValue> = obj.get(cx, key).unwrap();
  let handle: Handle<'a, JsString> = jsv
    .downcast::<JsString, FunctionContext>(cx)
    .map_err(|e| JsPolarsEr::Other(format!("Internal Error {}", e)))?;
  Ok(handle.value(cx))
}

pub(crate) fn get_num_param<'a>(cx: &mut FunctionContext<'a>, key: &str) -> Result<u8, JsPolarsEr> {
  let obj: Handle<'a, JsObject> = cx
    .argument::<JsObject>(0)
    .map_err(|e| JsPolarsEr::Other(format!("Internal Error {}", e)))?;

  let jsv: Handle<'a, JsValue> = obj.get(cx, key).unwrap();
  Ok(jsv.extract::<u8>(cx)?)
}

pub type BoxedDataFrame = JsBox<JsDataFrame>;
pub type DataFrameResult<'a> = JsResult<'a, BoxedDataFrame>;
pub trait FromJsValue<'a, T> {
  type Output;
  fn extract<Output>(&self, cx: &mut FunctionContext<'a>) -> Result<Self::Output, JsPolarsEr>;
}

impl<'a> FromJsValue<'a, u8> for Handle<'a, JsValue> {
  type Output = u8;
  fn extract<Output>(&self, cx: &mut FunctionContext<'a>) -> Result<Self::Output, JsPolarsEr> {
    let value_res: Result<Handle<JsNumber>, _> = self.downcast::<JsNumber, FunctionContext>(cx);
    let value = match value_res {
      Ok(v) => v.value(cx),
      Err(_e) => return Err(JsPolarsEr::Other(format!("{}", "Type must be number type"))),
    };
    if value.is_infinite() || value.is_nan() || value.fract() > f64::EPSILON {
      return Err(JsPolarsEr::Other(format!("{} is not an integer", value)));
    }
    let value = value as i64;
    if value > u8::MAX as i64 {
      Err(JsPolarsEr::Other(format!(
        "{} is bigger than {}",
        value,
        u8::MAX
      )))
    } else if value < i8::MIN as i64 {
      Err(JsPolarsEr::Other(format!(
        "{} is smaller than {}",
        value,
        u8::MIN
      )))
    } else {
      Ok(value as u8)
    }
  }
}
