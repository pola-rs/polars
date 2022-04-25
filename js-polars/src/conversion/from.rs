use wasm_bindgen::prelude::*;
use wasm_bindgen::JsObject;

use super::super::{error::JsPolarsErr, JsResult};


// macro_rules! numbers {
//   ($($n:ident)*) => ($(
//       impl FromJs<JsValue> for $n {
//           fn extract(val: JsValue) -> JsResult<Self>  {
//             match val.as_f64() {
//               Some(v) => Ok(v as $n),
//               None => Err(JsPolarsErr::Other("invalid cast".into()).into())
//             }
//           }
//       }
//   )*)
// }

// numbers! { i8 u8 i16 u16 i32 u32 f32 f64 usize }

#[repr(u32)]
#[wasm_bindgen(js_name=DataType)]
pub enum JsDataType {
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
  Float32,
  Float64,
  Bool,
  Utf8,
  List,
  Date,
  Datetime,
  Time,
  Object,
  Categorical,
}
