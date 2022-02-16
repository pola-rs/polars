use crate::prelude::NewChunkedArray;
use crate::prelude::*;
use num::{Num, NumCast};
use std::borrow::Borrow;
use std::hash::Hasher;

fn deserialize_boolean<'a>(av: AnyValue<'a>) -> AnyValue<'a> {
  match av {
    AnyValue::Boolean(v) => av,
    _ => AnyValue::Null,
  }
}

fn deserialize_num<'a, T: Num + NumCast>(av: AnyValue<'a>) -> Option<T> {
  use AnyValue::*;
  match av {
    Int32(v) => num::cast::<i32, T>(v),
    Int64(v) => num::cast::<i64, T>(v),
    UInt32(v) => num::cast::<u32, T>(v),
    UInt64(v) => num::cast::<u64, T>(v),
    Float32(v) => num::cast::<f32, T>(v),
    Float64(v) => num::cast::<f64, T>(v),
    Boolean(v) => num::cast::<i32, T>(v as i32),
    Utf8(v) => match v.parse::<i32>() {
      Ok(val) => num::cast::<i32, T>(val),
      Err(_) => None,
    },
    _ => None,
  }
}

fn deserialize_utf8<'a>(av: AnyValue<'a>) -> Option<String> {
  use AnyValue::*;

  match av {
    Utf8(v) => Some(v.clone().to_string()),
    Int32(v) => Some(v.to_string()),
    Int64(v) => Some(v.to_string()),
    UInt32(v) => Some(v.to_string()),
    UInt64(v) => Some(v.to_string()),
    Float32(v) => Some(v.to_string()),
    Float64(v) => Some(v.to_string()),
    Boolean(v) => Some(v.to_string()),
    _ => None,
  }
}

pub fn anyvalue_cast<'a>(av: &AnyValue<'a>, dtype: DataType) -> AnyValue<'a> {
  use DataType::*;
  let av = av.clone();

  match dtype {
    Int32 => deserialize_num::<i32>(av)
      .map(AnyValue::Int32)
      .unwrap_or(AnyValue::Null),
    Int64 => deserialize_num::<i64>(av)
      .map(AnyValue::Int64)
      .unwrap_or(AnyValue::Null),
    UInt32 => deserialize_num::<u32>(av)
      .map(AnyValue::UInt32)
      .unwrap_or(AnyValue::Null),
    UInt64 => deserialize_num::<u64>(av)
      .map(AnyValue::UInt64)
      .unwrap_or(AnyValue::Null),
    Float32 => deserialize_num::<f32>(av)
      .map(AnyValue::Float32)
      .unwrap_or(AnyValue::Null),
    Float64 => deserialize_num::<f64>(av)
      .map(AnyValue::Float64)
      .unwrap_or(AnyValue::Null),
    Boolean => deserialize_boolean(av),
    _ => todo!(),
  }
}
