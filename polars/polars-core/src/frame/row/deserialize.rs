// use crate::prelude::NewChunkedArray;
// use crate::prelude::*;
// use num::{Num, NumCast};
// use std::borrow::Borrow;
// use std::hash::Hasher;

// fn coerce_to_bool<(av: AnyValueOwned) -> AnyValueOwned {

//   match av {
//     AnyValueOwned::Boolean(v) => av,
//     AnyValueOwned::Int32(v) => if v != 0 {AnyValueOwned::Boolean(true)} else {AnyValueOwned::Boolean(true)},
//     AnyValueOwned::Int64(v) => if v != 0 {AnyValueOwned::Boolean(true)} else {AnyValueOwned::Boolean(true)},
//     AnyValueOwned::UInt32(v) => if v != 0 {AnyValueOwned::Boolean(true)} else {AnyValueOwned::Boolean(true)},
//     AnyValueOwned::UInt64(v) => if v != 0 {AnyValueOwned::Boolean(true)} else {AnyValueOwned::Boolean(true)},
//     AnyValueOwned::Float32(v) => if v != 0 {AnyValueOwned::Boolean(true)} else {AnyValueOwned::Boolean(true)},
//     AnyValueOwned::Float64(v) => if v != 0 {AnyValueOwned::Boolean(true)} else {AnyValueOwned::Boolean(true)},
//     _ => AnyValue::Null,
//   }
// }

// fn coerce_to_num<T: Num + NumCast>(av: AnyValueOwned) -> AnyValueOwned {
//   use AnyValueOwned::*;
//   match av {
//     Int32(v) => num::cast::<i32, T>(v).map(AnyValueOwned::Int32),
//     Int64(v) => num::cast::<i64, T>(v).map(AnyValueOwned::Int64),
//     UInt32(v) => num::cast::<u32, T>(v).map(AnyValueOwned::UInt32),
//     UInt64(v) => num::cast::<u64, T>(v).map(AnyValueOwned::UInt64),
//     Float32(v) => num::cast::<f32, T>(v).map(AnyValueOwned::Float32),
//     Float64(v) => num::cast::<f64, T>(v).map(AnyValueOwned::Float64),
//     Boolean(v) => num::cast::<i32, T>(v as i32).map(AnyValueOwned::Int32),
//     Utf8(v) => match v.parse::<i32>() {
//       Ok(val) => num::cast::<i32, T>(val),
//       Err(_) => None,
//     },
//     _ => None,
//   }.unwrap_or(AnyValueOwned::Null)
// }

// fn coerce_to_string<'a>(av: AnyValueOwned) -> AnyValueOwned {
//   use AnyValueOwned::*;

//   match av {
//     Utf8(v) => av,
//     _ => AnyValueOwned::Utf8(v.to_string()),
//   }
// }

// pub fn anyvalue_cast(av: AnyValueOwned, dtype: DataType) -> AnyValueOwned {
//   use DataType::*;
//   let av = av.clone();

//   match dtype {
//     Int32 => coerce_to_num::<i32>(av),
//     Int64 => coerce_to_num::<i64>(av),
//     UInt32 => coerce_to_num::<u32>(av),
//     UInt64 => coerce_to_num::<u64>(av),
//     Float32 => coerce_to_num::<f32>(av),
//     Float64 => coerce_to_num::<f64>(av),
//     Utf8 => coerce_to_string::<f64>(av),
//     Boolean => coerce_to_bool(av),
//     _ => todo!(),
//   }
// }
