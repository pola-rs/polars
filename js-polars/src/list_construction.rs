use crate::conversion::prelude::*;
use crate::prelude::*;
use neon::prelude::*;
use polars_core::utils::CustomIterTools;

pub fn js_arr_to_list<'a>(
  cx: &mut FunctionContext<'a>,
  name: &str,
  arr: &Vec<WrappedValue<'a>>,
  dtype: &DataType,
) -> NeonResult<Series> {
  let len = arr.len();
  let s = match dtype {
    DataType::Int64 => {
      let mut builder =
        ListPrimitiveChunkedBuilder::<i64>::new(name, len, len * 5, DataType::Int64);
      for sub_seq in arr.iter() {
        let sub_seq = sub_seq.to_array(cx)?;
        let len = arr.len();
        let iter = sub_seq
          .iter()
          .map(|v| {
            if v.is_null_or_undefined(cx) {
              None
            } else {
              Some(v.extract::<i64>(cx).unwrap())
            }
          })
          .trust_my_length(len);
        builder.append_iter(iter)
      }
      builder.finish().into_series()
    }
    DataType::Float64 => {
      let mut builder =
        ListPrimitiveChunkedBuilder::<f64>::new(name, len, len * 5, DataType::Float64);
      for sub_seq in arr.iter() {
        let sub_seq = sub_seq.to_array(cx)?;
        let len = arr.len();
        let iter = sub_seq
          .iter()
          .map(|v| {
            if v.is_null_or_undefined(cx) {
              None
            } else {
              Some(v.extract::<f64>(cx).unwrap())
            }
          })
          .trust_my_length(len);
        builder.append_iter(iter)
      }
      builder.finish().into_series()
    }
    DataType::Boolean => {
      let mut builder = ListBooleanChunkedBuilder::new(name, len, len * 5);
      for sub_seq in arr.iter() {
        let sub_seq = sub_seq.to_array(cx)?;
        let len = arr.len();
        let iter = sub_seq
          .iter()
          .map(|v| {
            if v.is_null_or_undefined(cx) {
              None
            } else {
              Some(v.extract::<bool>(cx).unwrap())
            }
          })
          .trust_my_length(len);
        builder.append_iter(iter)
      }
      builder.finish().into_series()
    }
    DataType::Utf8 => {
      let mut builder = ListUtf8ChunkedBuilder::new(name, len, len * 5);
      for sub_seq in arr.iter() {
        let sub_seq = sub_seq.to_array(cx)?;
        let len = arr.len();
        let iter = sub_seq
          .iter()
          .map(|v| {
            if v.is_null_or_undefined(cx) {
              None
            } else {
              Some(v.extract::<&str>(cx).unwrap())
            }
          })
          .trust_my_length(len);
        builder.append_iter(iter)
      }
      builder.finish().into_series()
    }
    dt => {
      panic!("cannot create list array from {:?}", dt);
    }
  };

  Ok(s)
}
