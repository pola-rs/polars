use crate::conversion::prelude::*;
use crate::prelude::*;
use napi::JsBoolean;
use napi::JsNumber;
use napi::JsObject;
use napi::JsTypedArrayValue;
use napi::JsUnknown;

pub fn js_arr_to_list(name: &str, obj: &JsObject, dtype: &DataType) -> JsResult<Series> {
  let len = obj.get_array_length()?;
  let s = match dtype {
    DataType::Int8 => {
      let mut builder = ListPrimitiveChunkedBuilder::<i8>::new(
        name,
        len as usize,
        (len as usize) * 5,
        DataType::Int64,
      );
      for idx in 0..len {
        let sub_seq: JsObject = obj.get_element(idx)?;
        let sub_seq_len = sub_seq.get_array_length()?;
        let mut v: Vec<Option<i8>> = Vec::with_capacity(sub_seq_len as usize);
        for idx in 0..sub_seq_len {
          let item: JsResult<JsNumber> = sub_seq.get_element(idx);
          match item {
            Ok(i) => v.push(Some(i.get_uint32()? as i8)),
            _ => v.push(None),
          }
        }
        builder.append_iter(v.into_iter());
      }
      builder.finish().into_series()
    }
    DataType::UInt8 => {
      let mut builder = ListPrimitiveChunkedBuilder::<u8>::new(
        name,
        len as usize,
        (len as usize) * 5,
        DataType::Int64,
      );
      for idx in 0..len {
        let sub_seq: JsObject = obj.get_element(idx)?;
        let sub_seq_len = sub_seq.get_array_length()?;
        let mut v: Vec<Option<u8>> = Vec::with_capacity(sub_seq_len as usize);
        for idx in 0..sub_seq_len {
          let item: JsResult<JsNumber> = sub_seq.get_element(idx);
          match item {
            Ok(i) => v.push(Some(i.get_uint32()? as u8)),
            _ => v.push(None),
          }
        }
        builder.append_iter(v.into_iter());
      }
      builder.finish().into_series()
    }
    DataType::Int16 => {
      let mut builder = ListPrimitiveChunkedBuilder::<i16>::new(
        name,
        len as usize,
        (len as usize) * 5,
        DataType::Int64,
      );
      for idx in 0..len {
        let sub_seq: JsObject = obj.get_element(idx)?;
        let sub_seq_len = sub_seq.get_array_length()?;
        let mut v: Vec<Option<i16>> = Vec::with_capacity(sub_seq_len as usize);
        for idx in 0..sub_seq_len {
          let item: JsResult<JsNumber> = sub_seq.get_element(idx);
          match item {
            Ok(i) => v.push(Some(i.get_uint32()? as i16)),
            _ => v.push(None),
          }
        }
        builder.append_iter(v.into_iter());
      }
      builder.finish().into_series()
    }
    DataType::UInt16 => {
      let mut builder = ListPrimitiveChunkedBuilder::<u16>::new(
        name,
        len as usize,
        (len as usize) * 5,
        DataType::Int64,
      );
      for idx in 0..len {
        let sub_seq: JsObject = obj.get_element(idx)?;
        let sub_seq_len = sub_seq.get_array_length()?;
        let mut v: Vec<Option<u16>> = Vec::with_capacity(sub_seq_len as usize);
        for idx in 0..sub_seq_len {
          let item: JsResult<JsNumber> = sub_seq.get_element(idx);
          match item {
            Ok(i) => v.push(Some(i.get_uint32()? as u16)),
            _ => v.push(None),
          }
        }
        builder.append_iter(v.into_iter());
      }
      builder.finish().into_series()
    }
    DataType::Int32 => {
      let mut builder = ListPrimitiveChunkedBuilder::<i32>::new(
        name,
        len as usize,
        (len as usize) * 5,
        DataType::Int64,
      );
      for idx in 0..len {
        let sub_seq: JsObject = obj.get_element(idx)?;
        let sub_seq_len = sub_seq.get_array_length()?;
        let mut v: Vec<Option<i32>> = Vec::with_capacity(sub_seq_len as usize);
        for idx in 0..sub_seq_len {
          let item: JsResult<JsNumber> = sub_seq.get_element(idx);
          match item {
            Ok(i) => v.push(Some(i.get_uint32()? as i32)),
            _ => v.push(None),
          }
        }
        builder.append_iter(v.into_iter());
      }
      builder.finish().into_series()
    }
    DataType::UInt32 => {
      let mut builder = ListPrimitiveChunkedBuilder::<u32>::new(
        name,
        len as usize,
        (len as usize) * 5,
        DataType::Int64,
      );
      for idx in 0..len {
        let sub_seq: JsObject = obj.get_element(idx)?;
        let sub_seq_len = sub_seq.get_array_length()?;
        let mut v: Vec<Option<u32>> = Vec::with_capacity(sub_seq_len as usize);
        for idx in 0..sub_seq_len {
          let item: JsResult<JsNumber> = sub_seq.get_element(idx);
          match item {
            Ok(i) => v.push(Some(i.get_uint32()?)),
            _ => v.push(None),
          }
        }
        builder.append_iter(v.into_iter());
      }
      builder.finish().into_series()
    }
    DataType::Float32 => {
      let mut builder = ListPrimitiveChunkedBuilder::<f32>::new(
        name,
        len as usize,
        (len as usize) * 5,
        DataType::Int64,
      );
      for idx in 0..len {
        let sub_seq: JsObject = obj.get_element(idx)?;
        let sub_seq_len = sub_seq.get_array_length()?;
        let mut v: Vec<Option<f32>> = Vec::with_capacity(sub_seq_len as usize);
        for idx in 0..sub_seq_len {
          let item: JsResult<JsNumber> = sub_seq.get_element(idx);
          match item {
            Ok(i) => v.push(Some(i.get_double()? as f32)),
            _ => v.push(None),
          }
        }
        builder.append_iter(v.into_iter());
      }
      builder.finish().into_series()
    }
    DataType::Int64 => {
      let mut builder = ListPrimitiveChunkedBuilder::<i64>::new(
        name,
        len as usize,
        (len as usize) * 5,
        DataType::Int64,
      );
      for idx in 0..len {
        let sub_seq: JsObject = obj.get_element(idx)?;
        let sub_seq_len = sub_seq.get_array_length()?;
        let mut v: Vec<Option<i64>> = Vec::with_capacity(sub_seq_len as usize);
        for idx in 0..sub_seq_len {
          let item: JsResult<JsNumber> = sub_seq.get_element(idx);
          match item {
            Ok(i) => v.push(Some(i.get_int64()?)),
            _ => v.push(None),
          }
        }
        builder.append_iter(v.into_iter());
      }
      builder.finish().into_series()
    }
    DataType::UInt64 => {
      let mut builder = ListPrimitiveChunkedBuilder::<u64>::new(
        name,
        len as usize,
        (len as usize) * 5,
        DataType::Int64,
      );
      for idx in 0..len {
        let sub_seq: JsObject = obj.get_element(idx)?;
        let sub_seq_len = sub_seq.get_array_length()?;
        let mut v: Vec<Option<u64>> = Vec::with_capacity(sub_seq_len as usize);
        for idx in 0..sub_seq_len {
          let item: JsResult<JsNumber> = sub_seq.get_element(idx);
          match item {
            Ok(i) => v.push(Some(i.get_int64()? as u64)),
            _ => v.push(None),
          }
        }
        builder.append_iter(v.into_iter());
      }
      builder.finish().into_series()
    }
    DataType::Float64 => {
      let mut builder = ListPrimitiveChunkedBuilder::<f64>::new(
        name,
        len as usize,
        (len as usize) * 5,
        DataType::Float64,
      );
      for idx in 0..len {
        let sub_seq: JsObject = obj.get_element(idx)?;
        let sub_seq_len = sub_seq.get_array_length()?;
        let mut v: Vec<Option<f64>> = Vec::with_capacity(sub_seq_len as usize);
        for idx in 0..sub_seq_len {
          let item: JsResult<JsNumber> = sub_seq.get_element(idx);
          match item {
            Ok(i) => v.push(Some(i.get_double()?)),
            _ => v.push(None),
          }
        }
        builder.append_iter(v.into_iter());
      }
      builder.finish().into_series()
    }
    DataType::Boolean => {
      let mut builder = ListBooleanChunkedBuilder::new(name, len as usize, (len as usize) * 5);
      for idx in 0..len {
        let sub_seq: JsObject = obj.get_element(idx)?;
        let sub_seq_len = sub_seq.get_array_length()?;
        let mut v: Vec<Option<bool>> = Vec::with_capacity(sub_seq_len as usize);
        for idx in 0..sub_seq_len {
          let item: JsResult<JsBoolean> = sub_seq.get_element(idx);
          match item {
            Ok(i) => v.push(Some(i.get_value()?)),
            _ => v.push(None),
          }
        }
        builder.append_iter(v.into_iter());
      }
      builder.finish().into_series()
    }
    DataType::Utf8 => {
      let mut builder = ListUtf8ChunkedBuilder::new(name, len as usize, (len as usize) * 5);
      for idx in 0..len {
        let sub_seq: JsObject = obj.get_element(idx)?;
        let sub_seq_len = sub_seq.get_array_length()?;
        let mut v: Vec<Option<&str>> = Vec::with_capacity(sub_seq_len as usize);
        for idx in 0..sub_seq_len {
          let item: JsResult<JsUnknown> = sub_seq.get_element(idx);
          match item {
            Ok(i) => {
              let s: WrappedValue = i.into();
              let s = s.extract::<&str>()?;
              v.push(Some(s))
            }
            _ => v.push(None),
          }
        }
        builder.append_iter(v.into_iter());
      }
      builder.finish().into_series()
    }
    dt => {
      panic!("cannot create list array from {:?}", dt);
    }
  };
  Ok(s)
}

pub fn from_typed_array(arr: &JsTypedArrayValue) -> JsResult<Series> {
  let dtype: JsDataType = arr.typedarray_type.into();
  let series = match dtype {
    JsDataType::Int8 => {
      let v: &[i8] = arr.as_ref();
      let v: Vec<i8> = v.to_owned();
      let v: ChunkedArray<UInt32Type> = v.into_iter().map(|v| Some(v as u32)).collect();
      v.into_series()
    }
    JsDataType::UInt8 => {
      let v: &[u8] = arr.as_ref();
      let v: Vec<u8> = v.to_owned();
      let v: ChunkedArray<UInt32Type> = v.into_iter().map(|v| Some(v as u32)).collect();
      v.into_series()
    }
    JsDataType::Int16 => {
      let v: &[i16] = arr.as_ref();
      let v: Vec<i16> = v.to_owned();
      let v: ChunkedArray<UInt32Type> = v.into_iter().map(|v| Some(v as u32)).collect();
      v.into_series()
    }
    JsDataType::UInt16 => {
      let v: &[u16] = arr.as_ref();
      let v: Vec<u16> = v.to_owned();
      let v: ChunkedArray<UInt32Type> = v.into_iter().map(|v| Some(v as u32)).collect();
      v.into_series()
    }
    JsDataType::Int32 => {
      let v: &[i32] = arr.as_ref();
      let v: Vec<i32> = v.to_owned();
      let v: ChunkedArray<Int32Type> = v.into_iter().map(Some).collect();
      v.into_series()
    }
    JsDataType::UInt32 => {
      let v: &[u32] = arr.as_ref();
      let v: Vec<u32> = v.to_owned();
      let v: ChunkedArray<UInt32Type> = v.into_iter().map(Some).collect();
      v.into_series()
    }
    JsDataType::Float32 => {
      let v: &[f32] = arr.as_ref();
      let v: Vec<f32> = v.to_owned();
      let v: ChunkedArray<Float32Type> = v.into_iter().map(Some).collect();
      v.into_series()
    }
    JsDataType::Float64 => {
      let v: &[f64] = arr.as_ref();
      let v: Vec<f64> = v.to_owned();
      let v: ChunkedArray<Float64Type> = v.into_iter().map(Some).collect();
      v.into_series()
    }
    JsDataType::Int64 => {
      let v: &[i64] = arr.as_ref();
      let v: Vec<i64> = v.to_owned();
      let v: ChunkedArray<Int64Type> = v.into_iter().map(Some).collect();
      v.into_series()
    }
    JsDataType::UInt64 => {
      let v: &[u64] = arr.as_ref();
      let v: Vec<u64> = v.to_owned();
      let v: ChunkedArray<UInt64Type> = v.into_iter().map(Some).collect();
      v.into_series()
    }
    _ => {
      panic!("cannot create series from");
    }
  };

  Ok(series)
}


