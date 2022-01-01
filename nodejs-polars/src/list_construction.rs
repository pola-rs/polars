use crate::conversion::prelude::*;
use crate::prelude::*;
use napi::{JsBoolean, JsNumber, JsObject, JsString, JsTypedArrayValue, JsUnknown};
use polars::chunked_array::ChunkedArray;

macro_rules! typed_to_chunked {
    ($arr:expr, $type:ty, $pl_type:ty) => {{
        let v: &[$type] = $arr.as_ref();
        let mut buffer = Vec::<$type>::new();
        buffer.extend_from_slice(v);
        ChunkedArray::<$pl_type>::new_from_aligned_vec("", buffer)
    }};
}

pub fn js_arr_to_list(name: &str, obj: &JsObject, dtype: &DataType) -> JsResult<Series> {
    let len = obj.get_array_length()?;
    let s = match dtype {
        DataType::Int8 => {
            let mut builder = ListPrimitiveChunkedBuilder::<i8>::new(
                name,
                len as usize,
                (len as usize) * 5,
                DataType::Int8,
            );
            for idx in 0..len {
                let sub_seq: JsObject = obj.get_element(idx)?;
                if sub_seq.is_typedarray()? {
                    let buff: napi::JsTypedArray = unsafe { sub_seq.into_unknown().cast() };
                    let v = buff.into_value()?;
                    let ca = typed_to_chunked!(v, i8, Int8Type);
                    builder.append_iter(ca.into_iter())
                } else {
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
            }
            builder.finish().into_series()
        }
        DataType::UInt8 => {
            let mut builder = ListPrimitiveChunkedBuilder::<u8>::new(
                name,
                len as usize,
                (len as usize) * 5,
                DataType::UInt8,
            );
            for idx in 0..len {
                let sub_seq: JsObject = obj.get_element(idx)?;
                if sub_seq.is_typedarray()? {
                    let buff: napi::JsTypedArray = unsafe { sub_seq.into_unknown().cast() };
                    let v = buff.into_value()?;
                    let ca = typed_to_chunked!(v, u8, UInt8Type);
                    builder.append_iter(ca.into_iter())
                } else {
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
            }
            builder.finish().into_series()
        }
        DataType::Int16 => {
            let mut builder = ListPrimitiveChunkedBuilder::<i16>::new(
                name,
                len as usize,
                (len as usize) * 5,
                DataType::Int16,
            );
            for idx in 0..len {
                let sub_seq: JsObject = obj.get_element(idx)?;
                if sub_seq.is_typedarray()? {
                    let buff: napi::JsTypedArray = unsafe { sub_seq.into_unknown().cast() };
                    let v = buff.into_value()?;
                    let ca = typed_to_chunked!(v, i16, Int16Type);
                    builder.append_iter(ca.into_iter())
                } else {
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
            }
            builder.finish().into_series()
        }
        DataType::UInt16 => {
            let mut builder = ListPrimitiveChunkedBuilder::<u16>::new(
                name,
                len as usize,
                (len as usize) * 5,
                DataType::UInt16,
            );
            for idx in 0..len {
                let sub_seq: JsObject = obj.get_element(idx)?;
                if sub_seq.is_typedarray()? {
                    let buff: napi::JsTypedArray = unsafe { sub_seq.into_unknown().cast() };
                    let v = buff.into_value()?;
                    let ca = typed_to_chunked!(v, u16, UInt16Type);
                    builder.append_iter(ca.into_iter())
                } else {
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
            }
            builder.finish().into_series()
        }
        DataType::Int32 => {
            let mut builder = ListPrimitiveChunkedBuilder::<i32>::new(
                name,
                len as usize,
                (len as usize) * 5,
                DataType::Int32,
            );
            for idx in 0..len {
                let sub_seq: JsObject = obj.get_element(idx)?;
                if sub_seq.is_typedarray()? {
                    let buff: napi::JsTypedArray = unsafe { sub_seq.into_unknown().cast() };
                    let v = buff.into_value()?;
                    let ca = typed_to_chunked!(v, i32, Int32Type);
                    builder.append_iter(ca.into_iter())
                } else {
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
            }
            builder.finish().into_series()
        }
        DataType::UInt32 => {
            let mut builder = ListPrimitiveChunkedBuilder::<u32>::new(
                name,
                len as usize,
                (len as usize) * 5,
                DataType::UInt32,
            );
            for idx in 0..len {
                let sub_seq: JsObject = obj.get_element(idx)?;
                if sub_seq.is_typedarray()? {
                    let buff: napi::JsTypedArray = unsafe { sub_seq.into_unknown().cast() };
                    let v = buff.into_value()?;
                    let ca = typed_to_chunked!(v, u32, UInt32Type);
                    builder.append_iter(ca.into_iter())
                } else {
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
            }
            builder.finish().into_series()
        }
        DataType::Float32 => {
            let mut builder = ListPrimitiveChunkedBuilder::<f32>::new(
                name,
                len as usize,
                (len as usize) * 5,
                DataType::Float32,
            );
            for idx in 0..len {
                let sub_seq: JsObject = obj.get_element(idx)?;
                if sub_seq.is_typedarray()? {
                    let buff: napi::JsTypedArray = unsafe { sub_seq.into_unknown().cast() };
                    let v = buff.into_value()?;
                    let ca = typed_to_chunked!(v, f32, Float32Type);
                    builder.append_iter(ca.into_iter())
                } else {
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
                if sub_seq.is_typedarray()? {
                    let buff: napi::JsTypedArray = unsafe { sub_seq.into_unknown().cast() };
                    let v = buff.into_value()?;
                    let ca = typed_to_chunked!(v, i64, Int64Type);
                    builder.append_iter(ca.into_iter())
                } else {
                    let sub_seq: JsObject = obj.get_element(idx)?;
                    let sub_seq_len = sub_seq.get_array_length()?;
                    let mut v: Vec<Option<i64>> = Vec::with_capacity(sub_seq_len as usize);
                    for idx in 0..sub_seq_len {
                        let item: JsResult<napi::JsBigint> = sub_seq.get_element(idx);
                        match item {
                            Ok(i) => v.push(Some(i.try_into()?)),
                            _ => v.push(None),
                        }
                    }
                    builder.append_iter(v.into_iter());
                }
            }
            builder.finish().into_series()
        }
        DataType::UInt64 => {
            let mut builder = ListPrimitiveChunkedBuilder::<u64>::new(
                name,
                len as usize,
                (len as usize) * 5,
                DataType::UInt64,
            );
            for idx in 0..len {
                let sub_seq: JsObject = obj.get_element(idx)?;
                if sub_seq.is_typedarray()? {
                    let buff: napi::JsTypedArray = unsafe { sub_seq.into_unknown().cast() };
                    let v = buff.into_value()?;
                    let s = from_typed_array(&v)?;
                    builder.append_series(&s)
                } else {
                    let sub_seq: JsObject = obj.get_element(idx)?;
                    let sub_seq_len = sub_seq.get_array_length()?;
                    let mut v: Vec<Option<u64>> = Vec::with_capacity(sub_seq_len as usize);
                    for idx in 0..sub_seq_len {
                        let item: JsResult<napi::JsBigint> = sub_seq.get_element(idx);
                        match item {
                            Ok(i) => v.push(Some(i.try_into()?)),
                            _ => v.push(None),
                        }
                    }
                    builder.append_iter(v.into_iter());
                }
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
                if sub_seq.is_typedarray()? {
                    let buff: napi::JsTypedArray = unsafe { sub_seq.into_unknown().cast() };
                    let v = buff.into_value()?;
                    let s = from_typed_array(&v)?;
                    builder.append_series(&s)
                } else {
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
            }
            builder.finish().into_series()
        }
        DataType::Boolean => {
            let mut builder =
                ListBooleanChunkedBuilder::new(name, len as usize, (len as usize) * 5);
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
                    let item: JsResult<JsString> = sub_seq.get_element(idx);
                    match item {
                        Ok(i) => {
                            let s = i.into_utf8()?.into_owned()?;
                            let s = Box::leak(s.into_boxed_str());
                            v.push(Some(s))
                        }
                        _ => v.push(None),
                    }
                }
                builder.append_iter(v.into_iter());
            }
            builder.finish().into_series()
        }
        DataType::Datetime(_, _) => {
            let mut builder = ListPrimitiveChunkedBuilder::<i64>::new(
                name,
                len as usize,
                (len as usize) * 5,
                DataType::Int64,
            );
            for idx in 0..len {
                let sub_seq: JsObject = obj.get_element(idx)?;
                let sub_seq_len = sub_seq.get_array_length()?;
                let mut inner_builder =
                    PrimitiveChunkedBuilder::<Int64Type>::new(name, sub_seq_len as usize);
                for inner_idx in 0..sub_seq_len {
                    let item: JsResult<JsObject> = sub_seq.get_element(inner_idx);
                    match item {
                        Ok(obj) => {
                            if obj.is_date()? {
                                let d: &napi::JsDate = unsafe { &obj.into_unknown().cast() };
                                match d.value_of() {
                                    Ok(v) => inner_builder.append_value(v as i64),
                                    Err(_) => inner_builder.append_null(),
                                }
                            }
                        }
                        Err(_) => inner_builder.append_null(),
                    }
                }
                let dt_series = inner_builder
                    .finish()
                    .into_series()
                    .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                    .map_err(JsPolarsEr::from)?;
                builder.append_series(&dt_series);
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
        JsDataType::Int8 => typed_to_chunked!(arr, i8, Int8Type).into(),
        JsDataType::UInt8 => typed_to_chunked!(arr, u8, UInt8Type).into(),
        JsDataType::Int16 => typed_to_chunked!(arr, i16, Int16Type).into(),
        JsDataType::UInt16 => typed_to_chunked!(arr, u16, UInt16Type).into(),
        JsDataType::Int32 => typed_to_chunked!(arr, i32, Int32Type).into(),
        JsDataType::UInt32 => typed_to_chunked!(arr, u32, UInt32Type).into(),
        JsDataType::Float32 => typed_to_chunked!(arr, f32, Float32Type).into(),
        JsDataType::Float64 => typed_to_chunked!(arr, f64, Float64Type).into(),
        JsDataType::Int64 => typed_to_chunked!(arr, i64, Int64Type).into(),
        JsDataType::UInt64 => typed_to_chunked!(arr, u64, UInt64Type).into(),
        _ => panic!("cannot create series from"),
    };

    Ok(series)
}

macro_rules! arr_to_series {
    ($len:expr, $arr:expr, $type:ty) => {{
        let v: Vec<Option<$type>> = (0..$len)
            .map(|idx| {
                let val: WrappedValue = $arr
                    .get_element::<JsUnknown>(idx)
                    .expect("downcast err")
                    .into();
                match val.extract::<$type>() {
                    Ok(val) => Some(val),
                    Err(_) => {
                        panic!("unable to downcast value, multi type lists are not supported")
                    }
                }
            })
            .collect();
        Series::new("", v)
    }};
}

pub fn arr_to_list(obj: &JsObject, dtype: DataType) -> JsResult<Series> {
    let len = obj.get_array_length()?;
    let s = match dtype {
        DataType::Int64 => arr_to_series!(len, obj, i64),
        DataType::Float64 => arr_to_series!(len, obj, f64),
        DataType::Utf8 => arr_to_series!(len, obj, &str),
        DataType::Boolean => arr_to_series!(len, obj, bool),
        // cast other values to int32
        _ => arr_to_series!(len, obj, i32),
    };
    Ok(s)
}
