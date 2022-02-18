use crate::conversion::prelude::*;
use crate::datatypes::JsDataType;
use crate::error::JsPolarsEr;
use crate::list_construction::from_typed_array;
use crate::list_construction::js_arr_to_list;
use crate::prelude::JsResult;
use napi::Either;
use napi::JsBoolean;
use napi::JsExternal;
use napi::JsNumber;
use napi::JsTypedArray;
use napi::JsUndefined;
use napi::JsUnknown;
use napi::{CallContext, JsNull, JsObject, JsString, ValueType};
use polars::prelude::*;
use polars_core::utils::CustomIterTools;

pub struct JsSeries {}

impl IntoJs<JsExternal> for Series {
    fn try_into_js(self, cx: &CallContext) -> JsResult<JsExternal> {
        cx.env.create_external(self, None)
    }
}

#[js_function(1)]
pub fn new_from_typed_array(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let name = params.get_as::<&str>("name")?;
    let buff: JsTypedArray = params.0.get_named_property("values")?;
    let v = buff.into_value()?;
    let mut series = from_typed_array(&v)?;
    series.rename(name);
    series.try_into_js(&cx)
}

#[js_function(1)]
pub fn new_bool(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let name = params.get_as::<&str>("name")?;
    let items = params.get_as::<Vec<bool>>("values")?;
    Series::new(name, items).try_into_js(&cx)
}

#[js_function(1)]
pub fn new_opt_date(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let name = params.get_as::<&str>("name")?;
    let strict = params.get_as::<Option<bool>>("strict")?;
    let (arr, len) = params.get_arr("values")?;
    let mut builder = PrimitiveChunkedBuilder::<Int64Type>::new(name, len);
    for item in arr.into_iter() {
        match item.0.get_type()? {
            ValueType::Object => {
                let obj: &JsObject = unsafe { &item.0.cast() };
                if obj.is_date()? {
                    let d: &napi::JsDate = unsafe { &item.0.cast() };
                    match d.value_of() {
                        Ok(v) => builder.append_value(v as i64),
                        Err(e) => {
                            if strict.unwrap_or(false) {
                                return Err(e);
                            }
                            builder.append_null()
                        }
                    }
                }
            }
            ValueType::Null | ValueType::Undefined => builder.append_null(),
            _ => return Err(JsPolarsEr::Other("Series must be of date type".to_owned()).into()),
        }
    }
    let ca: ChunkedArray<Int64Type> = builder.finish();
    ca.into_datetime(TimeUnit::Milliseconds, None)
        .into_series()
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn new_opt_bool(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let name = params.get_as::<&str>("name")?;
    let (items, len) = params.get_arr("values")?;
    let mut builder = BooleanChunkedBuilder::new(name, len);

    for item in items.into_iter() {
        match item.0.get_type()? {
            ValueType::Boolean => {
                let b = item.extract::<bool>()?;
                builder.append_value(b)
            }
            ValueType::Null => builder.append_null(),
            ValueType::Undefined => builder.append_null(),
            _ => return Err(JsPolarsEr::Other("Series must be of boolean type".to_owned()).into()),
        }
    }

    let ca: ChunkedArray<BooleanType> = builder.finish();
    ca.into_series().try_into_js(&cx)
}

#[js_function(1)]
pub fn new_str(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let name = params.get_as::<&str>("name")?;
    let val = params.get_as::<Wrap<Utf8Chunked>>("values")?;
    let mut s = val.0.into_series();
    s.rename(name);
    s.try_into_js(&cx)
}

#[js_function(1)]
pub fn new_list(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let name = params.get_as::<String>("name")?;
    let values = params.get::<JsObject>("values")?;
    let dtype: DataType = params.get_as::<JsDataType>("dtype")?.into();
    js_arr_to_list(&name, &values, &dtype)?.try_into_js(&cx)
}

#[js_function(1)]
pub fn new_object(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series: Series = cx.env.from_js_value(&params.0)?;
    series.try_into_js(&cx)
}

#[js_function(1)]
pub fn repeat(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let name = params.get_as::<&str>("name")?;
    let val: WrappedValue = params.get::<JsUnknown>("value")?.into();
    let n = params.get_as::<usize>("n")?;
    let dtype: DataType = params.get_as::<JsDataType>("dtype")?.into();
    let s: Series = match dtype {
        DataType::Utf8 => {
            let val = val.extract::<&str>().unwrap();
            let mut ca: Utf8Chunked = (0..n).map(|_| val).collect_trusted();
            ca.rename(name);
            ca.into_series()
        }
        DataType::Int64 => {
            let val = val.extract::<i64>().unwrap();
            let mut ca: NoNull<Int64Chunked> = (0..n).map(|_| val).collect_trusted();
            ca.rename(name);
            ca.into_inner().into_series()
        }
        DataType::Float64 => {
            let val = val.extract::<Option<f64>>().unwrap_or(None);
            let mut ca: Float64Chunked = (0..n).map(|_| val).collect_trusted();
            ca.rename(name);
            ca.into_series()
        }
        DataType::Boolean => {
            let val = val.extract::<bool>().unwrap();
            let mut ca: BooleanChunked = (0..n).map(|_| val).collect_trusted();
            ca.rename(name);
            ca.into_series()
        }
        DataType::Datetime(_, _) => {
            let val = val.extract::<i64>().unwrap();
            let mut ca: NoNull<Int64Chunked> = (0..n).map(|_| val).collect_trusted();
            ca.rename(name);
            ca.clone()
                .into_datetime(TimeUnit::Milliseconds, None)
                .into_series()
        }
        dt => {
            panic!("cannot create repeat with dtype: {:?}", dt);
        }
    };
    s.try_into_js(&cx)
}

#[js_function(1)]
pub fn dtype(cx: CallContext) -> JsResult<JsString> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let dt: JsDataType = series.dtype().into();
    let s = dt.to_str();
    cx.env.create_string(s)
}

#[js_function(1)]
pub(crate) fn get_fmt(cx: CallContext) -> JsResult<JsString> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let s = format!("{}", series);
    cx.env.create_string(&s)
}

#[js_function(1)]
pub fn abs(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    series.abs().map_err(JsPolarsEr::from)?.try_into_js(&cx)
}

#[js_function(1)]
pub fn chunk_lengths(cx: CallContext) -> JsResult<JsObject> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let mut arr = cx.env.create_array()?;
    for (idx, chunk_len) in series.chunk_lengths().enumerate() {
        let js_num = cx.env.create_int64(chunk_len as i64)?;
        arr.set_element(idx as u32, js_num)?;
    }
    Ok(arr)
}
#[js_function(1)]
pub fn name(cx: CallContext) -> JsResult<JsString> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    series.name().try_into_js(&cx)
}

#[js_function(1)]
pub fn rename(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let series = params.get_external_mut::<Series>(&cx, "_series")?;
    let name = params.get_as::<&str>("name")?;
    series.rename(name);
    cx.env.get_undefined()
}

#[js_function(1)]
pub fn get_idx(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let idx = params.get_as::<usize>("idx")?;
    let val: AnyValue = series.get(idx);
    val.try_into_js(&cx)
}

#[js_function(1)]
pub fn bitand(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let other = params.get_external::<Series>(&cx, "other")?;
    series
        .bitand(other)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn bitor(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let other = params.get_external::<Series>(&cx, "other")?;
    series
        .bitor(other)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn bitxor(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let other = params.get_external::<Series>(&cx, "other")?;
    series
        .bitxor(other)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn mean(cx: CallContext) -> JsResult<JsNumber> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let mean = match series.dtype() {
        DataType::Boolean => {
            let s = series.cast(&DataType::UInt8).unwrap();
            s.mean()
        }
        _ => series.mean(),
    }
    .unwrap();
    cx.env.create_double(mean)
}

#[js_function(1)]
pub fn max(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    match series.dtype() {
        DataType::Float32 | DataType::Float64 => {
            let max = series.max::<f64>().unwrap();
            cx.env.create_double(max).map(|v| v.into_unknown())
        }
        DataType::Boolean => {
            let max = series.max::<u32>().map(|v| v == 1).unwrap();
            cx.env.get_boolean(max).map(|v| v.into_unknown())
        }
        _ => {
            let max = series.max::<i64>().unwrap();
            cx.env.create_int64(max).map(|v| v.into_unknown())
        }
    }
}

#[js_function(1)]
pub fn min(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    match series.dtype() {
        DataType::Float32 | DataType::Float64 => {
            let min = series.min::<f64>().unwrap();
            cx.env.create_double(min).map(|v| v.into_unknown())
        }
        DataType::Boolean => {
            let min = series.min::<u32>().map(|v| v == 1).unwrap();
            cx.env.get_boolean(min).map(|v| v.into_unknown())
        }
        _ => {
            let min = series.min::<i64>().unwrap();
            cx.env.create_int64(min).map(|v| v.into_unknown())
        }
    }
}

#[js_function(1)]
pub fn median(cx: CallContext) -> JsResult<Either<JsNumber, JsUndefined>> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let median = match series.dtype() {
        DataType::Boolean => {
            let s = series.cast(&DataType::UInt8).unwrap();
            s.median()
        }
        _ => series.median(),
    };
    match median {
        Some(m) => Ok(Either::A(cx.env.create_double(m)?)),
        None => Ok(Either::B(cx.env.get_undefined()?)),
    }
}

#[js_function(1)]
pub fn sum(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    match series.dtype() {
        DataType::Float32 | DataType::Float64 => {
            let sum = series.sum::<f64>().unwrap();
            cx.env.create_double(sum).map(|v| v.into_unknown())
        }
        DataType::Boolean => {
            let sum = series.sum::<i32>().unwrap();
            cx.env.create_int32(sum).map(|v| v.into_unknown())
        }
        _ => {
            let sum = series.sum::<i64>().unwrap();
            cx.env.create_int64(sum).map(|v| v.into_unknown())
        }
    }
}

#[js_function(1)]
pub fn slice(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let offset = params.get_as::<i64>("offset")?;
    let length = params.get_as::<usize>("length")?;
    series.slice(offset, length).try_into_js(&cx)
}

#[js_function(1)]
pub fn append(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let series = params.get_external_mut::<Series>(&cx, "_series")?;
    let other = params.get_external::<Series>(&cx, "other")?;
    series.append(other).map_err(JsPolarsEr::from)?;
    cx.env.get_undefined()
}

#[js_function(1)]
pub fn filter(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let filter = params.get_external::<Series>(&cx, "filter")?;
    if let Ok(ca) = filter.bool() {
        series
            .filter(ca)
            .map_err(JsPolarsEr::from)?
            .try_into_js(&cx)
    } else {
        Err(JsPolarsEr::Other(String::from("Expected a boolean mask")).into())
    }
}

#[js_function(1)]
pub fn add(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let other = params.get_external::<Series>(&cx, "other")?;
    (series + other).try_into_js(&cx)
}

#[js_function(1)]
pub fn sub(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let other = params.get_external::<Series>(&cx, "other")?;
    (series - other).try_into_js(&cx)
}

#[js_function(1)]
pub fn mul(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let other = params.get_external::<Series>(&cx, "other")?;
    (series * other).try_into_js(&cx)
}

#[js_function(1)]
pub fn div(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let other = params.get_external::<Series>(&cx, "other")?;
    (series / other).try_into_js(&cx)
}

#[js_function(1)]
pub fn rem(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let other = params.get_external::<Series>(&cx, "other")?;
    (series % other).try_into_js(&cx)
}

#[js_function(1)]
pub fn sort_in_place(_: CallContext) -> JsResult<JsExternal> {
    todo!()
}

#[js_function(1)]
pub fn value_counts(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    series
        .value_counts()
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn arg_min(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let arg_min = series.arg_min();
    match arg_min {
        Some(v) => cx.env.create_uint32(v as u32).map(|v| v.into_unknown()),
        None => cx.env.get_undefined().map(|v| v.into_unknown()),
    }
}

#[js_function(1)]
pub fn arg_max(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let arg_max = series.arg_max();
    match arg_max {
        Some(v) => cx.env.create_uint32(v as u32).map(|v| v.into_unknown()),
        None => cx.env.get_undefined().map(|v| v.into_unknown()),
    }
}

#[js_function(1)]
pub fn take(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let indices = params.get_as::<Vec<u32>>("indices")?;
    let chunked: UInt32Chunked = indices.into_iter().map(Some).collect();
    series
        .take(&chunked)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn take_with_series(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let indices = params.get_external::<Series>(&cx, "indices")?;
    let idx = indices.u32().map_err(JsPolarsEr::from)?;
    series.take(idx).map_err(JsPolarsEr::from)?.try_into_js(&cx)
}

#[js_function(1)]
pub fn series_equal(cx: CallContext) -> JsResult<JsBoolean> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let other = params.get_external::<Series>(&cx, "other")?;
    let null_equal = params.get_as::<bool>("null_equal")?;
    let b = if null_equal {
        series.series_equal_missing(other)
    } else {
        series.series_equal(other)
    };
    cx.env.get_boolean(b)
}

#[js_function(1)]
pub fn not(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let b = series.bool().map_err(JsPolarsEr::from)?;
    (!b).into_series().try_into_js(&cx)
}

#[js_function(1)]
pub fn as_str(cx: CallContext) -> JsResult<JsString> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let s = format!("{:?}", series);
    cx.env.create_string(&s)
}

#[js_function(1)]
pub fn len(cx: CallContext) -> JsResult<JsNumber> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let len = series.len();
    cx.env.create_int64(len as i64)
}

#[js_function(1)]
pub fn quantile(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let quantile = params.get_as::<f64>("quantile")?;
    let q = series
        .quantile_as_series(quantile, QuantileInterpolOptions::default())
        .map_err(JsPolarsEr::from)?;
    q.get(0).try_into_js(&cx)
}

#[js_function(1)]
pub fn fill_null(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let strategy = params.get_as::<String>("strategy")?;
    let strat = parse_strategy(strategy);
    series
        .fill_null(strat)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn map(_cx: CallContext) -> JsResult<JsExternal> {
    todo!()
}

#[js_function(1)]
pub fn zip_with(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let mask = params.get_external::<Series>(&cx, "mask")?;
    let other = params.get_external::<Series>(&cx, "other")?;
    let mask = mask.bool().map_err(JsPolarsEr::from)?;
    series
        .zip_with(mask, other)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn str_parse_date(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let fmt = params.get_as::<Option<&str>>("fmt")?;
    if let Ok(ca) = series.utf8() {
        ca.as_date(fmt)
            .map_err(JsPolarsEr::from)?
            .into_series()
            .try_into_js(&cx)
    } else {
        Err(JsPolarsEr::Other("cannot parse Date expected utf8 type".into()).into())
    }
}

#[js_function(1)]
pub fn str_parse_datetime(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let fmt = params.get_as::<Option<&str>>("fmt")?;
    if let Ok(ca) = series.utf8() {
        ca.as_datetime(fmt, TimeUnit::Milliseconds)
            .map_err(JsPolarsEr::from)?
            .into_series()
            .try_into_js(&cx)
    } else {
        Err(JsPolarsEr::Other("cannot parse Date expected utf8 type".into()).into())
    }
}
#[js_function(1)]
pub fn arr_lengths(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let ca = series.list().map_err(JsPolarsEr::from)?;
    let s: Series = ca.clone().into_series();
    s.try_into_js(&cx)
}

#[js_function(1)]
pub fn to_dummies(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let _series = params.get_external::<Series>(&cx, "_series")?;
    todo!()
}

#[js_function(1)]
pub fn get_list(cx: CallContext) -> JsResult<Either<JsExternal, JsNull>> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let index = params.get_as::<usize>("index")?;

    if let Ok(ca) = series.list() {
        let s = ca.get(index);
        if let Some(item) = s {
            item.try_into_js(&cx).map(Either::A)
        } else {
            cx.env.get_null().map(Either::B)
        }
    } else {
        cx.env.get_null().map(Either::B)
    }
}

#[js_function(1)]
pub fn shrink_to_fit(_: CallContext) -> JsResult<JsUnknown> {
    todo!()
}

#[js_function(1)]
pub fn to_array(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let series: &Series = params.get_external::<Series>(&cx, "_series")?;

    match *series.dtype() {
        DataType::UInt8 => cx.env.to_js_value(series.u8().unwrap()),
        DataType::UInt16 => cx.env.to_js_value(series.u16().unwrap()),
        DataType::UInt32 => cx.env.to_js_value(series.u32().unwrap()),
        DataType::UInt64 => cx.env.to_js_value(series.u64().unwrap()),
        DataType::Int8 => cx.env.to_js_value(series.i8().unwrap()),
        DataType::Int16 => cx.env.to_js_value(series.i16().unwrap()),
        DataType::Int32 => cx.env.to_js_value(series.i32().unwrap()),
        DataType::Int64 => cx.env.to_js_value(series.i64().unwrap()),
        DataType::Float32 => cx.env.to_js_value(series.f32().unwrap()),
        DataType::Float64 => cx.env.to_js_value(series.f64().unwrap()),
        DataType::Utf8 => cx.env.to_js_value(series.utf8().unwrap()),
        DataType::Date => cx.env.to_js_value(series.date().unwrap()),
        DataType::Datetime(_, _) => cx.env.to_js_value(series.datetime().unwrap()),
        DataType::List(_) => cx.env.to_js_value(series.list().unwrap()),
        DataType::Categorical => cx.env.to_js_value(series.categorical().unwrap()),
        _ => todo!(),
    }
}

#[js_function(1)]
pub fn is_in(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let other = params.get_external::<Series>(&cx, "other")?;
    series
        .is_in(other)
        .map_err(JsPolarsEr::from)?
        .into_series()
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn dot(cx: CallContext) -> JsResult<Either<JsNumber, JsNull>> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let other = params.get_external::<Series>(&cx, "other")?;
    let dot = series.dot(other);
    match dot {
        Some(d) => cx.env.create_double(d).map(Either::A),
        None => cx.env.get_null().map(Either::B),
    }
}
#[js_function(1)]
pub fn hash(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let k0 = params.get_as::<u64>("k0")?;
    let k1 = params.get_as::<u64>("k1")?;
    let k2 = params.get_as::<u64>("k2")?;
    let k3 = params.get_as::<u64>("k3")?;
    let hb = ahash::RandomState::with_seeds(k0, k1, k2, k3);
    series.hash(hb).into_series().try_into_js(&cx)
}

#[js_function(1)]
pub fn reinterpret(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let signed = params.get_as::<bool>("signed")?;
    let s = match (series.dtype(), signed) {
        (DataType::UInt64, true) => {
            let ca = series.u64().unwrap();
            Ok(ca.reinterpret_signed().into_series())
        }
        (DataType::UInt64, false) => Ok(series.clone()),
        (DataType::Int64, false) => {
            let ca = series.i64().unwrap();
            Ok(ca.reinterpret_unsigned().into_series())
        }
        (DataType::Int64, true) => Ok(series.clone()),
        _ => Err(PolarsError::ComputeError(
            "reinterpret is only allowed for 64bit integers dtype, use cast otherwise".into(),
        )),
    }
    .map_err(JsPolarsEr::from)?;
    s.try_into_js(&cx)
}

#[js_function(1)]
pub fn rank(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let method = params.get_as::<String>("method")?;
    let method = str_to_rankmethod(method)?;
    series
        .rank(RankOptions {
            method,
            descending: false,
        })
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn diff(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let n = params.get_as::<usize>("n")?;
    let null_behavior = params.get_as::<String>("null_behavior")?;
    let null_behavior = str_to_null_behavior(null_behavior)?;
    series.diff(n, null_behavior).try_into_js(&cx)
}

#[js_function(1)]
pub fn cast(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let dtype: DataType = params.get_as::<JsDataType>("dtype")?.into();
    let strict = params.get_as::<bool>("strict")?;
    let out = if strict {
        series.strict_cast(&dtype)
    } else {
        series.cast(&dtype)
    };
    out.map_err(JsPolarsEr::from)?.try_into_js(&cx)
}

#[js_function(1)]
pub fn skew(cx: CallContext) -> JsResult<Either<JsNumber, JsNull>> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let bias = params.get_as::<bool>("bias")?;
    let skew = series.skew(bias).map_err(JsPolarsEr::from)?;
    match skew {
        Some(skew) => cx.env.create_double(skew).map(Either::A),
        None => cx.env.get_null().map(Either::B),
    }
}
#[js_function(1)]
pub fn kurtosis(cx: CallContext) -> JsResult<Either<JsNumber, JsNull>> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let fisher = params.get_as::<bool>("fisher")?;
    let bias = params.get_as::<bool>("bias")?;
    let kurtosis = series.kurtosis(fisher, bias).map_err(JsPolarsEr::from)?;
    match kurtosis {
        Some(k) => cx.env.create_double(k).map(Either::A),
        None => cx.env.get_null().map(Either::B),
    }
}

#[js_function(1)]
pub fn get_str(cx: CallContext) -> JsResult<Either<JsString, JsNull>> {
    let params = get_params(&cx)?;
    let series = params.get_external_mut::<Series>(&cx, "_series")?;
    let index = params.get_as::<i64>("index")?;
    if let Ok(ca) = series.utf8() {
        let index = if index < 0 {
            (ca.len() as i64 + index) as usize
        } else {
            index as usize
        };
        match ca.get(index) {
            Some(s) => s.try_into_js(&cx).map(Either::A),
            None => cx.env.get_null().map(Either::B),
        }
    } else {
        cx.env.get_null().map(Either::B)
    }
}

#[js_function(1)]
pub fn argsort(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let val = params.get_as::<bool>("reverse")?;

    series.argsort(val).into_series().try_into_js(&cx)
}
#[js_function(1)]
pub fn n_chunks(cx: CallContext) -> JsResult<JsNumber> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let n = series.n_chunks();
    cx.env.create_uint32(n as u32)
}

#[js_function(1)]
pub fn null_count(cx: CallContext) -> JsResult<JsNumber> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let n = series.null_count();
    cx.env.create_uint32(n as u32)
}

#[js_function(1)]
pub fn has_validity(cx: CallContext) -> JsResult<JsBoolean> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let n = series.has_validity();
    cx.env.get_boolean(n)
}
#[js_function(1)]
pub fn get_date(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let index = params.get_as::<i64>("index")?;
    match series.date() {
        Ok(ca) => {
            let index = if index < 0 {
                (ca.len() as i64 + index) as usize
            } else {
                index as usize
            };
            match ca.get(index) {
                Some(v) => cx.env.create_date(v as f64).map(|v| v.into_unknown()),
                None => cx.env.get_null().map(|v| v.into_unknown()),
            }
        }
        Err(_) => cx.env.get_null().map(|v| v.into_unknown()),
    }
}

#[js_function(1)]
pub fn get_datetime(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let index = params.get_as::<i64>("index")?;
    match series.datetime() {
        Ok(ca) => {
            let index = if index < 0 {
                (ca.len() as i64 + index) as usize
            } else {
                index as usize
            };
            match ca.get(index) {
                Some(v) => cx.env.create_date(v as f64).map(|v| v.into_unknown()),
                None => cx.env.get_null().map(|v| v.into_unknown()),
            }
        }
        Err(_) => cx.env.get_null().map(|v| v.into_unknown()),
    }
}

#[js_function(1)]
pub fn to_js(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let obj: JsUnknown = cx.env.to_js_value(series)?;
    Ok(obj)
}

#[js_function(1)]
pub fn to_json(cx: CallContext) -> JsResult<napi::JsBuffer> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let buf = serde_json::to_vec(series).unwrap();

    let bytes = cx.env.create_buffer_with_data(buf).unwrap();
    let js_buff = bytes.into_raw();
    Ok(js_buff)
}

#[js_function(1)]
pub fn rechunk(cx: CallContext) -> JsResult<Either<JsExternal, JsUndefined>> {
    let params = get_params(&cx)?;
    let series = params.get_external_mut::<Series>(&cx, "_series")?;
    let in_place = params.get_as::<Option<bool>>("inPlace")?;
    let rechunked_series = series.rechunk();
    if in_place.unwrap_or(false) {
        cx.env.get_undefined().map(Either::B)
    } else {
        rechunked_series.try_into_js(&cx).map(Either::A)
    }
}

#[js_function(1)]
pub fn extend_constant(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external_mut::<Series>(&cx, "_series")?;
    let val = params.get_as::<AnyValue>("value")?;
    let n = params.get_as::<usize>("n")?;
    series
        .extend_constant(val, n)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn extend(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let series: &mut Series = params.get_external_mut::<Series>(&cx, "_series")?;
    let other = params.get_external::<Series>(&cx, "other")?;
    series.extend(other).map_err(JsPolarsEr::from)?;
    cx.env.get_undefined()
}

macro_rules! init_method {
    ($name:ident, $js_type:ty, $type:ty, $getter:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let name = params.get_as::<&str>("name")?;
            let arr = params.get::<JsObject>("values")?;
            let len = arr.get_array_length()?;
            let mut items: Vec<$type> = Vec::with_capacity(len as usize);

            for i in 0..len {
                let item: $js_type = arr.get_element_unchecked(i)?;
                let item = item.$getter()? as $type;
                items.push(item)
            }
            Series::new(name, items).try_into_js(&cx)
        }
    };
}

init_method!(new_i8, JsNumber, i8, get_int32);
init_method!(new_i16, JsNumber, i16, get_int32);
init_method!(new_i32, JsNumber, i32, get_int32);
init_method!(new_i64, JsNumber, i64, get_int64);
init_method!(new_u8, JsNumber, u8, get_uint32);
init_method!(new_u16, JsNumber, u16, get_uint32);
init_method!(new_u32, JsNumber, u32, get_uint32);
init_method!(new_u64, JsNumber, u64, get_int64);
init_method!(new_f32, JsNumber, f32, get_double);
init_method!(new_f64, JsNumber, f64, get_double);

macro_rules! init_method_opt {
    ($name:ident, $type:ty, $native:ty) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let name = params.get_as::<&str>("name")?;
            let strict = params.get_as::<bool>("strict")?;
            let arr = params.get::<JsObject>("values")?;
            let len = arr.get_array_length()?;

            let mut builder = PrimitiveChunkedBuilder::<$type>::new(name, len as usize);
            for i in 0..len {
                let item: JsUnknown = arr.get_element_unchecked(i)?;
                let item: WrappedValue = item.into();
                match item.extract::<$native>() {
                    Ok(val) => builder.append_value(val),
                    Err(e) => {
                        if strict {
                            return Err(e);
                        }
                        builder.append_null()
                    }
                }
            }

            let ca: ChunkedArray<$type> = builder.finish();
            ca.into_series().try_into_js(&cx)
        }
    };
}

init_method_opt!(new_opt_f64, Float64Type, f64);
init_method_opt!(new_opt_u16, UInt16Type, u16);
init_method_opt!(new_opt_u32, UInt32Type, u32);
init_method_opt!(new_opt_u64, UInt64Type, u64);
init_method_opt!(new_opt_i8, Int8Type, i8);
init_method_opt!(new_opt_i16, Int16Type, i16);
init_method_opt!(new_opt_i32, Int32Type, i32);
init_method_opt!(new_opt_i64, Int64Type, i64);
init_method_opt!(new_opt_f32, Float32Type, f32);

macro_rules! impl_from_chunked {
    ($name:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            series.$name().into_series().try_into_js(&cx)
        }
    };
}

impl_from_chunked!(is_null);
impl_from_chunked!(is_not_null);
impl_from_chunked!(peak_max);
impl_from_chunked!(peak_min);

macro_rules! impl_from_chunked_with_err {
    ($name:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            series
                .$name()
                .map_err(JsPolarsEr::from)?
                .into_series()
                .try_into_js(&cx)
        }
    };
}
impl_from_chunked_with_err!(arg_unique);
impl_from_chunked_with_err!(is_not_nan);
impl_from_chunked_with_err!(is_nan);
impl_from_chunked_with_err!(is_finite);
impl_from_chunked_with_err!(is_infinite);
impl_from_chunked_with_err!(is_unique);
impl_from_chunked_with_err!(arg_true);
impl_from_chunked_with_err!(is_duplicated);
impl_from_chunked_with_err!(year);
impl_from_chunked_with_err!(month);
impl_from_chunked_with_err!(weekday);
impl_from_chunked_with_err!(week);
impl_from_chunked_with_err!(day);
impl_from_chunked_with_err!(ordinal_day);
impl_from_chunked_with_err!(hour);
impl_from_chunked_with_err!(minute);
impl_from_chunked_with_err!(second);
impl_from_chunked_with_err!(nanosecond);
impl_from_chunked_with_err!(is_first);
impl_from_chunked_with_err!(timestamp);

macro_rules! impl_method {
    ($name:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            series.$name().try_into_js(&cx)
        }
    };
    ($name:ident, $type:ty) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<$type> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            series.$name().try_into_js(&cx)
        }
    };
    ($name:ident, $type:ty, $property:expr ) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let prop = params.get_as::<$type>($property)?;
            series.$name(prop).try_into_js(&cx)
        }
    };
}

impl_method!(clone);
impl_method!(drop_nulls);
impl_method!(interpolate);

impl_method!(cumsum, bool, "reverse");
impl_method!(cummax, bool, "reverse");
impl_method!(cummin, bool, "reverse");
impl_method!(cumprod, bool, "reverse");
impl_method!(sort, bool, "reverse");
impl_method!(tail, Option<usize>, "length");
impl_method!(head, Option<usize>, "length");
impl_method!(limit, usize, "num_elements");
impl_method!(shift, i64, "periods");
impl_method!(take_every, usize, "n");

macro_rules! impl_method_with_err {
    ($name:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            series.$name().map_err(JsPolarsEr::from)?.try_into_js(&cx)
        }
    };
    ($name:ident, $type:ty) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<$type> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            series.$name().map_err(JsPolarsEr::from)?.try_into_js(&cx)
        }
    };
    ($name:ident, $type:ty, $property:expr ) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let prop = params.get_as::<$type>($property)?;
            series
                .$name(prop)
                .map_err(JsPolarsEr::from)?
                .try_into_js(&cx)
        }
    };
    ($name:ident, $type1:ty, $property1:expr , $type2:ty, $property2:expr ) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let prop1 = params.get_as::<$type1>($property1)?;
            let prop2 = params.get_as::<$type2>($property2)?;
            series
                .$name(prop1, prop2)
                .map_err(JsPolarsEr::from)?
                .try_into_js(&cx)
        }
    };
}

impl_method_with_err!(unique);
impl_method_with_err!(explode);
impl_method_with_err!(floor);
impl_method_with_err!(ceil);
impl_method_with_err!(mode);

impl_method_with_err!(n_unique, JsNumber);
impl_method_with_err!(round, u32, "decimals");
impl_method_with_err!(strftime, &str, "fmt");

#[js_function(1)]
pub(crate) fn clip(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let min = params.get_as::<f64>("min")?;
    let max = params.get_as::<f64>("max")?;
    series
        .clip(min, max)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn sample_n(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let n = params.get_as::<usize>("n")?;
    let with_replacement = params.get_as::<bool>("withReplacement")?;
    series
        .sample_n(n, with_replacement, 0)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn sample_frac(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let frac = params.get_as::<f64>("frac")?;
    let with_replacement = params.get_as::<bool>("withReplacement")?;
    series
        .sample_frac(frac, with_replacement, 0)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

macro_rules! impl_equality {
    ($name:ident, $method:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let rhs = params.get_external::<Series>(&cx, "rhs")?;
            series.$method(rhs).into_series().try_into_js(&cx)
        }
    };
}

impl_equality!(eq, equal);
impl_equality!(neq, not_equal);
impl_equality!(gt, gt);
impl_equality!(gt_eq, gt_eq);
impl_equality!(lt, lt);
impl_equality!(lt_eq, lt_eq);

macro_rules! impl_str_method {
    ($name:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let ca = series.utf8().map_err(JsPolarsEr::from)?;
            ca.$name().into_series().try_into_js(&cx)
        }
    };
    ($fn_name:ident, $method_name:ident) => {
        #[js_function(1)]
        pub fn $fn_name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let ca = series.utf8().map_err(JsPolarsEr::from)?;
            ca.$method_name().into_series().try_into_js(&cx)
        }
    };
}

impl_str_method!(str_lengths);
impl_str_method!(str_to_uppercase, to_uppercase);
impl_str_method!(str_to_lowercase, to_lowercase);

macro_rules! impl_str_method_with_err {
    ($name:ident, $type:ty, $property:expr ) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let ca = series.utf8().map_err(JsPolarsEr::from)?;
            let pat = params.get_as::<$type>($property)?;
            ca.$name(&pat)
                .map_err(JsPolarsEr::from)?
                .into_series()
                .try_into_js(&cx)
        }
    };
    ($fn_name:ident, $method_name:ident, $type:ty, $property:expr ) => {
        #[js_function(1)]
        pub fn $fn_name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let ca = series.utf8().map_err(JsPolarsEr::from)?;
            let arg = params.get_as::<$type>($property)?;
            ca.$method_name(arg)
                .map_err(JsPolarsEr::from)?
                .into_series()
                .try_into_js(&cx)
        }
    };
    ($fn_name:ident, $method_name:ident,  $type1:ty, $property1:expr, $type2:ty, $property2:expr) => {
        #[js_function(1)]
        pub fn $fn_name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let ca = series.utf8().map_err(JsPolarsEr::from)?;
            let prop1 = params.get_as::<$type1>($property1)?;
            let prop2 = params.get_as::<$type2>($property2)?;
            ca.$method_name(prop1, prop2)
                .map_err(JsPolarsEr::from)?
                .into_series()
                .try_into_js(&cx)
        }
    };
}

impl_str_method_with_err!(str_contains, contains, &str, "pat");
impl_str_method_with_err!(str_json_path_match, json_path_match, &str, "pat");
impl_str_method_with_err!(str_extract, extract, &str, "pat", usize, "groupIndex");
impl_str_method_with_err!(str_replace, replace, &str, "pat", &str, "val");
impl_str_method_with_err!(str_replace_all, replace_all, &str, "pat", &str, "val");
impl_str_method_with_err!(str_slice, str_slice, i64, "start", Option<u64>, "length");

#[js_function(1)]
pub fn hex_encode(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let ca = series.utf8().map_err(JsPolarsEr::from)?;
    ca.hex_encode().into_series().try_into_js(&cx)
}
#[js_function(1)]
pub fn hex_decode(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let strict = params.get_as::<Option<bool>>("strict")?;

    let ca = series.utf8().map_err(JsPolarsEr::from)?;
    ca.hex_decode(strict)
        .map_err(JsPolarsEr::from)?
        .into_series()
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn base64_encode(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let ca = series.utf8().map_err(JsPolarsEr::from)?;
    ca.base64_encode().into_series().try_into_js(&cx)
}

#[js_function(1)]
pub fn base64_decode(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let strict = params.get_as::<Option<bool>>("strict")?;

    let ca = series.utf8().map_err(JsPolarsEr::from)?;
    ca.base64_decode(strict)
        .map_err(JsPolarsEr::from)?
        .into_series()
        .try_into_js(&cx)
}

macro_rules! impl_rolling_method {
    ($name:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let window_size = params.get_as::<usize>("window_size")?;
            let weights = params.get_as::<Option<Vec<f64>>>("weights")?;
            let min_periods = params.get_as::<usize>("min_periods")?;
            let center = params.get_as::<bool>("center")?;
            let options = RollingOptions {
                window_size,
                weights,
                min_periods,
                center,
            };
            series
                .$name(options)
                .map_err(JsPolarsEr::from)?
                .try_into_js(&cx)
        }
    };
}

#[js_function(1)]
pub fn rolling_quantile(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let quantile = params.get_as::<f64>("quantile")?;
    let interpolation: QuantileInterpolOptions =
        params.get_or("interpolation", QuantileInterpolOptions::Nearest)?;
    let window_size: usize = params.get_or("window_size", 2)?;
    let min_periods: usize = params.get_or("min_periods", 2)?;
    let center: bool = params.get_or("center", false)?;
    let weights = params.get_as::<Option<Vec<f64>>>("weights")?;
    let options = RollingOptions {
        window_size,
        weights,
        min_periods,
        center,
    };

    series
        .rolling_quantile(quantile, interpolation, options)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}
impl_rolling_method!(rolling_sum);
impl_rolling_method!(rolling_mean);
impl_rolling_method!(rolling_max);
impl_rolling_method!(rolling_min);
impl_rolling_method!(rolling_var);
impl_rolling_method!(rolling_std);
impl_rolling_method!(rolling_median);

macro_rules! impl_set_with_mask {
    ($name:ident, $native:ty, $cast:ident, $variant:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let filter = params.get_external::<Series>(&cx, "filter")?;
            let value = params.get_as::<Option<$native>>("value")?;
            let mask = filter.bool().map_err(JsPolarsEr::from)?;
            let ca = series.$cast().map_err(JsPolarsEr::from)?;
            let new = ca.set(mask, value).map_err(JsPolarsEr::from)?;

            new.into_series().try_into_js(&cx)
        }
    };
}

impl_set_with_mask!(set_with_mask_str, &str, utf8, Utf8);
impl_set_with_mask!(set_with_mask_f64, f64, f64, Float64);
impl_set_with_mask!(set_with_mask_f32, f32, f32, Float32);
impl_set_with_mask!(set_with_mask_u8, u8, u8, UInt8);
impl_set_with_mask!(set_with_mask_u16, u16, u16, UInt16);
impl_set_with_mask!(set_with_mask_u32, u32, u32, UInt32);
impl_set_with_mask!(set_with_mask_u64, u64, u64, UInt64);
impl_set_with_mask!(set_with_mask_i8, i8, i8, Int8);
impl_set_with_mask!(set_with_mask_i16, i16, i16, Int16);
impl_set_with_mask!(set_with_mask_i32, i32, i32, Int32);
impl_set_with_mask!(set_with_mask_i64, i64, i64, Int64);
impl_set_with_mask!(set_with_mask_bool, bool, bool, Boolean);

macro_rules! impl_set_at_idx {
    ($name:ident, $native:ty, $cast:ident, $variant:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsUndefined> {
            let params = get_params(&cx)?;
            let series: &mut Series = params.get_external_mut::<Series>(&cx, "_series")?;
            let indices = params.get_as::<Vec<u32>>("indices")?;
            let value = params.get_as::<Option<$native>>("value")?;
            let ca = series.$cast().map_err(JsPolarsEr::from)?;
            let ca = ca
                .set_at_idx(indices.iter().map(|idx| *idx as usize), value)
                .map_err(JsPolarsEr::from)?
                .into_series();
            let old_series = std::mem::replace(series, ca);
            std::mem::drop(old_series);
            cx.env.get_undefined()
        }
    };
}

impl_set_at_idx!(set_at_idx_str, &str, utf8, Utf8);
impl_set_at_idx!(set_at_idx_f64, f64, f64, Float64);
impl_set_at_idx!(set_at_idx_f32, f32, f32, Float32);
impl_set_at_idx!(set_at_idx_u8, u8, u8, UInt8);
impl_set_at_idx!(set_at_idx_u16, u16, u16, UInt16);
impl_set_at_idx!(set_at_idx_u32, u32, u32, UInt32);
impl_set_at_idx!(set_at_idx_u64, u64, u64, UInt64);
impl_set_at_idx!(set_at_idx_i8, i8, i8, Int8);
impl_set_at_idx!(set_at_idx_i16, i16, i16, Int16);
impl_set_at_idx!(set_at_idx_i32, i32, i32, Int32);
impl_set_at_idx!(set_at_idx_i64, i64, i64, Int64);

macro_rules! impl_get {
    ($name:ident, $series_variant:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsUnknown> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let index = params.get_as::<i64>("index")?;
            match series.$series_variant() {
                Ok(ca) => {
                    let index = if index < 0 {
                        (ca.len() as i64 + index) as usize
                    } else {
                        index as usize
                    };
                    match ca.get(index) {
                        Some(v) => v.try_into_js(&cx).map(|v| v.into_unknown()),
                        None => cx.env.get_null().map(|v| v.into_unknown()),
                    }
                }
                Err(_) => cx.env.get_null().map(|v| v.into_unknown()),
            }
        }
    };
}

impl_get!(get_f32, f32);
impl_get!(get_f64, f64);
impl_get!(get_u8, u8);
impl_get!(get_u16, u16);
impl_get!(get_u32, u32);
impl_get!(get_i8, i8);
impl_get!(get_i16, i16);
impl_get!(get_i32, i32);
// impl_get!(get_i64, i64);

#[js_function(1)]
pub fn get_u64(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let index = params.get_as::<i64>("index")?;
    match series.u64() {
        Ok(ca) => {
            let index = if index < 0 {
                (ca.len() as i64 + index) as usize
            } else {
                index as usize
            };
            match ca.get(index) {
                Some(v) => v.try_into_js(&cx).map(|v| v.into_unknown().unwrap()),
                None => cx.env.get_null().map(|v| v.into_unknown()),
            }
        }
        Err(_) => cx.env.get_null().map(|v| v.into_unknown()),
    }
}
#[js_function(1)]
pub fn get_i64(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let index = params.get_as::<i64>("index")?;
    match series.i64() {
        Ok(ca) => {
            let index = if index < 0 {
                (ca.len() as i64 + index) as usize
            } else {
                index as usize
            };
            match ca.get(index) {
                Some(v) => v.try_into_js(&cx).map(|v| v.into_unknown().unwrap()),
                None => cx.env.get_null().map(|v| v.into_unknown()),
            }
        }
        Err(_) => cx.env.get_null().map(|v| v.into_unknown()),
    }
}

#[js_function(1)]
pub fn get_bool(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let series = params.get_external::<Series>(&cx, "_series")?;
    let index = params.get_as::<i64>("index")?;
    match series.bool() {
        Ok(ca) => {
            let index = if index < 0 {
                (ca.len() as i64 + index) as usize
            } else {
                index as usize
            };
            match ca.get(index) {
                Some(v) => cx.env.get_boolean(v).map(|v| v.into_unknown()),
                None => cx.env.get_null().map(|v| v.into_unknown()),
            }
        }
        Err(_) => cx.env.get_null().map(|v| v.into_unknown()),
    }
}

macro_rules! impl_arithmetic {
  ($name:ident, $type:ty, $operand:tt) => {
    #[js_function(1)]
      pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
          let params = get_params(&cx)?;
          let series = params.get_external::<Series>(&cx, "_series")?;
          let other = params.get_as::<$type>("other")?;
          (series $operand other).try_into_js(&cx)
      }
  };
}

impl_arithmetic!(add_u8, u8, +);
impl_arithmetic!(add_u16, u16, +);
impl_arithmetic!(add_u32, u32, +);
impl_arithmetic!(add_u64, u64, +);
impl_arithmetic!(add_i8, i8, +);
impl_arithmetic!(add_i16, i16, +);
impl_arithmetic!(add_i32, i32, +);
impl_arithmetic!(add_i64, i64, +);
impl_arithmetic!(add_f32, f32, +);
impl_arithmetic!(add_f64, f64, +);
impl_arithmetic!(sub_u8, u8, -);
impl_arithmetic!(sub_u16, u16, -);
impl_arithmetic!(sub_u32, u32, -);
impl_arithmetic!(sub_u64, u64, -);
impl_arithmetic!(sub_i8, i8, -);
impl_arithmetic!(sub_i16, i16, -);
impl_arithmetic!(sub_i32, i32, -);
impl_arithmetic!(sub_i64, i64, -);
impl_arithmetic!(sub_f32, f32, -);
impl_arithmetic!(sub_f64, f64, -);
impl_arithmetic!(div_u8, u8, /);
impl_arithmetic!(div_u16, u16, /);
impl_arithmetic!(div_u32, u32, /);
impl_arithmetic!(div_u64, u64, /);
impl_arithmetic!(div_i8, i8, /);
impl_arithmetic!(div_i16, i16, /);
impl_arithmetic!(div_i32, i32, /);
impl_arithmetic!(div_i64, i64, /);
impl_arithmetic!(div_f32, f32, /);
impl_arithmetic!(div_f64, f64, /);
impl_arithmetic!(mul_u8, u8, *);
impl_arithmetic!(mul_u16, u16, *);
impl_arithmetic!(mul_u32, u32, *);
impl_arithmetic!(mul_u64, u64, *);
impl_arithmetic!(mul_i8, i8, *);
impl_arithmetic!(mul_i16, i16, *);
impl_arithmetic!(mul_i32, i32, *);
impl_arithmetic!(mul_i64, i64, *);
impl_arithmetic!(mul_f32, f32, *);
impl_arithmetic!(mul_f64, f64, *);
impl_arithmetic!(rem_u8, u8, %);
impl_arithmetic!(rem_u16, u16, %);
impl_arithmetic!(rem_u32, u32, %);
impl_arithmetic!(rem_u64, u64, %);
impl_arithmetic!(rem_i8, i8, %);
impl_arithmetic!(rem_i16, i16, %);
impl_arithmetic!(rem_i32, i32, %);
impl_arithmetic!(rem_i64, i64, %);
impl_arithmetic!(rem_f32, f32, %);
impl_arithmetic!(rem_f64, f64, %);

macro_rules! impl_rhs_arithmetic {
    ($name:ident, $type:ty, $operand:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let other = params.get_as::<$type>("other")?;
            other.$operand(series).try_into_js(&cx)
        }
    };
}
impl_rhs_arithmetic!(add_u8_rhs, u8, add);
impl_rhs_arithmetic!(add_u16_rhs, u16, add);
impl_rhs_arithmetic!(add_u32_rhs, u32, add);
impl_rhs_arithmetic!(add_u64_rhs, u64, add);
impl_rhs_arithmetic!(add_i8_rhs, i8, add);
impl_rhs_arithmetic!(add_i16_rhs, i16, add);
impl_rhs_arithmetic!(add_i32_rhs, i32, add);
impl_rhs_arithmetic!(add_i64_rhs, i64, add);
impl_rhs_arithmetic!(add_f32_rhs, f32, add);
impl_rhs_arithmetic!(add_f64_rhs, f64, add);
impl_rhs_arithmetic!(sub_u8_rhs, u8, sub);
impl_rhs_arithmetic!(sub_u16_rhs, u16, sub);
impl_rhs_arithmetic!(sub_u32_rhs, u32, sub);
impl_rhs_arithmetic!(sub_u64_rhs, u64, sub);
impl_rhs_arithmetic!(sub_i8_rhs, i8, sub);
impl_rhs_arithmetic!(sub_i16_rhs, i16, sub);
impl_rhs_arithmetic!(sub_i32_rhs, i32, sub);
impl_rhs_arithmetic!(sub_i64_rhs, i64, sub);
impl_rhs_arithmetic!(sub_f32_rhs, f32, sub);
impl_rhs_arithmetic!(sub_f64_rhs, f64, sub);
impl_rhs_arithmetic!(div_u8_rhs, u8, div);
impl_rhs_arithmetic!(div_u16_rhs, u16, div);
impl_rhs_arithmetic!(div_u32_rhs, u32, div);
impl_rhs_arithmetic!(div_u64_rhs, u64, div);
impl_rhs_arithmetic!(div_i8_rhs, i8, div);
impl_rhs_arithmetic!(div_i16_rhs, i16, div);
impl_rhs_arithmetic!(div_i32_rhs, i32, div);
impl_rhs_arithmetic!(div_i64_rhs, i64, div);
impl_rhs_arithmetic!(div_f32_rhs, f32, div);
impl_rhs_arithmetic!(div_f64_rhs, f64, div);
impl_rhs_arithmetic!(mul_u8_rhs, u8, mul);
impl_rhs_arithmetic!(mul_u16_rhs, u16, mul);
impl_rhs_arithmetic!(mul_u32_rhs, u32, mul);
impl_rhs_arithmetic!(mul_u64_rhs, u64, mul);
impl_rhs_arithmetic!(mul_i8_rhs, i8, mul);
impl_rhs_arithmetic!(mul_i16_rhs, i16, mul);
impl_rhs_arithmetic!(mul_i32_rhs, i32, mul);
impl_rhs_arithmetic!(mul_i64_rhs, i64, mul);
impl_rhs_arithmetic!(mul_f32_rhs, f32, mul);
impl_rhs_arithmetic!(mul_f64_rhs, f64, mul);
impl_rhs_arithmetic!(rem_u8_rhs, u8, rem);
impl_rhs_arithmetic!(rem_u16_rhs, u16, rem);
impl_rhs_arithmetic!(rem_u32_rhs, u32, rem);
impl_rhs_arithmetic!(rem_u64_rhs, u64, rem);
impl_rhs_arithmetic!(rem_i8_rhs, i8, rem);
impl_rhs_arithmetic!(rem_i16_rhs, i16, rem);
impl_rhs_arithmetic!(rem_i32_rhs, i32, rem);
impl_rhs_arithmetic!(rem_i64_rhs, i64, rem);
impl_rhs_arithmetic!(rem_f32_rhs, f32, rem);
impl_rhs_arithmetic!(rem_f64_rhs, f64, rem);

macro_rules! impl_eq_num {
    ($name:ident, $type:ty) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let rhs = params.get_as::<$type>("rhs")?;
            series.equal(rhs).into_series().try_into_js(&cx)
        }
    };
}

impl_eq_num!(eq_u8, u8);
impl_eq_num!(eq_u16, u16);
impl_eq_num!(eq_u32, u32);
impl_eq_num!(eq_u64, u64);
impl_eq_num!(eq_i8, i8);
impl_eq_num!(eq_i16, i16);
impl_eq_num!(eq_i32, i32);
impl_eq_num!(eq_i64, i64);
impl_eq_num!(eq_f32, f32);
impl_eq_num!(eq_f64, f64);
impl_eq_num!(eq_str, &str);

macro_rules! impl_neq_num {
    ($name:ident, $type:ty) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let rhs = params.get_as::<$type>("rhs")?;
            series.not_equal(rhs).into_series().try_into_js(&cx)
        }
    };
}

impl_neq_num!(neq_u8, u8);
impl_neq_num!(neq_u16, u16);
impl_neq_num!(neq_u32, u32);
impl_neq_num!(neq_u64, u64);
impl_neq_num!(neq_i8, i8);
impl_neq_num!(neq_i16, i16);
impl_neq_num!(neq_i32, i32);
impl_neq_num!(neq_i64, i64);
impl_neq_num!(neq_f32, f32);
impl_neq_num!(neq_f64, f64);
impl_neq_num!(neq_str, &str);

macro_rules! impl_gt_num {
    ($name:ident, $type:ty) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let rhs = params.get_as::<$type>("rhs")?;
            series.gt(rhs).into_series().try_into_js(&cx)
        }
    };
}

impl_gt_num!(gt_u8, u8);
impl_gt_num!(gt_u16, u16);
impl_gt_num!(gt_u32, u32);
impl_gt_num!(gt_u64, u64);
impl_gt_num!(gt_i8, i8);
impl_gt_num!(gt_i16, i16);
impl_gt_num!(gt_i32, i32);
impl_gt_num!(gt_i64, i64);
impl_gt_num!(gt_f32, f32);
impl_gt_num!(gt_f64, f64);
impl_gt_num!(gt_str, &str);

macro_rules! impl_gt_eq_num {
    ($name:ident, $type:ty) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let rhs = params.get_as::<$type>("rhs")?;
            series.gt_eq(rhs).into_series().try_into_js(&cx)
        }
    };
}

impl_gt_eq_num!(gt_eq_u8, u8);
impl_gt_eq_num!(gt_eq_u16, u16);
impl_gt_eq_num!(gt_eq_u32, u32);
impl_gt_eq_num!(gt_eq_u64, u64);
impl_gt_eq_num!(gt_eq_i8, i8);
impl_gt_eq_num!(gt_eq_i16, i16);
impl_gt_eq_num!(gt_eq_i32, i32);
impl_gt_eq_num!(gt_eq_i64, i64);
impl_gt_eq_num!(gt_eq_f32, f32);
impl_gt_eq_num!(gt_eq_f64, f64);
impl_gt_eq_num!(gt_eq_str, &str);

macro_rules! impl_lt_num {
    ($name:ident, $type:ty) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let rhs = params.get_as::<$type>("rhs")?;
            series.lt(rhs).into_series().try_into_js(&cx)
        }
    };
}

impl_lt_num!(lt_u8, u8);
impl_lt_num!(lt_u16, u16);
impl_lt_num!(lt_u32, u32);
impl_lt_num!(lt_u64, u64);
impl_lt_num!(lt_i8, i8);
impl_lt_num!(lt_i16, i16);
impl_lt_num!(lt_i32, i32);
impl_lt_num!(lt_i64, i64);
impl_lt_num!(lt_f32, f32);
impl_lt_num!(lt_f64, f64);
impl_lt_num!(lt_str, &str);

macro_rules! impl_lt_eq_num {
    ($name:ident, $type:ty) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let series = params.get_external::<Series>(&cx, "_series")?;
            let rhs = params.get_as::<$type>("rhs")?;
            series.lt_eq(rhs).into_series().try_into_js(&cx)
        }
    };
}

impl_lt_eq_num!(lt_eq_u8, u8);
impl_lt_eq_num!(lt_eq_u16, u16);
impl_lt_eq_num!(lt_eq_u32, u32);
impl_lt_eq_num!(lt_eq_u64, u64);
impl_lt_eq_num!(lt_eq_i8, i8);
impl_lt_eq_num!(lt_eq_i16, i16);
impl_lt_eq_num!(lt_eq_i32, i32);
impl_lt_eq_num!(lt_eq_i64, i64);
impl_lt_eq_num!(lt_eq_f32, f32);
impl_lt_eq_num!(lt_eq_f64, f64);
impl_lt_eq_num!(lt_eq_str, &str);
