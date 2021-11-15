// use crate::dataframe::JsDataFrame;
// use crate::datatypes::JsDataType;
// use crate::error::JsPolarsEr;
use crate::conversion::prelude::*;
use crate::prelude::*;
use neon::prelude::*;
use neon::types::JsDate;

#[repr(transparent)]
#[derive(Clone)]
pub struct JsSeries {
    pub(crate) series: Series,
}

impl From<Series> for JsSeries {
    fn from(series: Series) -> Self {
        Self { series }
    }
}
impl JsSeries {
    pub(crate) fn new(series: Series) -> Self {
        JsSeries { series }
    }
}
macro_rules! init_method {
    ($name:ident,$js_type:ty,$type:ty) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let name = params.get_as::<&str, _>(&mut cx, "name")?;
                let (js_arr, len) = params.get_arr(&mut cx, "values")?;
                let mut items: Vec<$type> = Vec::with_capacity(len);

                for i in js_arr.to_vec(&mut cx)?.iter() {
                    let v = i
                        .downcast_or_throw::<$js_type, _>(&mut cx)
                        .expect("err happening here");
                    let item = v.value(&mut cx) as $type;
                    items.push(item)
                }

                Ok(cx.boxed(Series::new(name, items).into()))
            }
        }
    };
}

impl JsSeries {
    pub fn new_bool(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let name = params.get_as::<&str, _>(&mut cx, "name")?;
        let (js_arr, len) = params.get_arr(&mut cx, "values")?;
        let mut items: Vec<bool> = Vec::with_capacity(len);

        for i in js_arr.to_vec(&mut cx)?.iter() {
            let i: &Handle<JsValue> = i;
            let s = i.to_string(&mut cx)?.value(&mut cx);
            println!("boolval={:#?}", s);
            let v = i
                .downcast_or_throw::<JsBoolean, _>(&mut cx)
                .expect("err happening here");

            let item = v.value(&mut cx) as bool;
            items.push(item)
        }

        Ok(cx.boxed(Series::new(name, items).into()))
    }
}
init_method!(new_i8, JsNumber, i8);
init_method!(new_i16, JsNumber, i16);
init_method!(new_i32, JsNumber, i32);
init_method!(new_i64, JsNumber, i64);
init_method!(new_u8, JsNumber, u8);
init_method!(new_u16, JsNumber, u16);
init_method!(new_u32, JsNumber, u32);
init_method!(new_u64, JsNumber, u64);
init_method!(new_f32, JsNumber, f32);
init_method!(new_f64, JsNumber, f64);
// init_method!(new_bool, JsBoolean, bool);
// Init with lists that can contain Nones
macro_rules! init_method_opt {
    ($name:ident, $type:ty, $native: ty) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let name = params.get_as::<&str, _>(&mut cx, "name")?;
                let strict = params.get_as::<bool, _>(&mut cx, "strict")?;
                let (js_arr, len) = params.get_arr(&mut cx, "values")?;
                let mut builder = PrimitiveChunkedBuilder::<$type>::new(name, len);
                for item in js_arr.to_vec(&mut cx)?.iter() {
                    if item.is_a::<JsNull, _>(&mut cx) {
                        builder.append_null()
                    } else if item.is_a::<JsUndefined, _>(&mut cx) {
                        builder.append_null()
                    } else {
                        match <$native>::from_js(&mut cx, *item) {
                            Ok(val) => builder.append_value(val),
                            Err(e) => {
                                if strict {
                                    return Err(e);
                                }

                                builder.append_null()
                            }
                        }
                    }
                }
                let ca: ChunkedArray<$type> = builder.finish();

                let s = ca.into_series();
                Ok(cx.boxed(JsSeries { series: s }))
            }
        }
    };
}

macro_rules! single_arg_method {
    ($name:ident, $type:ty, $property:expr) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let prop = params.get_as::<$type, _>(&mut cx, $property)?;
                let s: JsSeries = series.series.$name(prop).into();
                Ok(s.into_js_box(&mut cx))
            }
        }
    };
}

single_arg_method!(cumsum, bool, "reverse");
single_arg_method!(cummax, bool, "reverse");
single_arg_method!(cummin, bool, "reverse");
single_arg_method!(cumprod, bool, "reverse");
single_arg_method!(tail, Option<usize>, "tail");
single_arg_method!(head, Option<usize>, "head");

init_method_opt!(new_opt_u16, UInt16Type, u16);
init_method_opt!(new_opt_u32, UInt32Type, u32);
init_method_opt!(new_opt_u64, UInt64Type, u64);
init_method_opt!(new_opt_i8, Int8Type, i8);
init_method_opt!(new_opt_i16, Int16Type, i16);
init_method_opt!(new_opt_i32, Int32Type, i32);
init_method_opt!(new_opt_i64, Int64Type, i64);
init_method_opt!(new_opt_f32, Float32Type, f32);
init_method_opt!(new_opt_f64, Float64Type, f64);

impl Finalize for JsSeries {}

type SeriesResult<'a> = JsResult<'a, JsBox<JsSeries>>;
impl JsSeries {
    pub fn new_opt_date(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let name = params.get_as::<&str, _>(&mut cx, "name")?;
        let strict = params.get_as::<bool, _>(&mut cx, "strict")?;
        let arr = params.get_arr_values(&mut cx, "values")?;

        let mut builder = PrimitiveChunkedBuilder::<Int64Type>::new(name, arr.len());
        for item in arr.iter() {
            if item.is_a::<JsNull>(&mut cx) {
                builder.append_null()
            } else if item.is_a::<JsUndefined>(&mut cx) {
                builder.append_null()
            } else {
                match item.0.downcast_or_throw::<JsDate, _>(&mut cx) {
                    Ok(val) => builder.append_value(val.value(&mut cx) as i64),
                    Err(e) => {
                        if strict {
                            return Err(e);
                        }
                        builder.append_null()
                    }
                }
            }
        }
        let ca: ChunkedArray<Int64Type> = builder.finish();
        let s = ca.into_date().into_series();
        Ok(cx.boxed(JsSeries { series: s }))
    }
    pub fn new_opt_bool(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let name = params.get_as::<&'_ str, _>(&mut cx, "name")?;
        let (js_arr, len) = params.get_arr(&mut cx, "values")?;
        let mut builder = BooleanChunkedBuilder::new(name, len);
        for item in js_arr.to_vec(&mut cx)?.iter() {
            if item.is_a::<JsNull, _>(&mut cx) {
                builder.append_null()
            } else if item.is_a::<JsUndefined, _>(&mut cx) {
                builder.append_null()
            } else {
                match <bool>::from_js(&mut cx, *item) {
                    Ok(val) => builder.append_value(val),
                    Err(_) => builder.append_null(),
                }
            }
        }
        let ca: ChunkedArray<BooleanType> = builder.finish();

        let s = ca.into_series();
        Ok(cx.boxed(JsSeries { series: s }))
    }

    pub fn new_str(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let name = params.get_as::<&'_ str, _>(&mut cx, "name")?;
        let val = params.get_as::<Wrap<Utf8Chunked>, _>(&mut cx, "values")?;
        let mut s = val.0.into_series();
        s.rename(name);
        Ok(JsSeries::new(s).into_js_box(&mut cx))
    }

    pub fn new_object(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let name = params.get_as::<&'_ str, _>(&mut cx, "name")?;
        let values = params.get_as::<Vec<ObjectValue>, _>(&mut cx, "values")?;
        let s = ObjectChunked::<ObjectValue>::new_from_vec(name, values).into_series();
        Ok(JsSeries::new(s).into_js_box(&mut cx))
    }

    pub fn dtype(mut cx: FunctionContext) -> JsResult<JsNumber> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let dt: JsDataType = series.series.dtype().into();
        Ok(dt.into_js(&mut cx))
    }

    pub fn get_fmt(mut cx: FunctionContext) -> JsResult<JsString> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let s = format!("{}", &series.series);
        Ok(JsString::new(&mut cx, s))
    }

    pub fn chunk_lengths(mut _cx: FunctionContext) -> JsResult<JsArray> {
        // let params = get_params(&mut cx)?;
        // let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        unimplemented!()
    }

    pub fn name(mut cx: FunctionContext) -> JsResult<JsString> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let n = series.series.name();
        let s = n.to_owned();
        Ok(s.into_js(&mut cx))
    }

    pub fn rename(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let name = params.get_as::<&'_ str, _>(&mut cx, "name")?;
        let mut s: Series = (&series.series).clone();
        let s = s.rename(name);
        let s: JsSeries = s.clone().into();
        Ok(s.into_js_box(&mut cx))
    }

    pub fn add(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let series: JsSeries = (&series.series + &other.series).into();
        Ok(series.into_js_box(&mut cx))
    }

    pub fn sub(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let series: JsSeries = (&series.series - &other.series).into();
        Ok(series.into_js_box(&mut cx))
    }

    pub fn mul(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let series: JsSeries = (&series.series * &other.series).into();
        Ok(series.into_js_box(&mut cx))
    }

    pub fn div(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let series: JsSeries = (&series.series / &other.series).into();
        Ok(series.into_js_box(&mut cx))
    }

    pub fn get_idx(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let idx = params.get_as::<usize, _>(&mut cx, "idx")?;
        let _wv = Wrap(series.series.get(idx));

        todo!()
    }
}
