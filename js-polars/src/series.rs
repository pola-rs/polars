// use crate::dataframe::JsDataFrame;
// use crate::datatypes::JsDataType;
// use crate::error::JsPolarsEr;
use crate::prelude::*;
use neon::prelude::*;

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
                let mut out: Vec<$type> = Vec::with_capacity(len);

                for i in js_arr.to_vec(&mut cx)?.iter() {
                    let v = i.downcast_or_throw::<$js_type, _>(&mut cx)?;
                    let item = v.value(&mut cx) as $type;
                    out.push(item)
                }

                Ok(cx.boxed(Series::new(name, out).into()))
            }
        }
    };
}
init_method!(new_i8, JsNumber, i8);
init_method!(new_i16, JsNumber, i16);
init_method!(new_i32, JsNumber, i32);
init_method!(new_i64, JsNumber, i64);
init_method!(new_u8, JsNumber, u8);
init_method!(new_u16, JsNumber, u16);
init_method!(new_u32, JsNumber, u32);
init_method!(new_u64, JsNumber, u64);
init_method!(new_bool, JsBoolean, bool);

// Init with lists that can contain Nones
macro_rules! init_method_opt {
    ($name:ident, $type:ty, $native: ty) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let name = params.get_as::<&str, _>(&mut cx, "name")?;
                let strict = params.get_as::<bool, _>(&mut cx, "strict")?;
                let (js_arr, len) = params.get_arr(&mut cx, "values")?;
                let mut builder= PrimitiveChunkedBuilder::<$type>::new(name, len);
                builder.append_null();
                for item in js_arr.to_vec(&mut cx)?.iter() {
                    if item.is_a::<JsNull, _>(&mut cx) {
                        builder.append_null()
                    } else if item.is_a::<JsNull, _>(&mut cx) {
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
    pub fn read_objects(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let name = params.get_as::<String, _>(&mut cx, "name")?;
        let values = get_array_param(&mut cx, "values")?;

        let values = values.to_vec(&mut cx).expect("none");
        let first = &values[0];
        if first.is_a::<JsNumber, _>(&mut cx) {
            let series = Float64Chunked::new_from_opt_iter(
                &name,
                values.iter().map(|v| {
                    Some(
                        v.downcast_or_throw::<JsNumber, _>(&mut cx)
                            .unwrap()
                            .value(&mut cx),
                    )
                }),
            )
            .into_series();
            Ok(cx.boxed(JsSeries { series }))
        } else if first.is_a::<JsString, _>(&mut cx) {
            let series = Utf8Chunked::new_from_opt_iter(
                &name,
                values.iter().map(|v| {
                    Some(
                        v.downcast_or_throw::<JsString, _>(&mut cx)
                            .unwrap()
                            .value(&mut cx),
                    )
                }),
            )
            .into_series();
            Ok(cx.boxed(JsSeries { series }))
        } else {
            unimplemented!()
        }
    }

    pub fn get_fmt(mut cx: FunctionContext) -> JsResult<JsString> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let s = format!("{}", &series.series);
        Ok(JsString::new(&mut cx, s))
    }

    pub fn add(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let series: JsSeries = (&series.series + &other.series).into();
        Ok(series.to_js_box(&mut cx))
    }

    pub fn sub(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let series: JsSeries = (&series.series - &other.series).into();
        Ok(series.to_js_box(&mut cx))
    }

    pub fn mul(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let series: JsSeries = (&series.series * &other.series).into();
        Ok(series.to_js_box(&mut cx))
    }
    pub fn div(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let series: JsSeries = (&series.series / &other.series).into();
        Ok(series.to_js_box(&mut cx))
    }
    pub fn head(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let length = params.get_as::<Option<usize>, _>(&mut cx, "length")?;

        let series: JsSeries = series.series.head(length).into();
        Ok(series.to_js_box(&mut cx))
    }

    pub fn tail(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let length = params.get_as::<Option<usize>, _>(&mut cx, "length")?;
        let series: JsSeries = series.series.tail(length).into();
        Ok(series.to_js_box(&mut cx))
    }
}
