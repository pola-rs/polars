use crate::conversion::prelude::*;
use crate::dataframe::{DataFrameResult, JsDataFrame};
use crate::error::JsPolarsEr;
use crate::list_construction::*;
use crate::prelude::*;
use neon::prelude::*;
use neon::types::JsDate;
use polars::prelude::*;
use polars_core::utils::CustomIterTools;
use crate::conversion::from::*;

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
impl Finalize for JsSeries {}

type SeriesResult<'a> = JsResult<'a, JsBox<JsSeries>>;

impl JsSeries {
    pub fn new_bool(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let name = params.get_as::<&str, _>(&mut cx, "name")?;
        let (js_arr, len) = params.get_arr(&mut cx, "values")?;
        let mut items: Vec<bool> = Vec::with_capacity(len);

        for i in js_arr.to_vec(&mut cx)?.iter() {
            let i: &Handle<JsValue> = i;
            let v = i
                .downcast_or_throw::<JsBoolean, _>(&mut cx)
                .expect("err happening here");
            let item = v.value(&mut cx);
            items.push(item)
        }

        Ok(cx.boxed(Series::new(name, items).into()))
    }
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
    pub fn new_list(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let name = params.get_as::<String, _>(&mut cx, "name")?;
        let arr = params.get_arr_values(&mut cx, "values")?;
        let dtype: DataType = params.get_as::<JsDataType, _>(&mut cx, "dtype")?.into();
        let s: JsSeries = js_arr_to_list(&mut cx, &name, &arr, &dtype).map(|s| s.into())?;
        Ok(s.into_js_box(&mut cx))
    }
    pub fn new_object(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let name = params.get_as::<&'_ str, _>(&mut cx, "name")?;
        let values = params.get_as::<Vec<ObjectValue>, _>(&mut cx, "values")?;
        let s = ObjectChunked::<ObjectValue>::new_from_vec(name, values).into_series();
        Ok(JsSeries::new(s).into_js_box(&mut cx))
    }
    pub fn repeat(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let name = params.get_as::<&'_ str, _>(&mut cx, "name")?;
        let val: WrappedValue = cx.argument::<JsObject>(0)?.get(&mut cx, "value")?.into();
        let n = params.get_as::<usize, _>(&mut cx, "n")?;
        let dtype: DataType = params.get_as::<JsDataType, _>(&mut cx, "dtype")?.into();
        let s: Series = match dtype {
            DataType::Utf8 => {
                let val = val.extract::<&str>(&mut cx).unwrap();
                let mut ca: Utf8Chunked = (0..n).map(|_| val).collect_trusted();
                ca.rename(name);
                ca.into_series().into()
            }
            DataType::Int64 => {
                let val = val.extract::<i64>(&mut cx).unwrap();
                let mut ca: NoNull<Int64Chunked> = (0..n).map(|_| val).collect_trusted();
                ca.rename(name);
                ca.into_inner().into_series().into()
            }
            DataType::Float64 => {
                let val = val.extract::<f64>(&mut cx).unwrap();
                let mut ca: NoNull<Float64Chunked> = (0..n).map(|_| val).collect_trusted();
                ca.rename(name);
                ca.into_inner().into_series().into()
            }
            DataType::Boolean => {
                let val = val.extract::<bool>(&mut cx).unwrap();
                let mut ca: BooleanChunked = (0..n).map(|_| val).collect_trusted();
                ca.rename(name);
                ca.into_series().into()
            }

            dt => {
                panic!("cannot create repeat with dtype: {:?}", dt);
            }
        };
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
        todo!()
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
    pub fn get_idx(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series =
            params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let idx = params.get_as::<usize, _>(&mut cx, "idx")?;
        let obj: Wrap<AnyValue> = series.series.get(idx).into();
        Ok(obj.into_js(&mut cx))
    }
    pub fn bitand(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let out: JsSeries = series
            .series
            .bitand(&other.series)
            .map_err(JsPolarsEr::from)?
            .into();
        Ok(out.into_js_box(&mut cx))
    }
    pub fn bitor(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let out: JsSeries = series
            .series
            .bitor(&other.series)
            .map_err(JsPolarsEr::from)?
            .into();
        Ok(out.into_js_box(&mut cx))
    }
    pub fn bitxor(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let out: JsSeries = series
            .series
            .bitxor(&other.series)
            .map_err(JsPolarsEr::from)?
            .into();
        Ok(out.into_js_box(&mut cx))
    }
    pub fn mean(mut cx: FunctionContext) -> JsResult<JsNumber> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let s = match series.series.dtype() {
            DataType::Boolean => {
                let s = series.series.cast(&DataType::UInt8).unwrap();
                s.mean()
            }
            _ => series.series.mean(),
        }
        .unwrap();
        Ok(s.into_js(&mut cx))
    }
    pub fn max(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let s = match series.series.dtype() {
            DataType::Float32 | DataType::Float64 => series
                .series
                .max::<f64>()
                .unwrap()
                .into_js(&mut cx)
                .upcast(),
            DataType::Boolean => series
                .series
                .max::<u32>()
                .map(|v| v == 1)
                .unwrap()
                .into_js(&mut cx)
                .upcast(),
            _ => series
                .series
                .max::<i64>()
                .unwrap()
                .into_js(&mut cx)
                .upcast(),
        };
        Ok(s)
    }
    pub fn min(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let s = match series.series.dtype() {
            DataType::Float32 | DataType::Float64 => series
                .series
                .min::<f64>()
                .unwrap()
                .into_js(&mut cx)
                .upcast(),
            DataType::Boolean => series
                .series
                .min::<u32>()
                .map(|v| v == 1)
                .unwrap()
                .into_js(&mut cx)
                .upcast(),
            _ => series
                .series
                .min::<i64>()
                .unwrap()
                .into_js(&mut cx)
                .upcast(),
        };
        Ok(s)
    }
    pub fn sum(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let s = match series.series.dtype() {
            DataType::Float32 | DataType::Float64 => series
                .series
                .sum::<f64>()
                .unwrap()
                .into_js(&mut cx)
                .upcast(),
            DataType::Boolean => series
                .series
                .sum::<u32>()
                .unwrap()
                .into_js(&mut cx)
                .upcast(),
            _ => series
                .series
                .sum::<i64>()
                .unwrap()
                .into_js(&mut cx)
                .upcast(),
        };
        Ok(s)
    }
    pub fn slice(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let offset = params.get_as::<i64, _>(&mut cx, "offset")?;
        let length = params.get_as::<usize, _>(&mut cx, "length")?;
        let series = series.series.slice(offset, length);
        Ok(JsSeries::new(series).into_js_box(&mut cx))
    }

    /// Cant figure out a pattern for mutably borrowing from JS values without refactoring the entire box
    pub fn append(mut _cx: FunctionContext) -> JsResult<JsUndefined> {
        todo!()
    }
    pub fn filter(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let filter = params.extract_boxed::<JsSeries>(&mut cx, "filter")?;
        let filter_series = &filter.series;
        if let Ok(ca) = filter_series.bool() {
            let series = series.series.filter(ca).map_err(JsPolarsEr::from)?;
            Ok(JsSeries::new(series).into_js_box(&mut cx).upcast())
        } else {
            Err(JsPolarsEr::Other(String::from("Expected a boolean mask")).into())
        }
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
    pub fn rem(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let series: JsSeries = (&series.series % &other.series).into();
        Ok(series.into_js_box(&mut cx))
    }
    pub fn sort_in_place(mut _cx: FunctionContext) -> JsResult<JsUndefined> {
        todo!()
    }

    pub fn value_counts(mut cx: FunctionContext) -> DataFrameResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let unique: JsDataFrame = series
            .series
            .value_counts()
            .map_err(JsPolarsEr::from)?
            .into();
        Ok(unique.into_js_box(&mut cx))
    }
    pub fn arg_min(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let arg_min = series.series.arg_min();
        match arg_min {
            Some(v) => Ok(v.into_js(&mut cx).upcast()),
            None => Ok(cx.undefined().upcast()),
        }
    }
    pub fn arg_max(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let arg_max = series.series.arg_max();
        match arg_max {
            Some(v) => Ok(v.into_js(&mut cx).upcast()),
            None => Ok(cx.undefined().upcast()),
        }
    }

    // pub fn take(&self, indices: Wrap<AlignedVec<u32>>) -> PyResult<Self> {
    //     let indices = indices.0;
    //     let indices = UInt32Chunked::new_from_aligned_vec("", indices);
    //     let take = self.series.take(&indices).map_err(PyPolarsEr::from)?;
    //     Ok(PySeries::new(take))
    // }
    pub fn take(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let _series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        todo!()
    }
    pub fn take_with_series(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let indices = params.extract_boxed::<JsSeries>(&mut cx, "indices")?;
        let idx = indices.series.u32().map_err(JsPolarsEr::from)?;
        let take = series.series.take(idx).map_err(JsPolarsEr::from)?;
        Ok(JsSeries::new(take).into_js_box(&mut cx))
    }
    pub fn series_equal(mut cx: FunctionContext) -> JsResult<JsBoolean> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let null_equal = params.get_as::<bool, _>(&mut cx, "null_equal")?;
        if null_equal {
            Ok(series
                .series
                .series_equal_missing(&other.series)
                .into_js(&mut cx))
        } else {
            Ok(series.series.series_equal(&other.series).into_js(&mut cx))
        }
    }

    pub fn _not(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let b = series.series.bool().map_err(JsPolarsEr::from)?;
        let s: JsSeries = (!b).into_series().into();
        Ok(s.into_js_box(&mut cx))
    }
    pub fn as_str(mut cx: FunctionContext) -> JsResult<JsString> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let s = format!("{:?}", series.series);
        Ok(s.into_js(&mut cx))
    }
    pub fn len(mut cx: FunctionContext) -> JsResult<JsNumber> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let len = series.series.len();
        Ok(len.into_js(&mut cx))
    }

    // to_list equivalent
    pub fn to_array(mut cx: FunctionContext) -> JsResult<JsArray> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let js_arr = cx.empty_array();

        // let b: Vec<_> = (&series).series.bool().unwrap().clone().into();

        // for (idx, val) in b.into_iter().enumerate() {
        //     match val {
        //         Some(v) => {
        //             let jsv: Handle<JsValue> = v.into_js(&mut cx).upcast();
        //             js_arr.set(&mut cx, idx as u32, jsv)?;
        //         }
        //         None => {},
        //       }
        // }
    
        // // let arr = b.into_js_array(&mut cx);
        // // let primitive_to_list = |dt: &DataType, series: &Series| match dt {
        // //     DataType::Boolean => series.bool().unwrap(),
        // //     DataType::Utf8 => series.utf8().unwrap(),
        // //     DataType::UInt8 => series.u8().unwrap(),
        // //     DataType::UInt16 => series.u16().unwrap(),
        // //     DataType::UInt32 => series.u32().unwrap(),
        // //     DataType::UInt64 => series.u64().unwrap(),
        // //     DataType::Int8 => series.i8().unwrap(),
        // //     DataType::Int16 => series.i16().unwrap(),
        // //     DataType::Int32 => series.i32().unwrap(),
        // //     DataType::Int64 => series.i64().unwrap(),
        // //     DataType::Float32 => series.f32().unwrap(),
        // //     DataType::Float64 => series.f64().unwrap(),
        // //     dt => panic!("to_list() not implemented for {:?}", dt),
        // // };

        // let js_arr = match &series.series.dtype() {
        //     DataType::Categorical => series.series.categorical().unwrap().iter_str(),
        //     DataType::Object(_) => {
        //         let v = cx.empty_array();

        //         for i in 0..series.series.len() {
        //             let obj: Option<&ObjectValue> = series.series.get_object(i).map(|any| any.into());

        //             let val = obj.unwrap().into_js(&mut cx);

        //             v.append(val).unwrap();
        //         }
        //         v
        //     }
        //     DataType::List(inner_dtype) => {
        //         let v = PyList::empty(python);
        //         let ca = series.list().unwrap();
        //         for opt_s in ca.amortized_iter() {
        //             match opt_s {
        //                 None => {
        //                     v.append(python.None()).unwrap();
        //                 }
        //                 Some(s) => {
        //                     let pylst = primitive_to_list(&inner_dtype, s.as_ref());
        //                     AlignedVec  v.append(pylst).unwrap();
        //                 }
        //             }
        //         }
        //         v
        //     }
        //     DataType::Date => PyList::new(python, &series.date().unwrap().0),
        //     DataType::Datetime => PyList::new(python, &series.datetime().unwrap().0),
        //     dt => primitive_to_list(dt, series),
        // };
        // pylist.to_object(python)
        todo!()
    }
    pub fn median(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let median = match series.series.dtype() {
            DataType::Boolean => {
                let s = series.series.cast(&DataType::UInt8).unwrap();
                s.median()
            }
            _ => series.series.median(),
        };

        match median {
            Some(m) => Ok(m.into_js(&mut cx).upcast()),
            None => Ok(cx.undefined().upcast()),
        }
    }

    pub fn quantile(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let quantile = params.get_as::<f64, _>(&mut cx, "quantile")?;
        let s: Series = (&series.series).clone();
        let q = s.quantile_as_series(quantile).map_err(JsPolarsEr::from)?;
        let q = Wrap(q.get(0));
        Ok(q.into_js(&mut cx))
    }
    pub fn fill_null(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let strategy = params.get_as::<String, _>(&mut cx, "strategy")?;
        let strat = parse_strategy(strategy);
        let series = series.series.fill_null(strat).map_err(JsPolarsEr::from)?;
        Ok(JsSeries::new(series).into_js_box(&mut cx))
    }
    pub fn to_arrow(mut _cx: FunctionContext) -> SeriesResult {
        todo!()
    }

    pub fn apply_lambda(mut _cx: FunctionContext) -> SeriesResult {
        todo!()
    }
    pub fn zip_with(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let mask = params.extract_boxed::<JsSeries>(&mut cx, "mask")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let mask = mask.series.bool().map_err(JsPolarsEr::from)?;
        let s = series
            .series
            .zip_with(mask, &other.series)
            .map_err(JsPolarsEr::from)?;
        Ok(JsSeries::new(s).into_js_box(&mut cx))
    }

    pub fn str_parse_date(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let fmt = params.get_as::<Option<&str>, _>(&mut cx, "fmt")?;
        if let Ok(ca) = &series.series.utf8() {
            let ca = ca.as_date(fmt).map_err(JsPolarsEr::from)?;
            Ok(JsSeries::new(ca.into_series()).into_js_box(&mut cx))
        } else {
            Err(JsPolarsEr::Other("cannot parse Date expected utf8 type".into()).into())
        }
    }

    pub fn str_parse_datetime(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let fmt = params.get_as::<Option<&str>, _>(&mut cx, "fmt")?;
        if let Ok(ca) = &series.series.utf8() {
            let ca = ca.as_datetime(fmt).map_err(JsPolarsEr::from)?;
            Ok(JsSeries::new(ca.into_series()).into_js_box(&mut cx))
        } else {
            Err(JsPolarsEr::Other("cannot parse Date expected utf8 type".into()).into())
        }
    }

    pub fn arr_lengths(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let ca = series.series.list().map_err(JsPolarsEr::from)?;
        let s: Series = ca.clone().into_series();
        Ok(JsSeries::new(s).into_js_box(&mut cx))
    }

    pub fn to_dummies(mut cx: FunctionContext) -> DataFrameResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let dummies: JsDataFrame = series.series.to_dummies().map_err(JsPolarsEr::from)?.into();
        Ok(dummies.into_js_box(&mut cx))
    }
    pub fn get_list(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let index = params.get_as::<usize, _>(&mut cx, "index")?;

        if let Ok(ca) = &series.series.list() {
            let s = ca.get(index);
            let s: Handle<JsValue> = s
                .map(|s| JsSeries::new(s).into_js_box(&mut cx).upcast())
                .unwrap_or(cx.undefined().upcast());
            Ok(s)
        } else {
            Ok(cx.undefined().upcast())
        }
    }
    pub fn shrink_to_fit(mut _cx: FunctionContext) -> SeriesResult {
        todo!()
    }

    #[cfg(feature = "is_in")]
    pub fn is_in(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let s: JsSeries = series.series.is_in(&other.series).into_series().into();
        Ok(s.into_js_box(&mut cx))
    }
    pub fn dot(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let other = params.extract_boxed::<JsSeries>(&mut cx, "other")?;
        let dot = series.series.dot(&other.series);
        match dot {
            Some(d) => Ok(d.into_js(&mut cx).upcast()),
            None => Ok(cx.undefined().upcast()),
        }
    }
    pub fn hash(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let k0 = params.get_as::<u64, _>(&mut cx, "k0")?;
        let k1 = params.get_as::<u64, _>(&mut cx, "k1")?;
        let k2 = params.get_as::<u64, _>(&mut cx, "k2")?;
        let k3 = params.get_as::<u64, _>(&mut cx, "k3")?;
        let hb = ahash::RandomState::with_seeds(k0, k1, k2, k3);
        let s: JsSeries = series.series.hash(hb).into_series().into();
        Ok(s.into_js_box(&mut cx))
    }
    pub fn reinterpret(mut _cx: FunctionContext) -> SeriesResult {
        todo!()
    }
    pub fn rank(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let method = params.get_as::<String, _>(&mut cx, "method")?;
        let method = str_to_rankmethod(method)?;
        let rank: JsSeries = series.series.rank(method).into();
        Ok(rank.into_js_box(&mut cx))
    }
    pub fn diff(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let n = params.get_as::<usize, _>(&mut cx, "n")?;
        let null_behavior = params.get_as::<String, _>(&mut cx, "null_behavior")?;
        let null_behavior = str_to_null_behavior(null_behavior)?;
        let diff: JsSeries = series.series.diff(n, null_behavior).into();
        Ok(diff.into_js_box(&mut cx))
    }
    pub fn cast(mut cx: FunctionContext) -> SeriesResult {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let dtype: DataType = params.get_as::<JsDataType, _>(&mut cx, "dtype")?.into();
        let strict = params.get_as::<bool, _>(&mut cx, "strict")?;
        let out = if strict {
            series.series.strict_cast(&dtype)
        } else {
            series.series.cast(&dtype)
        };
        let out: JsSeries = out.map_err(JsPolarsEr::from)?.into();
        Ok(out.into_js_box(&mut cx))
    }
    pub fn skew(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let bias = params.get_as::<bool, _>(&mut cx, "bias")?;
        let skew = series.series.skew(bias).map_err(JsPolarsEr::from)?;
        match skew {
            Some(d) => Ok(d.into_js(&mut cx).upcast()),
            None => Ok(cx.undefined().upcast()),
        }
    }
    pub fn kurtosis(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let fisher = params.get_as::<bool, _>(&mut cx, "fisher")?;
        let bias = params.get_as::<bool, _>(&mut cx, "bias")?;
        let kurtosis = series
            .series
            .kurtosis(fisher, bias)
            .map_err(JsPolarsEr::from)?;
        match kurtosis {
            Some(d) => Ok(d.into_js(&mut cx).upcast()),
            None => Ok(cx.undefined().upcast()),
        }
    }
    pub fn get_str(mut cx: FunctionContext) -> JsResult<JsValue> {
        let params = get_params(&mut cx)?;
        let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
        let index = params.get_as::<i64, _>(&mut cx, "index")?;
        let series = (&series).series.clone();
        match series.utf8() {
            Ok(ca) => {
                let index = if index < 0 {
                    (ca.len() as i64 + index) as usize
                } else {
                    index as usize
                };
                let ca = ca.get(index);
                match ca {
                    Some(s) => {
                        let s = s.to_string();
                        Ok(cx.string(s).upcast())
                    }
                    None => Ok(cx.undefined().upcast()),
                }
            }
            Err(_) => Ok(cx.undefined().upcast()),
        }
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

init_method_opt!(new_opt_u16, UInt16Type, u16);
init_method_opt!(new_opt_u32, UInt32Type, u32);
init_method_opt!(new_opt_u64, UInt64Type, u64);
init_method_opt!(new_opt_i8, Int8Type, i8);
init_method_opt!(new_opt_i16, Int16Type, i16);
init_method_opt!(new_opt_i32, Int32Type, i32);
init_method_opt!(new_opt_i64, Int64Type, i64);
init_method_opt!(new_opt_f32, Float32Type, f32);
init_method_opt!(new_opt_f64, Float64Type, f64);

macro_rules! impl_from_chunked {
    ($name:ident) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let $name: JsSeries = series.series.$name().into_series().into();
                Ok($name.into_js_box(&mut cx))
            }
        }
    };
    ($name:ident, $type:ty, $property:expr ) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let val = params.get_as::<$type, _>(&mut cx, $property)?;

                let $name: JsSeries = series.series.$name(val).into_series().into();
                Ok($name.into_js_box(&mut cx))
            }
        }
    };
}

impl_from_chunked!(is_null);
impl_from_chunked!(is_not_null);
impl_from_chunked!(peak_max);
impl_from_chunked!(peak_min);
impl_from_chunked!(argsort, bool, "reverse");

macro_rules! impl_from_chunked_with_err {
    ($name:ident) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let $name: JsSeries = series
                    .series
                    .$name()
                    .map_err(JsPolarsEr::from)?
                    .into_series()
                    .into();
                Ok($name.into_js_box(&mut cx))
            }
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
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let s: JsSeries = series.series.$name().into();
                Ok(s.into_js_box(&mut cx))
            }
        }
    };
    ($name:ident, $type:ty) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> JsResult<$type> {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                Ok(series.series.$name().into_js(&mut cx))
            }
        }
    };
    ($name:ident, $type:ty, $property:expr ) => {
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
impl_method!(clone);
impl_method!(drop_nulls);
impl_method!(interpolate);

impl_method!(n_chunks, JsNumber);
impl_method!(null_count, JsNumber);
impl_method!(has_validity, JsBoolean);

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
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let s: JsSeries = series.series.$name().map_err(JsPolarsEr::from)?.into();
                Ok(s.into_js_box(&mut cx))
            }
        }
    };
    ($name:ident, $type:ty) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> JsResult<$type> {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let t = series.series.$name().map_err(JsPolarsEr::from)?;

                Ok(t.into_js(&mut cx))
            }
        }
    };
    ($name:ident, $type:ty, $property:expr ) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let prop = params.get_as::<$type, _>(&mut cx, $property)?;
                let s: JsSeries = series.series.$name(prop).map_err(JsPolarsEr::from)?.into();
                Ok(s.into_js_box(&mut cx))
            }
        }
    };
    ($name:ident, $type1:ty, $property1:expr , $type2:ty, $property2:expr ) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let prop1 = params.get_as::<$type1, _>(&mut cx, $property1)?;
                let prop2 = params.get_as::<$type2, _>(&mut cx, $property2)?;
                let s: JsSeries = series
                    .series
                    .$name(prop1, prop2)
                    .map_err(JsPolarsEr::from)?
                    .into();
                Ok(s.into_js_box(&mut cx))
            }
        }
    };
}

impl_method_with_err!(unique);
impl_method_with_err!(explode);
impl_method_with_err!(floor);
impl_method_with_err!(mode);
impl_method_with_err!(n_unique, JsNumber);
impl_method_with_err!(round, u32, "decimals");
impl_method_with_err!(strftime, &str, "fmt");
impl_method_with_err!(sample_n, usize, "n", bool, "with_replacement");
impl_method_with_err!(sample_frac, f64, "frac", bool, "with_replacement");

macro_rules! impl_equality {
    ($name:ident) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let rhs = params.extract_boxed::<JsSeries>(&mut cx, "rhs")?;
                let s: JsSeries = series.series.$name(&rhs.series).into_series().into();
                Ok(s.into_js_box(&mut cx))
            }
        }
    };
}

impl_equality!(eq);
impl_equality!(neq);
impl_equality!(gt);
impl_equality!(gt_eq);
impl_equality!(lt);
impl_equality!(lt_eq);

macro_rules! impl_str_method {
    ($name:ident) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let ca = series.series.utf8().map_err(JsPolarsEr::from)?;
                let s = ca.$name().into_series();
                Ok(JsSeries::new(s).into_js_box(&mut cx))
            }
        }
    };
    ($fn_name:ident, $method_name:ident) => {
        impl JsSeries {
            pub fn $fn_name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let ca = series.series.utf8().map_err(JsPolarsEr::from)?;
                let s = ca.$method_name().into_series();
                Ok(JsSeries::new(s).into_js_box(&mut cx))
            }
        }
    };
}

impl_str_method!(str_lengths);
impl_str_method!(str_to_uppercase, to_uppercase);
impl_str_method!(str_to_lowercase, to_lowercase);

macro_rules! impl_str_method_with_err {
    ($name:ident, $type:ty, $property:expr ) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let ca = series.series.utf8().map_err(JsPolarsEr::from)?;
                let pat = params.get_as::<$type, _>(&mut cx, $property)?;
                let s = ca.$name(&pat).map_err(JsPolarsEr::from)?.into_series();
                Ok(JsSeries::new(s).into_js_box(&mut cx))
            }
        }
    };
    ($fn_name:ident, $method_name:ident, $type:ty, $property:expr ) => {
        impl JsSeries {
            pub fn $fn_name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let ca = series.series.utf8().map_err(JsPolarsEr::from)?;
                let arg = params.get_as::<$type, _>(&mut cx, $property)?;
                let s = ca
                    .$method_name(arg)
                    .map_err(JsPolarsEr::from)?
                    .into_series();
                Ok(JsSeries::new(s).into_js_box(&mut cx))
            }
        }
    };
    ($fn_name:ident, $method_name:ident,  $type1:ty, $property1:expr, $type2:ty, $property2:expr) => {
        impl JsSeries {
            pub fn $fn_name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let ca = series.series.utf8().map_err(JsPolarsEr::from)?;
                let prop1 = params.get_as::<$type1, _>(&mut cx, $property1)?;
                let prop2 = params.get_as::<$type2, _>(&mut cx, $property2)?;
                let s = ca
                    .$method_name(prop1, prop2)
                    .map_err(JsPolarsEr::from)?
                    .into_series();
                Ok(JsSeries::new(s).into_js_box(&mut cx))
            }
        }
    };
}

impl_str_method_with_err!(str_contains, contains, &str, "pat");
impl_str_method_with_err!(str_json_path_match, json_path_match, &str, "pat");
impl_str_method_with_err!(str_extract, extract, &str, "pat", usize, "group_index");
impl_str_method_with_err!(str_replace, replace, &str, "pat", &str, "val");
impl_str_method_with_err!(str_replace_all, replace_all, &str, "pat", &str, "val");
impl_str_method_with_err!(str_slice, str_slice, i64, "start", Option<u64>, "length");

macro_rules! impl_rolling_method {
    ($name:ident) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let window_size = params.get_as::<usize, _>(&mut cx, "window_size")?;
                let weights = params.get_as::<Option<Vec<f64>>, _>(&mut cx, "weights")?;
                let min_periods = params.get_as::<usize, _>(&mut cx, "min_periods")?;
                let center = params.get_as::<bool, _>(&mut cx, "center")?;
                let options = RollingOptions {
                    window_size,
                    weights,
                    min_periods,
                    center,
                };
                let s = series.series.$name(options).map_err(JsPolarsEr::from)?;
                Ok(JsSeries::new(s).into_js_box(&mut cx))
            }
        }
    };
}

impl_rolling_method!(rolling_sum);
impl_rolling_method!(rolling_mean);
impl_rolling_method!(rolling_max);
impl_rolling_method!(rolling_min);
impl_rolling_method!(rolling_var);
impl_rolling_method!(rolling_std);

macro_rules! impl_set_with_mask {
    ($name:ident, $native:ty, $cast:ident, $variant:ident) => {
        fn $name(series: &Series, filter: &JsSeries, value: Option<$native>) -> Result<Series> {
            let mask = filter.series.bool()?;
            let ca = series.$cast()?;
            let new = ca.set(mask, value)?;
            Ok(new.into_series())
        }

        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let filter = params.extract_boxed::<JsSeries>(&mut cx, "filter")?;
                let value = params.get_as::<Option<$native>, _>(&mut cx, "value")?;

                let series = $name(&series.series, &filter, value).map_err(JsPolarsEr::from)?;
                Ok(JsSeries::new(series).into_js_box(&mut cx))
            }
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
        impl JsSeries {
            pub fn $name(mut _cx: FunctionContext) -> SeriesResult {
                todo!()
            }
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
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> JsResult<JsValue> {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let index = params.get_as::<i64, _>(&mut cx, "index")?;
                match series.series.$series_variant() {
                    Ok(ca) => {
                        let index = if index < 0 {
                            (ca.len() as i64 + index) as usize
                        } else {
                            index as usize
                        };
                        match ca.get(index) {
                            Some(v) => Ok(v.into_js(&mut cx).upcast()),
                            None => Ok(cx.undefined().upcast()),
                        }
                    }
                    Err(_) => Ok(cx.undefined().upcast()),
                }
            }
        }
    };
}
impl_get!(get_f32, f32);
impl_get!(get_f64, f64);
impl_get!(get_u8, u8);
impl_get!(get_u16, u16);
impl_get!(get_u32, u32);
impl_get!(get_u64, u64);
impl_get!(get_i8, i8);
impl_get!(get_i16, i16);
impl_get!(get_i32, i32);
impl_get!(get_i64, i64);
impl_get!(get_date, date);
impl_get!(get_datetime, datetime);

macro_rules! impl_arithmetic {
    ($name:ident, $type:ty, $operand:tt) => {
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let other = params.get_as::<$type, _>(&mut cx, "other")?;
                let series = (&series.series $operand other);
                Ok(JsSeries::new(series).into_js_box(&mut cx))
            }
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
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let other = params.get_as::<$type, _>(&mut cx, "other")?;
                let series = other.$operand(&series.series);
                Ok(JsSeries::new(series).into_js_box(&mut cx))
            }
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
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let rhs = params.get_as::<$type, _>(&mut cx, "rhs")?;
                let series = series.series.eq(rhs).into_series();
                Ok(JsSeries::new(series).into_js_box(&mut cx))
            }
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
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let rhs = params.get_as::<$type, _>(&mut cx, "rhs")?;
                let series = series.series.neq(rhs).into_series();
                Ok(JsSeries::new(series).into_js_box(&mut cx))
            }
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
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let rhs = params.get_as::<$type, _>(&mut cx, "rhs")?;
                let series = series.series.gt(rhs).into_series();
                Ok(JsSeries::new(series).into_js_box(&mut cx))
            }
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
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let rhs = params.get_as::<$type, _>(&mut cx, "rhs")?;
                let series = series.series.gt_eq(rhs).into_series();
                Ok(JsSeries::new(series).into_js_box(&mut cx))
            }
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
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let rhs = params.get_as::<$type, _>(&mut cx, "rhs")?;
                let series = series.series.lt(rhs).into_series();
                Ok(JsSeries::new(series).into_js_box(&mut cx))
            }
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
        impl JsSeries {
            pub fn $name(mut cx: FunctionContext) -> SeriesResult {
                let params = get_params(&mut cx)?;
                let series = params.extract_boxed::<JsSeries>(&mut cx, "_series")?;
                let rhs = params.get_as::<$type, _>(&mut cx, "rhs")?;
                let series = series.series.lt_eq(rhs).into_series();
                Ok(JsSeries::new(series).into_js_box(&mut cx))
            }
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
