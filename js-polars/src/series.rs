use neon::prelude::*;

use crate::conversion::*;
use polars::prelude::{Float64Chunked, IntoSeries, NewChunkedArray, Series, Utf8Chunked};

pub struct JsSeries {
    pub(crate) series: Series,
}

impl From<Series> for JsSeries {
    fn from(series: Series) -> Self {
        Self { series }
    }
}

impl Finalize for JsSeries {}

type SeriesResult<'a> = JsResult<'a, JsBox<JsSeries>>;
impl JsSeries {
    pub fn new(mut cx: FunctionContext) -> SeriesResult {
        let name: String = cx.from_named_parameter("name")?;
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
        let series = JsSeries::extract_boxed(&mut cx, "_series")?;
        let s = format!("{}", series.series);
        Ok(JsString::new(&mut cx, s))
    }

    pub fn add(mut cx: FunctionContext) -> SeriesResult {
        let series = JsSeries::extract_boxed(&mut cx, "_series")?;
        let other = JsSeries::extract_boxed(&mut cx, "other")?;
        let series: JsSeries = (&series.series + &other.series).into();
        Ok(series.into_js_box(&mut cx))
    }

    pub fn sub(mut cx: FunctionContext) -> SeriesResult {
        let series = JsSeries::extract_boxed(&mut cx, "_series")?;
        let other = JsSeries::extract_boxed(&mut cx, "other")?;
        let series: JsSeries = (&series.series - &other.series).into();
        Ok(series.into_js_box(&mut cx))
    }

    pub fn mul(mut cx: FunctionContext) -> SeriesResult {
        let series = JsSeries::extract_boxed(&mut cx, "_series")?;
        let other = JsSeries::extract_boxed(&mut cx, "other")?;
        let series: JsSeries = (&series.series * &other.series).into();
        Ok(series.into_js_box(&mut cx))
    }
    pub fn div(mut cx: FunctionContext) -> SeriesResult {
        let series = JsSeries::extract_boxed(&mut cx, "_series")?;
        let other = JsSeries::extract_boxed(&mut cx, "other")?;
        let series: JsSeries = (&series.series / &other.series).into();
        Ok(series.into_js_box(&mut cx))
    }
    pub fn head(mut cx: FunctionContext) -> SeriesResult {
        let series = JsSeries::extract_boxed(&mut cx, "_series")?;
        let length: Option<usize> = cx.from_named_parameter("length")?;
        let series: JsSeries = series.series.head(length).into();
        Ok(series.into_js_box(&mut cx))
    }

    pub fn tail(mut cx: FunctionContext) -> SeriesResult {
        let series = JsSeries::extract_boxed(&mut cx, "_series")?;
        let length: Option<usize> = cx.from_named_parameter("length")?;
        let series: JsSeries = series.series.tail(length).into();
        Ok(series.into_js_box(&mut cx))
    }
}
