use polars_core::prelude::{
    Float64Chunked, IntoSeries, NewChunkedArray, Series as PSeries, Utf8Chunked,
};
use std::ops::{BitAnd, BitOr};
use wasm_bindgen::prelude::*;
use crate::{
    console_log,
    log,
};

#[wasm_bindgen]
#[repr(transparent)]
pub struct Series {
    pub(crate) series: PSeries,
}

impl From<PSeries> for Series {
    fn from(series: PSeries) -> Self {
        Self { series }
    }
}

#[wasm_bindgen]
impl Series {
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str, values: Box<[JsValue]>) -> Series {
        let first = &values[0];
        if first.as_f64().is_some() {
            let series = Float64Chunked::new_from_opt_iter(name, values.iter().map(|v| v.as_f64()))
                .into_series();
            Series { series }
        } else if first.as_string().is_some() {
            let series = Utf8Chunked::new_from_opt_iter(name, values.iter().map(|v| v.as_string()))
                .into_series();
            Series { series }
        } else {
            unimplemented!()
        }
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        format!("{}", self.series)
    }

    pub fn log(&self) {
        console_log!("{}", self.series)
    }

    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) ->  String {
        let mut series_fmt = String::with_capacity(10);
        series_fmt.push('[');
        let n  = std::cmp::min(self.series.len(), 5);
        for i in 0..n {
            let val = self.series.get(i);
            if i < n -1 {
                series_fmt.push_str(&format!("{}, ", val))
            } else {
                series_fmt.push_str(&format!("{}", val))
            }
        }
        series_fmt.push(']');

        format!(r#"{{ {}: {} }}"#, self.series.name(), series_fmt)
    }

    pub fn rechunk(&mut self, in_place: bool) -> Option<Series> {
        let series = self.series.rechunk();
        if in_place {
            self.series = series;
            None
        } else {
            Some(series.into())
        }
    }

    pub fn bitand(&self, other: &Series) -> Self {
        let s = self
            .series
            .bool()
            .expect("boolean")
            .bitand(other.series.bool().expect("boolean"))
            .into_series();
        s.into()
    }

    pub fn bitor(&self, other: &Series) -> Self {
        let s = self
            .series
            .bool()
            .expect("boolean")
            .bitor(other.series.bool().expect("boolean"))
            .into_series();
        s.into()
    }

    #[wasm_bindgen(js_name = cumSum)]
    pub fn cumsum(&self, reverse: bool) -> Self {
        self.series.cumsum(reverse).into()
    }

    #[wasm_bindgen(js_name = cumMax)]
    pub fn cummax(&self, reverse: bool) -> Self {
        self.series.cummax(reverse).into()
    }

    #[wasm_bindgen(js_name = cumMin)]
    pub fn cummin(&self, reverse: bool) -> Self {
        self.series.cummin(reverse).into()
    }

    #[wasm_bindgen(js_name = chunkLengths)]
    pub fn chunk_lengths(&self) -> Vec<usize> {
        self.series.chunk_lengths().clone()
    }

    pub fn name(&self) -> String {
        self.series.name().into()
    }

    pub fn rename(&mut self, name: &str) {
        self.series.rename(name);
    }

    pub fn mean(&self) -> Option<f64> {
        self.series.mean()
    }

    #[wasm_bindgen(js_name = nChunks)]
    pub fn n_chunks(&self) -> usize {
        self.series.n_chunks()
    }

    pub fn limit(&self, num_elements: usize) -> Self {
        let series = self.series.limit(num_elements);
        series.into()
    }

    pub fn slice(&self, offset: i64, length: usize) -> Self {
        let series = self.series.slice(offset, length);
        series.into()
    }

    pub fn append(&mut self, other: &Series) -> PolarsResult<(), JsValue> {
        let res = self.series.append(&other.series);
        if let Err(e) = res {
            Err(format!("{:?}", e).into())
        } else {
            Ok(())
        }
    }

    pub fn filter(&self, filter: &Series) -> PolarsResult<Series, JsValue> {
        let filter_series = &filter.series;
        if let Ok(ca) = filter_series.bool() {
            let series = self.series.filter(ca).unwrap();
            Ok(series.into())
        } else {
            Err("Expected a boolean mask".into())
        }
    }

    pub fn add(&self, other: &Series) -> Self {
        (&self.series + &other.series).into()
    }

    pub fn sub(&self, other: &Series) -> Self {
        (&self.series - &other.series).into()
    }

    pub fn mul(&self, other: &Series) -> Self {
        (&self.series * &other.series).into()
    }

    pub fn div(&self, other: &Series) -> Self {
        (&self.series / &other.series).into()
    }

    pub fn head(&self, length: Option<usize>) -> Self {
        (self.series.head(length)).into()
    }

    pub fn tail(&self, length: Option<usize>) -> Self {
        (self.series.tail(length)).into()
    }

    pub fn sort(&mut self, reverse: bool) -> Self {
        (self.series.sort(reverse)).into()
    }

    #[wasm_bindgen(js_name = argSort)]
    pub fn argsort(&self, reverse: bool) -> Self {
        self.series.argsort(reverse).into_series().into()
    }
}
