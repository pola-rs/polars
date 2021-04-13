use polars_core::prelude::{
    Series as PSeries, Utf8Chunked, NewChunkedArray, Float64Chunked, IntoSeries
};
use std::ops::{BitAnd, BitOr};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Series {
    series: PSeries,
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
            let series = Float64Chunked::new_from_opt_iter(name, values.iter().map(|v| v.as_f64())).into_series();
            Series {
                series
            }
        } else if first.as_string().is_some() {
                let series = Utf8Chunked::new_from_opt_iter(name, values.iter().map(|v| v.as_string())).into_series();
                Series {
                    series
                }
        } else {
            unimplemented!()
        }
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

    pub fn cum_sum(&self, reverse: bool) -> Self {
        self.series.cum_sum(reverse).into()
    }

    pub fn cum_max(&self, reverse: bool) -> Self {
        self.series.cum_max(reverse).into()
    }

    pub fn cum_min(&self, reverse: bool) -> Self {
        self.series.cum_min(reverse).into()
    }

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

    pub fn n_chunks(&self) -> usize {
        self.series.n_chunks()
    }

    pub fn limit(&self, num_elements: usize) -> Self {
        let series = self.series.limit(num_elements);
        series.into()
    }
}
