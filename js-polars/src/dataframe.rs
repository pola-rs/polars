use polars_core::prelude::DataFrame as PDataFrame;
use polars_core::prelude::Series as PSeries;
use super::{
    JsPolarsError,
    series::*,
};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[wasm_bindgen]
pub struct DataFrame {
    df: PDataFrame,
}

impl From<PDataFrame> for DataFrame {
    fn from(df: PDataFrame) -> Self {
        Self { df }
    }
}


#[wasm_bindgen]
impl DataFrame {

    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        PDataFrame::new_no_checks(vec![]).into()
    }

    pub fn assign(&self, series: Series) -> PolarsResult<DataFrame, JsValue> {
        let mut df = self.df.clone();
        df.with_column(series.series).map_err(JsPolarsError::from)?;
        Ok(df.into())
    }
}