use polars::prelude::PolarsError;
use thiserror::Error;
use wasm_bindgen::prelude::JsValue;

#[derive(Debug, Error)]
pub enum JsPolarsErr {
    #[error(transparent)]
    Any(#[from] PolarsError),
    #[error("{0}")]
    Other(String),
}

impl std::convert::From<JsPolarsErr> for JsValue {
    fn from(err: JsPolarsErr) -> JsValue {
        let reason = format!("{}", err);
        js_sys::Error::new(&reason).into()
    }
}
