use polars::prelude::PolarsError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum JsPolarsEr {
    #[error(transparent)]
    Any(#[from] PolarsError),
    #[error("{0}")]
    Other(String),
}

impl std::convert::From<JsPolarsEr> for napi::Error {
    fn from(err: JsPolarsEr) -> napi::Error {
        let reason = format!("{}", err);

        napi::Error::from_reason(reason)
    }
}
