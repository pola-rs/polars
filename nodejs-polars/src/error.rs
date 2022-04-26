use polars::prelude::PolarsError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum JsPolarsErr {
    #[error(transparent)]
    Any(#[from] PolarsError),
    #[error("{0}")]
    Other(String),
}

impl std::convert::From<JsPolarsErr> for napi::Error {
    fn from(err: JsPolarsErr) -> napi::Error {
        let reason = format!("{}", err);

        napi::Error::from_reason(reason)
    }
}
