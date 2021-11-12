use polars::prelude::PolarsError;
use thiserror::Error;
use neon::result::Throw;
#[derive(Debug, Error)]
pub enum JsPolarsEr {
    #[error(transparent)]
    Any(#[from] PolarsError),
    #[error("{0}")]
    Other(String),
}

impl std::convert::From<JsPolarsEr> for Throw {
    fn from(err: JsPolarsEr) -> Throw {
      eprintln!("{}", err);
      Throw{}
      
    }
}
