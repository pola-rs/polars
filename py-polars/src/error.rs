use polars::prelude::PolarsError;
use pyo3::{exceptions::RuntimeError, prelude::*};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PyPolarsEr {
    #[error(transparent)]
    Any(#[from] PolarsError),
    #[error("{0}")]
    Other(String),
}

impl std::convert::From<PyPolarsEr> for PyErr {
    fn from(err: PyPolarsEr) -> PyErr {
        RuntimeError::py_err(format!("{:?}", err))
    }
}
