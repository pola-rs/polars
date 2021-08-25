use polars::prelude::PolarsError;
use polars_core::error::ArrowError;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PyPolarsEr {
    #[error(transparent)]
    Any(#[from] PolarsError),
    #[error("{0}")]
    Other(String),
    #[error(transparent)]
    ArrowError(#[from] ArrowError),
}

impl std::convert::From<PyPolarsEr> for PyErr {
    fn from(err: PyPolarsEr) -> PyErr {
        PyRuntimeError::new_err(format!("{:?}", err))
    }
}
