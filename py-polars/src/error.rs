use polars::prelude::PolarsError;
use polars_core::error::ArrowError;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::fmt::{Debug, Formatter};
use thiserror::Error;

#[derive(Error)]
pub enum PyPolarsErr {
    #[error(transparent)]
    Polars(#[from] PolarsError),
    #[error("{0}")]
    Other(String),
    #[error(transparent)]
    Arrow(#[from] ArrowError),
}

impl std::convert::From<PyPolarsErr> for PyErr {
    fn from(err: PyPolarsErr) -> PyErr {
        PyRuntimeError::new_err(format!("{:?}", err))
    }
}

impl Debug for PyPolarsErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use PyPolarsErr::*;
        match self {
            Polars(err) => write!(f, "{:?}", err),
            Other(err) => write!(f, "BindingsError: {:?}", err),
            Arrow(err) => write!(f, "{:?}", err),
        }
    }
}
