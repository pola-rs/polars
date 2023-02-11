use std::fmt::{Debug, Formatter};
use std::io::Error;

use polars::prelude::PolarsError;
use polars_core::error::ArrowError;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyIOError, PyRuntimeError};
use pyo3::prelude::*;
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

impl std::convert::From<std::io::Error> for PyPolarsErr {
    fn from(value: Error) -> Self {
        PyPolarsErr::Other(format!("{value:?}"))
    }
}

impl std::convert::From<PyPolarsErr> for PyErr {
    fn from(err: PyPolarsErr) -> PyErr {
        let default = || PyRuntimeError::new_err(format!("{:?}", &err));

        use PyPolarsErr::*;
        match &err {
            Polars(err) => match err {
                PolarsError::ArrowError(err) => ArrowErrorException::new_err(format!("{err:?}")),
                PolarsError::ColumnNotFound(name) => ColumnNotFoundError::new_err(name.to_string()),
                PolarsError::ComputeError(err) => ComputeError::new_err(err.to_string()),
                PolarsError::Duplicate(err) => DuplicateError::new_err(err.to_string()),
                PolarsError::InvalidOperation(err) => {
                    InvalidOperationError::new_err(err.to_string())
                }
                PolarsError::Io(err) => PyIOError::new_err(err.to_string()),
                PolarsError::NoData(err) => NoDataError::new_err(err.to_string()),
                PolarsError::SchemaFieldNotFound(name) => {
                    SchemaFieldNotFoundError::new_err(name.to_string())
                }
                PolarsError::SchemaMisMatch(err) => SchemaError::new_err(err.to_string()),
                PolarsError::ShapeMisMatch(err) => ShapeError::new_err(err.to_string()),
                PolarsError::StructFieldNotFound(name) => {
                    StructFieldNotFoundError::new_err(name.to_string())
                }
            },
            Arrow(err) => ArrowErrorException::new_err(format!("{err:?}")),
            _ => default(),
        }
    }
}

impl Debug for PyPolarsErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use PyPolarsErr::*;
        match self {
            Polars(err) => write!(f, "{err:?}"),
            Other(err) => write!(f, "BindingsError: {err:?}"),
            Arrow(err) => write!(f, "{err:?}"),
        }
    }
}

create_exception!(exceptions, ArrowErrorException, PyException);
create_exception!(exceptions, ColumnNotFoundError, PyException);
create_exception!(exceptions, ComputeError, PyException);
create_exception!(exceptions, DuplicateError, PyException);
create_exception!(exceptions, InvalidOperationError, PyException);
create_exception!(exceptions, NoDataError, PyException);
create_exception!(exceptions, SchemaError, PyException);
create_exception!(exceptions, SchemaFieldNotFoundError, PyException);
create_exception!(exceptions, ShapeError, PyException);
create_exception!(exceptions, StructFieldNotFoundError, PyException);

#[macro_export]
macro_rules! raise_err(
    ($msg:expr, $err:ident) => {{
        Err(PolarsError::$err($msg.into())).map_err(PyPolarsErr::from)?;
        unreachable!()
    }}
);
