use std::fmt::{Debug, Formatter};

use polars::prelude::PolarsError;
use pyo3::create_exception;
use pyo3::exceptions::{
    PyAssertionError, PyException, PyIOError, PyIndexError, PyRuntimeError, PyValueError,
};
use pyo3::prelude::*;
use thiserror::Error;

#[derive(Error)]
pub enum PyPolarsErr {
    #[error(transparent)]
    Polars(#[from] PolarsError),
    #[error("{0}")]
    Other(String),
}

impl std::convert::From<PyPolarsErr> for PyErr {
    fn from(err: PyPolarsErr) -> PyErr {
        fn convert(err: PolarsError) -> PyErr {
            match err {
                PolarsError::AssertionError(error) => PyAssertionError::new_err(error.to_string()),
                PolarsError::ComputeError(err) => ComputeError::new_err(err.to_string()),
                PolarsError::NoData(err) => NoDataError::new_err(err.to_string()),
                PolarsError::ShapeMismatch(err) => ShapeError::new_err(err.to_string()),
                PolarsError::SchemaMismatch(err) => SchemaError::new_err(err.to_string()),
                PolarsError::IO { error, .. } => PyIOError::new_err(error.to_string()),
                PolarsError::OutOfBounds(err) => PyIndexError::new_err(err.to_string()),
                PolarsError::InvalidOperation(err) => PyValueError::new_err(err.to_string()),
                PolarsError::Duplicate(err) => DuplicateError::new_err(err.to_string()),
                PolarsError::ColumnNotFound(err) => ColumnNotFound::new_err(err.to_string()),
                PolarsError::SchemaFieldNotFound(err) => {
                    SchemaFieldNotFound::new_err(err.to_string())
                },
                PolarsError::StructFieldNotFound(err) => {
                    StructFieldNotFound::new_err(err.to_string())
                },
                PolarsError::StringCacheMismatch(err) => {
                    StringCacheMismatchError::new_err(err.to_string())
                },
                PolarsError::SQLInterface(err) => SQLInterface::new_err(err.to_string()),
                PolarsError::SQLSyntax(err) => SQLSyntax::new_err(err.to_string()),
                PolarsError::Context { error, .. } => convert(*error),
                PolarsError::Python { error } => error.0,
            }
        }

        use PyPolarsErr::*;
        match err {
            Polars(err) => convert(err),
            _ => PyRuntimeError::new_err(format!("{err:?}")),
        }
    }
}

impl Debug for PyPolarsErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use PyPolarsErr::*;
        match self {
            Polars(err) => write!(f, "{err:?}"),
            Other(err) => write!(f, "BindingsError: {err:?}"),
        }
    }
}

create_exception!(exceptions, AssertionError, PyException);
create_exception!(exceptions, ColumnNotFound, PyException);
create_exception!(exceptions, SchemaFieldNotFound, PyException);
create_exception!(exceptions, StructFieldNotFound, PyException);
create_exception!(exceptions, ComputeError, PyException);
create_exception!(exceptions, NoDataError, PyException);
create_exception!(exceptions, ShapeError, PyException);
create_exception!(exceptions, SchemaError, PyException);
create_exception!(exceptions, DuplicateError, PyException);
create_exception!(exceptions, StringCacheMismatchError, PyException);
create_exception!(exceptions, SQLInterface, PyException);
create_exception!(exceptions, SQLSyntax, PyException);
