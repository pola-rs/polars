use std::fmt::{Debug, Formatter};
use std::io::{Error, ErrorKind};

use polars::prelude::PolarsError;
use polars_error::PolarsWarning;
use pyo3::exceptions::{
    PyException, PyFileExistsError, PyFileNotFoundError, PyIOError, PyPermissionError,
    PyRuntimeError, PyUserWarning, PyWarning,
};
use pyo3::prelude::*;
use pyo3::{create_exception, PyTypeInfo};
use thiserror::Error;

use crate::Wrap;
#[derive(Error)]
pub enum PyPolarsErr {
    #[error(transparent)]
    Polars(#[from] PolarsError),
    #[error("{0}")]
    Other(String),
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
                PolarsError::ColumnNotFound(name) => ColumnNotFoundError::new_err(name.to_string()),
                PolarsError::ComputeError(err) => ComputeError::new_err(err.to_string()),
                PolarsError::Duplicate(err) => DuplicateError::new_err(err.to_string()),
                PolarsError::InvalidOperation(err) => {
                    InvalidOperationError::new_err(err.to_string())
                },
                PolarsError::Io(err) => match err.kind() {
                    ErrorKind::NotFound => PyFileNotFoundError::new_err(err.to_string()),
                    ErrorKind::PermissionDenied => PyPermissionError::new_err(err.to_string()),
                    ErrorKind::AlreadyExists => PyFileExistsError::new_err(err.to_string()),
                    _ => PyIOError::new_err(err.to_string()),
                },
                PolarsError::NoData(err) => NoDataError::new_err(err.to_string()),
                PolarsError::OutOfBounds(err) => OutOfBoundsError::new_err(err.to_string()),
                PolarsError::SchemaFieldNotFound(name) => {
                    SchemaFieldNotFoundError::new_err(name.to_string())
                },
                PolarsError::SchemaMismatch(err) => SchemaError::new_err(err.to_string()),
                PolarsError::ShapeMismatch(err) => ShapeError::new_err(err.to_string()),
                PolarsError::StringCacheMismatch(err) => {
                    StringCacheMismatchError::new_err(err.to_string())
                },
                PolarsError::StructFieldNotFound(name) => {
                    StructFieldNotFoundError::new_err(name.to_string())
                },
            },
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
        }
    }
}

create_exception!(polars.exceptions, ColumnNotFoundError, PyException);
create_exception!(polars.exceptions, ComputeError, PyException);
create_exception!(polars.exceptions, DuplicateError, PyException);
create_exception!(polars.exceptions, InvalidOperationError, PyException);
create_exception!(polars.exceptions, NoDataError, PyException);
create_exception!(polars.exceptions, OutOfBoundsError, PyException);
create_exception!(polars.exceptions, SchemaError, PyException);
create_exception!(polars.exceptions, SchemaFieldNotFoundError, PyException);
create_exception!(polars.exceptions, ShapeError, PyException);
create_exception!(polars.exceptions, StringCacheMismatchError, PyException);
create_exception!(polars.exceptions, StructFieldNotFoundError, PyException);
create_exception!(polars.exceptions, CategoricalRemappingWarning, PyWarning);

#[macro_export]
macro_rules! raise_err(
    ($msg:expr, $err:ident) => {{
        Err(PolarsError::$err($msg.into())).map_err(PyPolarsErr::from)?;
        unreachable!()
    }}
);

impl IntoPy<PyObject> for Wrap<PolarsWarning> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0 {
            PolarsWarning::CategoricalRemappingWarning => {
                CategoricalRemappingWarning::type_object(py).to_object(py)
            },
            PolarsWarning::UserWarning => PyUserWarning::type_object(py).to_object(py),
        }
    }
}
