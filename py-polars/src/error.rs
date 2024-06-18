use std::fmt::{Debug, Formatter};
use std::io::{Error, ErrorKind};

use polars::prelude::PolarsError;
use polars_error::PolarsWarning;
use pyo3::exceptions::{
    PyFileExistsError, PyFileNotFoundError, PyIOError, PyPermissionError, PyRuntimeError,
    PyUserWarning,
};
use pyo3::prelude::*;
use pyo3::PyTypeInfo;
use thiserror::Error;

use crate::exceptions::{
    CategoricalRemappingWarning, ColumnNotFoundError, ComputeError, DuplicateError,
    InvalidOperationError, MapWithoutReturnDtypeWarning, NoDataError, OutOfBoundsError,
    SQLInterfaceError, SQLSyntaxError, SchemaError, SchemaFieldNotFoundError, ShapeError,
    StringCacheMismatchError, StructFieldNotFoundError,
};
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
        match err {
            Polars(err) => match err {
                PolarsError::ColumnNotFound(name) => ColumnNotFoundError::new_err(name.to_string()),
                PolarsError::ComputeError(err) => ComputeError::new_err(err.to_string()),
                PolarsError::Duplicate(err) => DuplicateError::new_err(err.to_string()),
                PolarsError::InvalidOperation(err) => {
                    InvalidOperationError::new_err(err.to_string())
                },
                PolarsError::IO { error, msg } => {
                    let msg = if let Some(msg) = msg {
                        msg.to_string()
                    } else {
                        error.to_string()
                    };
                    match error.kind() {
                        ErrorKind::NotFound => PyFileNotFoundError::new_err(msg),
                        ErrorKind::PermissionDenied => PyPermissionError::new_err(msg),
                        ErrorKind::AlreadyExists => PyFileExistsError::new_err(msg),
                        _ => PyIOError::new_err(msg),
                    }
                },
                PolarsError::NoData(err) => NoDataError::new_err(err.to_string()),
                PolarsError::OutOfBounds(err) => OutOfBoundsError::new_err(err.to_string()),
                PolarsError::SQLInterface(name) => SQLInterfaceError::new_err(name.to_string()),
                PolarsError::SQLSyntax(name) => SQLSyntaxError::new_err(name.to_string()),
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
                PolarsError::Context { .. } => {
                    let tmp = PyPolarsErr::Polars(err.context_trace());
                    PyErr::from(tmp)
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
                CategoricalRemappingWarning::type_object_bound(py).to_object(py)
            },
            PolarsWarning::MapWithoutReturnDtypeWarning => {
                MapWithoutReturnDtypeWarning::type_object_bound(py).to_object(py)
            },
            PolarsWarning::UserWarning => PyUserWarning::type_object_bound(py).to_object(py),
        }
    }
}
