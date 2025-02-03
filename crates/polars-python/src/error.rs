use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::io::{Error as IoError, ErrorKind};

use polars::prelude::PolarsError;
use polars_error::PolarsWarning;
use pyo3::exceptions::{
    PyDeprecationWarning, PyFileExistsError, PyFileNotFoundError, PyIOError, PyPermissionError,
    PyRuntimeError, PyUserWarning,
};
use pyo3::prelude::*;
use pyo3::PyTypeInfo;

use crate::exceptions::{
    CategoricalRemappingWarning, ColumnNotFoundError, ComputeError, DuplicateError,
    InvalidOperationError, MapWithoutReturnDtypeWarning, NoDataError, OutOfBoundsError,
    SQLInterfaceError, SQLSyntaxError, SchemaError, SchemaFieldNotFoundError, ShapeError,
    StringCacheMismatchError, StructFieldNotFoundError,
};
use crate::Wrap;

pub enum PyPolarsErr {
    Polars(PolarsError),
    Python(PyErr),
    Other(String),
}

impl Error for PyPolarsErr {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Polars(err) => Some(err),
            Self::Python(err) => Some(err),
            Self::Other(_) => None,
        }
    }
}

impl std::fmt::Display for PyPolarsErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Polars(err) => Display::fmt(err, f),
            Self::Python(err) => Display::fmt(err, f),
            Self::Other(err) => write!(f, "{err}"),
        }
    }
}

impl From<PolarsError> for PyPolarsErr {
    fn from(err: PolarsError) -> Self {
        PyPolarsErr::Polars(err)
    }
}

impl From<PyErr> for PyPolarsErr {
    fn from(err: PyErr) -> Self {
        PyPolarsErr::Python(err)
    }
}

impl From<IoError> for PyPolarsErr {
    fn from(err: IoError) -> Self {
        PyPolarsErr::Other(format!("{err:?}"))
    }
}

impl From<PyPolarsErr> for PyErr {
    fn from(err: PyPolarsErr) -> PyErr {
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
            Python(err) => err,
            err => PyRuntimeError::new_err(format!("{:?}", &err)),
        }
    }
}

impl Debug for PyPolarsErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use PyPolarsErr::*;
        match self {
            Polars(err) => write!(f, "{err:?}"),
            Python(err) => write!(f, "{err:?}"),
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

impl<'py> IntoPyObject<'py> for Wrap<PolarsWarning> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self.0 {
            PolarsWarning::CategoricalRemappingWarning => {
                Ok(CategoricalRemappingWarning::type_object(py).into_any())
            },
            PolarsWarning::MapWithoutReturnDtypeWarning => {
                Ok(MapWithoutReturnDtypeWarning::type_object(py).into_any())
            },
            PolarsWarning::UserWarning => Ok(PyUserWarning::type_object(py).into_any()),
            PolarsWarning::Deprecation => Ok(PyDeprecationWarning::type_object(py).into_any()),
        }
    }
}
