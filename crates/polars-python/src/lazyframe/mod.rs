mod exitable;
#[cfg(feature = "pymethods")]
mod general;
mod optflags;
#[cfg(feature = "pymethods")]
mod serde;
mod sink;
pub mod visit;
pub mod visitor;

#[cfg(not(target_arch = "wasm32"))]
pub use exitable::PyInProcessQuery;
use polars::prelude::{Engine, LazyFrame, OptFlags};
use pyo3::exceptions::PyValueError;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, FromPyObject, PyAny, PyResult, pyclass};
pub use sink::{PyPartitioning, SinkTarget};

use crate::prelude::Wrap;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyLazyFrame {
    pub ldf: LazyFrame,
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyOptFlags {
    pub inner: OptFlags,
}

impl From<LazyFrame> for PyLazyFrame {
    fn from(ldf: LazyFrame) -> Self {
        PyLazyFrame { ldf }
    }
}

impl From<OptFlags> for PyOptFlags {
    fn from(inner: OptFlags) -> Self {
        PyOptFlags { inner }
    }
}

impl<'py> FromPyObject<'py> for Wrap<Engine> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = ob
            .extract::<PyBackedStr>()?
            .parse()
            .map_err(PyValueError::new_err)?;
        Ok(Wrap(parsed))
    }
}
