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
use parking_lot::RwLock;
use polars::prelude::{Engine, LazyFrame, OptFlags};
use pyo3::exceptions::PyValueError;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, FromPyObject, PyAny, PyResult, pyclass};
pub use sink::{PyPartitioning, SinkTarget};

use crate::prelude::Wrap;

#[pyclass(frozen)]
#[repr(transparent)]
pub struct PyLazyFrame {
    pub ldf: RwLock<LazyFrame>,
}

impl Clone for PyLazyFrame {
    fn clone(&self) -> Self {
        Self {
            ldf: RwLock::new(self.ldf.read().clone()),
        }
    }
}

impl From<LazyFrame> for PyLazyFrame {
    fn from(ldf: LazyFrame) -> Self {
        PyLazyFrame {
            ldf: RwLock::new(ldf),
        }
    }
}

impl From<PyLazyFrame> for LazyFrame {
    fn from(pldf: PyLazyFrame) -> Self {
        pldf.ldf.into_inner()
    }
}

#[pyclass(frozen)]
#[repr(transparent)]
pub struct PyOptFlags {
    pub inner: RwLock<OptFlags>,
}

impl Clone for PyOptFlags {
    fn clone(&self) -> Self {
        Self {
            inner: RwLock::new(*self.inner.read()),
        }
    }
}

impl From<OptFlags> for PyOptFlags {
    fn from(inner: OptFlags) -> Self {
        PyOptFlags {
            inner: RwLock::new(inner),
        }
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
