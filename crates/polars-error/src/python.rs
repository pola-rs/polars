use pyo3::pyclass::PyClassGuardError;
use pyo3::{PyErr, Python};

use crate::PolarsError;

pub struct PyErrWrap(pub PyErr);

impl From<PyErr> for PolarsError {
    fn from(value: PyErr) -> Self {
        PolarsError::Python {
            error: PyErrWrap(value),
        }
    }
}

impl<'a, 'py> From<PyClassGuardError<'a, 'py>> for PolarsError {
    fn from(err: PyClassGuardError) -> Self {
        PolarsError::from(PyErr::from(err))
    }
}

impl Clone for PyErrWrap {
    fn clone(&self) -> Self {
        Python::attach(|py| Self(self.0.clone_ref(py)))
    }
}

impl std::fmt::Debug for PyErrWrap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl std::fmt::Display for PyErrWrap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::ops::Deref for PyErrWrap {
    type Target = PyErr;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for PyErrWrap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
