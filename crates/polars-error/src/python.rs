use pyo3::Python;

use crate::PolarsError;

pub struct PyErrWrap(pub pyo3::PyErr);

impl From<pyo3::PyErr> for PolarsError {
    fn from(value: pyo3::PyErr) -> Self {
        PolarsError::Python {
            error: PyErrWrap(value),
        }
    }
}

impl Clone for PyErrWrap {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self(self.0.clone_ref(py)))
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
    type Target = pyo3::PyErr;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for PyErrWrap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
