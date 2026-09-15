use polars::lazy::dsl;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::PyExpr;
use crate::error::PyPolarsErr;

#[pyfunction]
pub fn when(condition: PyExpr) -> PyWhen {
    PyWhen {
        inner: dsl::when(condition.inner),
    }
}

#[pyclass(frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyWhen {
    inner: dsl::When,
}

#[pyclass(skip_from_py_object)] // Not marked as frozen for pickling, but that's the only &mut self method.
#[derive(Clone)]
pub struct PyThen {
    inner: dsl::Then,
}

#[pyclass(frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyChainedWhen {
    inner: dsl::ChainedWhen,
}

#[pyclass(skip_from_py_object)] // Not marked as frozen for pickling, but that's the only &mut self method.
#[derive(Clone)]
pub struct PyChainedThen {
    inner: dsl::ChainedThen,
}

#[pymethods]
impl PyWhen {
    fn then(&self, statement: PyExpr) -> PyThen {
        PyThen {
            inner: self.inner.clone().then(statement.inner),
        }
    }
}

#[pymethods]
impl PyThen {
    fn when(&self, condition: PyExpr) -> PyChainedWhen {
        PyChainedWhen {
            inner: self.inner.clone().when(condition.inner),
        }
    }

    fn otherwise(&self, statement: PyExpr) -> PyExpr {
        self.inner.clone().otherwise(statement.inner).into()
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mut bytes: Vec<u8> = vec![];
        self.inner
            .serialize_compact_into(&mut bytes)
            .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn __setstate__(&mut self, state: &Bound<PyAny>) -> PyResult<()> {
        let bytes = state.extract::<PyBackedBytes>()?;
        self.inner = dsl::Then::deserialize_compact_from(&mut &*bytes)
            .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
        Ok(())
    }
}

#[pymethods]
impl PyChainedWhen {
    fn then(&self, statement: PyExpr) -> PyChainedThen {
        PyChainedThen {
            inner: self.inner.clone().then(statement.inner),
        }
    }
}

#[pymethods]
impl PyChainedThen {
    fn when(&self, condition: PyExpr) -> PyChainedWhen {
        PyChainedWhen {
            inner: self.inner.clone().when(condition.inner),
        }
    }

    fn otherwise(&self, statement: PyExpr) -> PyExpr {
        self.inner.clone().otherwise(statement.inner).into()
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mut bytes: Vec<u8> = vec![];
        self.inner
            .serialize_compact_into(&mut bytes)
            .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn __setstate__(&mut self, state: &Bound<PyAny>) -> PyResult<()> {
        let bytes = state.extract::<PyBackedBytes>()?;
        self.inner = dsl::ChainedThen::deserialize_compact_from(&mut &*bytes)
            .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
        Ok(())
    }
}
