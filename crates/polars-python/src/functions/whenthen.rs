use polars::lazy::dsl;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::PyExpr;

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
        crate::conversion::serde_pickle(&self.inner, py)
    }

    fn __setstate__(&mut self, state: &Bound<PyAny>) -> PyResult<()> {
        crate::conversion::serde_unpickle(&mut self.inner, state)
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
}
