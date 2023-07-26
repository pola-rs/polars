use polars::lazy::dsl;
use pyo3::prelude::*;

use crate::PyExpr;

#[pyfunction]
pub fn when(predicate: PyExpr) -> PyWhen {
    PyWhen {
        inner: dsl::when(predicate.inner),
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyWhen {
    inner: dsl::When,
}

#[pyclass]
#[derive(Clone)]
pub struct PyThen {
    inner: dsl::Then,
}

#[pyclass]
#[derive(Clone)]
pub struct PyChainedWhen {
    inner: dsl::ChainedWhen,
}

#[pyclass]
#[derive(Clone)]
pub struct PyChainedThen {
    inner: dsl::ChainedThen,
}

#[pymethods]
impl PyWhen {
    fn then(&self, expr: PyExpr) -> PyThen {
        PyThen {
            inner: self.inner.clone().then(expr.inner),
        }
    }
}

#[pymethods]
impl PyThen {
    fn when(&self, predicate: PyExpr) -> PyChainedWhen {
        PyChainedWhen {
            inner: self.inner.clone().when(predicate.inner),
        }
    }

    fn otherwise(&self, expr: PyExpr) -> PyExpr {
        self.inner.clone().otherwise(expr.inner).into()
    }
}

#[pymethods]
impl PyChainedWhen {
    fn then(&self, expr: PyExpr) -> PyChainedThen {
        PyChainedThen {
            inner: self.inner.clone().then(expr.inner),
        }
    }
}

#[pymethods]
impl PyChainedThen {
    fn when(&self, predicate: PyExpr) -> PyChainedWhen {
        PyChainedWhen {
            inner: self.inner.clone().when(predicate.inner),
        }
    }

    fn otherwise(&self, expr: PyExpr) -> PyExpr {
        self.inner.clone().otherwise(expr.inner).into()
    }
}
