use polars::lazy::dsl;
use pyo3::prelude::*;

use crate::PyExpr;

#[pyfunction]
pub fn when(condition: PyExpr) -> PyWhen {
    PyWhen {
        inner: dsl::when(condition.inner),
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
