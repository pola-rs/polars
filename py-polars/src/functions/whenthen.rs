use polars::lazy::dsl;
use pyo3::prelude::*;

use crate::PyExpr;

#[pyfunction]
pub fn when(predicate: PyExpr) -> When {
    When { predicate }
}

#[pyclass]
#[derive(Clone)]
pub struct When {
    predicate: PyExpr,
}

#[pyclass]
#[derive(Clone)]
pub struct WhenThen {
    predicate: PyExpr,
    then: PyExpr,
}

#[pyclass]
#[derive(Clone)]
pub struct WhenThenThen {
    inner: dsl::WhenThenThen,
}

#[pymethods]
impl When {
    fn then(&self, expr: PyExpr) -> WhenThen {
        WhenThen {
            predicate: self.predicate.clone(),
            then: expr,
        }
    }
}

#[pymethods]
impl WhenThen {
    fn when(&self, predicate: PyExpr) -> WhenThenThen {
        let e = dsl::when(self.predicate.inner.clone())
            .then(self.then.inner.clone())
            .when(predicate.inner);
        WhenThenThen { inner: e }
    }

    fn otherwise(&self, expr: PyExpr) -> PyExpr {
        dsl::ternary_expr(
            self.predicate.inner.clone(),
            self.then.inner.clone(),
            expr.inner,
        )
        .into()
    }
}

#[pymethods]
impl WhenThenThen {
    fn when(&self, predicate: PyExpr) -> Self {
        Self {
            inner: self.inner.clone().when(predicate.inner),
        }
    }
    fn then(&self, expr: PyExpr) -> Self {
        Self {
            inner: self.inner.clone().then(expr.inner),
        }
    }
    fn otherwise(&self, expr: PyExpr) -> PyExpr {
        self.inner.clone().otherwise(expr.inner).into()
    }
}
