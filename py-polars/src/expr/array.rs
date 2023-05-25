use pyo3::pymethods;

use crate::expr::PyExpr;

#[pymethods]
impl PyExpr {
    fn array_max(&self) -> Self {
        self.inner.clone().arr().max().into()
    }

    fn array_min(&self) -> Self {
        self.inner.clone().arr().min().into()
    }

    fn array_sum(&self) -> Self {
        self.inner.clone().arr().sum().into()
    }
}
