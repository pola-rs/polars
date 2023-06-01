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
    fn array_unique(&self, maintain_order: bool) -> Self {
        if maintain_order {
            self.inner.clone().arr().unique_stable().into()
        } else {
            self.inner.clone().arr().unique().into()
        }
    }
}
