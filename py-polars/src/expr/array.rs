use pyo3::pymethods;

use crate::expr::PyExpr;

#[pymethods]
impl PyExpr {
    fn arr_max(&self) -> Self {
        self.inner.clone().arr().max().into()
    }

    fn arr_min(&self) -> Self {
        self.inner.clone().arr().min().into()
    }

    fn arr_sum(&self) -> Self {
        self.inner.clone().arr().sum().into()
    }

    fn arr_unique(&self, maintain_order: bool) -> Self {
        if maintain_order {
            self.inner.clone().arr().unique_stable().into()
        } else {
            self.inner.clone().arr().unique().into()
        }
    }

    fn arr_to_list(&self) -> Self {
        self.inner.clone().arr().to_list().into()
    }
}
