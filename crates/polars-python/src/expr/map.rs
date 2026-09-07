use pyo3::prelude::*;

use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn map_entries(&self) -> Self {
        self.inner.clone().map_().entries().into()
    }
}
