use pyo3::prelude::*;

use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn map_entries(&self) -> Self {
        self.inner.clone().map_().entries().into()
    }

    fn map_keys(&self) -> Self {
        self.inner.clone().map_().keys().into()
    }

    fn map_values(&self) -> Self {
        self.inner.clone().map_().values().into()
    }

    fn map_len(&self) -> Self {
        self.inner.clone().map_().len().into()
    }

    fn map_contains_key(&self, key: PyExpr) -> Self {
        self.inner.clone().map_().contains_key(key.inner).into()
    }

    fn map_get(&self, key: PyExpr) -> Self {
        self.inner.clone().map_().get(key.inner).into()
    }
}
