use pyo3::prelude::*;

use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn cat_get_categories(&self) -> Self {
        self.inner.clone().cat().get_categories().into()
    }

    fn cat_to_local(&self) -> Self {
        self.inner.clone().cat().to_local().into()
    }
}
