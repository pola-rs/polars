use polars::prelude::*;
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn cat_set_ordering(&self, ordering: Wrap<CategoricalOrdering>) -> Self {
        self.inner.clone().cat().set_ordering(ordering.0).into()
    }
}
