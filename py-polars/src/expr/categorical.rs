use polars::prelude::*;
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn cat_set_ordering(&self, ordering: Wrap<CategoricalOrdering>) -> Self {
        self.inner
            .clone()
            .cast(DataType::Categorical(None, ordering.0))
            .into()
    }

    fn cat_get_categories(&self) -> Self {
        self.inner.clone().cat().get_categories().into()
    }
}
