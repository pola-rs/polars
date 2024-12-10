use pyo3::prelude::*;

use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn cat_get_categories(&self) -> Self {
        self.inner.clone().cat().get_categories().into()
    }

    fn cat_len_bytes(&self) -> Self {
        self.inner.clone().cat().len_bytes().into()
    }

    fn cat_len_chars(&self) -> Self {
        self.inner.clone().cat().len_chars().into()
    }

    fn cat_starts_with(&self, sub: Self) -> Self {
        self.inner.clone().cat().starts_with(sub.inner).into()
    }

    fn cat_ends_with(&self, sub: Self) -> Self {
        self.inner.clone().cat().ends_with(sub.inner).into()
    }
}
