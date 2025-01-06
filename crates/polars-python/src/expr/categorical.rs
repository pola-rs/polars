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

    fn cat_starts_with(&self, prefix: String) -> Self {
        self.inner.clone().cat().starts_with(prefix).into()
    }

    fn cat_ends_with(&self, suffix: String) -> Self {
        self.inner.clone().cat().ends_with(suffix).into()
    }

    #[pyo3(signature = (pat, literal, strict))]
    #[cfg(feature = "regex")]
    fn cat_contains(&self, pat: &str, literal: Option<bool>, strict: bool) -> Self {
        let lit = literal.unwrap_or(false);
        self.inner.clone().cat().contains(pat, lit, strict).into()
    }

    #[cfg(feature = "find_many")]
    fn cat_contains_any(&self, patterns: PyExpr, ascii_case_insensitive: bool) -> Self {
        self.inner
            .clone()
            .cat()
            .contains_any(patterns.inner, ascii_case_insensitive)
            .into()
    }
}
