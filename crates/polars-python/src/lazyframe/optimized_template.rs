use std::collections::HashMap;

use polars::prelude::{LazyFrame, OptimizedTemplate};
use polars_utils::aliases::PlHashMap;
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;

use super::PyLazyFrame;
use crate::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::utils::EnterPolarsExt;

#[pyclass(frozen)]
#[repr(transparent)]
pub struct PyOptimizedTemplate {
    inner: OptimizedTemplate,
}

impl From<OptimizedTemplate> for PyOptimizedTemplate {
    fn from(inner: OptimizedTemplate) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "pymethods")]
#[pymethods]
impl PyOptimizedTemplate {
    /// Bind concrete LazyFrames to placeholders and collect immediately.
    ///
    /// This is the fast path: skips optimization entirely.
    fn bind_and_collect(
        &self,
        py: Python<'_>,
        bindings: HashMap<String, PyLazyFrame>,
    ) -> PyResult<PyDataFrame> {
        let plan_bindings: PlHashMap<PlSmallStr, LazyFrame> = bindings
            .into_iter()
            .map(|(k, v)| (PlSmallStr::from(k.as_str()), v.ldf.read().clone()))
            .collect();
        let df = py.enter_polars(|| {
            self.inner
                .bind_and_collect(plan_bindings)
                .map_err(PyPolarsErr::from)
        })?;
        Ok(df.into())
    }

    /// Bind concrete LazyFrames to placeholders, returning a LazyFrame.
    fn bind(&self, bindings: HashMap<String, PyLazyFrame>) -> PyResult<PyLazyFrame> {
        let plan_bindings: PlHashMap<PlSmallStr, LazyFrame> = bindings
            .into_iter()
            .map(|(k, v)| (PlSmallStr::from(k.as_str()), v.ldf.read().clone()))
            .collect();
        let lf = self.inner.bind(plan_bindings).map_err(PyPolarsErr::from)?;
        Ok(lf.into())
    }

    /// Get the names of all placeholders in this template.
    fn placeholder_names(&self) -> Vec<String> {
        self.inner
            .placeholder_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }
}
