use pyo3::prelude::*;

use super::*;
use crate::{PyExpr, PyPolarsErr};

#[pymethods]
impl PyExpr {
    fn meta_pop(&self) -> Vec<PyExpr> {
        self.inner.clone().meta().pop().to_pyexprs()
    }

    fn meta_eq(&self, other: PyExpr) -> bool {
        self.inner == other.inner
    }

    fn meta_roots(&self) -> Vec<String> {
        self.inner
            .clone()
            .meta()
            .root_names()
            .iter()
            .map(|name| name.to_string())
            .collect()
    }

    fn meta_output_name(&self) -> PyResult<String> {
        let name = self
            .inner
            .clone()
            .meta()
            .output_name()
            .map_err(PyPolarsErr::from)?;
        Ok(name.to_string())
    }

    fn meta_undo_aliases(&self) -> PyExpr {
        self.inner.clone().meta().undo_aliases().into()
    }

    fn meta_has_multiple_outputs(&self) -> bool {
        self.inner.clone().meta().has_multiple_outputs()
    }
    fn meta_is_regex_projection(&self) -> bool {
        self.inner.clone().meta().is_regex_projection()
    }
}
