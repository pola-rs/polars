use std::io::BufWriter;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::expr::ToPyExprs;
use crate::file::get_file_like;
use crate::prelude::polars_err;
use crate::{PyExpr, PyPolarsErr};

#[pymethods]
impl PyExpr {
    fn meta_eq(&self, other: Self) -> bool {
        self.inner == other.inner
    }

    fn meta_pop(&self) -> Vec<Self> {
        self.inner.clone().meta().pop().to_pyexprs()
    }

    fn meta_root_names(&self) -> Vec<String> {
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

    fn meta_undo_aliases(&self) -> Self {
        self.inner.clone().meta().undo_aliases().into()
    }

    fn meta_has_multiple_outputs(&self) -> bool {
        self.inner.clone().meta().has_multiple_outputs()
    }

    fn meta_is_column(&self) -> bool {
        self.inner.clone().meta().is_column()
    }

    fn meta_is_regex_projection(&self) -> bool {
        self.inner.clone().meta().is_regex_projection()
    }

    fn meta_is_column_selection(&self, allow_aliasing: bool) -> bool {
        self.inner
            .clone()
            .meta()
            .is_column_selection(allow_aliasing)
    }

    fn _meta_selector_add(&self, other: PyExpr) -> PyResult<PyExpr> {
        let out = self
            .inner
            .clone()
            .meta()
            ._selector_add(other.inner)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn _meta_selector_and(&self, other: PyExpr) -> PyResult<PyExpr> {
        let out = self
            .inner
            .clone()
            .meta()
            ._selector_and(other.inner)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn _meta_selector_sub(&self, other: PyExpr) -> PyResult<PyExpr> {
        let out = self
            .inner
            .clone()
            .meta()
            ._selector_sub(other.inner)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn _meta_selector_xor(&self, other: PyExpr) -> PyResult<PyExpr> {
        let out = self
            .inner
            .clone()
            .meta()
            ._selector_xor(other.inner)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn _meta_as_selector(&self) -> PyExpr {
        self.inner.clone().meta()._into_selector().into()
    }

    #[cfg(all(feature = "json", feature = "serde_json"))]
    fn serialize(&self, py_f: PyObject) -> PyResult<()> {
        let file = BufWriter::new(get_file_like(py_f, true)?);
        serde_json::to_writer(file, &self.inner)
            .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
        Ok(())
    }

    #[staticmethod]
    #[cfg(feature = "json")]
    fn deserialize(py_f: PyObject) -> PyResult<PyExpr> {
        // it is faster to first read to memory and then parse: https://github.com/serde-rs/json/issues/160
        // so don't bother with files.
        let mut json = String::new();
        let _ = get_file_like(py_f, false)?
            .read_to_string(&mut json)
            .unwrap();

        // SAFETY:
        // We skipped the serializing/deserializing of the static in lifetime in `DataType`
        // so we actually don't have a lifetime at all when serializing.

        // &str still has a lifetime. But it's ok, because we drop it immediately
        // in this scope.
        let json = unsafe { std::mem::transmute::<&'_ str, &'static str>(json.as_str()) };

        let inner: polars_lazy::prelude::Expr = serde_json::from_str(json).map_err(|_| {
            let msg = "could not deserialize input into an expression";
            PyPolarsErr::from(polars_err!(ComputeError: msg))
        })?;
        Ok(PyExpr { inner })
    }

    fn meta_tree_format(&self) -> PyResult<String> {
        let e = self
            .inner
            .clone()
            .meta()
            .into_tree_formatter()
            .map_err(PyPolarsErr::from)?;
        Ok(format!("{e}"))
    }
}
