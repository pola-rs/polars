use polars_utils::python_function::PythonObject;
use pyo3::prelude::*;

use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn name_keep(&self) -> Self {
        self.inner.clone().name().keep().into()
    }

    fn name_map(&self, lambda: PyObject) -> Self {
        self.inner
            .clone()
            .name()
            .map_udf(PythonObject(lambda))
            .into()
    }

    fn name_prefix(&self, prefix: &str) -> Self {
        self.inner.clone().name().prefix(prefix).into()
    }

    fn name_suffix(&self, suffix: &str) -> Self {
        self.inner.clone().name().suffix(suffix).into()
    }

    fn name_to_lowercase(&self) -> Self {
        self.inner.clone().name().to_lowercase().into()
    }

    fn name_to_uppercase(&self) -> Self {
        self.inner.clone().name().to_uppercase().into()
    }

    fn name_map_fields(&self, name_mapper: PyObject) -> Self {
        self.inner
            .clone()
            .name()
            .map_fields_udf(PythonObject(name_mapper))
            .into()
    }

    fn name_prefix_fields(&self, prefix: &str) -> Self {
        self.inner.clone().name().prefix_fields(prefix).into()
    }

    fn name_suffix_fields(&self, suffix: &str) -> Self {
        self.inner.clone().name().suffix_fields(suffix).into()
    }
}
