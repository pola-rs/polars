use pyo3::prelude::*;

use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn struct_field_by_index(&self, index: i64) -> Self {
        self.inner.clone().struct_().field_by_index(index).into()
    }

    fn struct_field_by_name(&self, name: &str) -> Self {
        self.inner.clone().struct_().field_by_name(name).into()
    }

    fn struct_prefix(&self, prefix: &str) -> Self {
        self.inner.clone().struct_().prefix(prefix).into()
    }

    fn struct_rename_fields(&self, names: Vec<String>) -> Self {
        self.inner.clone().struct_().rename_fields(names).into()
    }

    fn struct_suffix(&self, suffix: &str) -> Self {
        self.inner.clone().struct_().suffix(suffix).into()
    }
}
