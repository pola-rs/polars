use pyo3::prelude::*;

use crate::error::PyPolarsErr;
use crate::expr::ToExprs;
use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn struct_field_by_index(&self, index: i64) -> Self {
        self.inner.clone().struct_().field_by_index(index).into()
    }

    fn struct_field_by_name(&self, name: &str) -> Self {
        self.inner.clone().struct_().field_by_name(name).into()
    }

    fn struct_multiple_fields(&self, names: Vec<String>) -> Self {
        self.inner.clone().struct_().field_by_names(&names).into()
    }

    fn struct_rename_fields(&self, names: Vec<String>) -> Self {
        self.inner.clone().struct_().rename_fields(names).into()
    }

    #[cfg(feature = "json")]
    fn struct_json_encode(&self) -> Self {
        self.inner.clone().struct_().json_encode().into()
    }

    fn struct_with_fields(&self, fields: Vec<PyExpr>) -> PyResult<Self> {
        let fields = fields.to_exprs();
        let e = self
            .inner
            .clone()
            .struct_()
            .with_fields(fields)
            .map_err(PyPolarsErr::from)?;
        Ok(e.into())
    }
}
