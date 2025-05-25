use pyo3::prelude::*;
use polars_plan::dsl::struct_::StructNameSpace;

use crate::PyExpr;
use crate::error::PyPolarsErr;
use crate::expr::ToExprs;

#[pymethods]
impl PyExpr {
    fn struct_field_by_index(&self, index: i64) -> Self {
        StructNameSpace(self.inner.clone()).field_by_index(index).into()
    }

    fn struct_field_by_name(&self, name: &str) -> Self {
        StructNameSpace(self.inner.clone()).field_by_name(name).into()
    }

    fn struct_multiple_fields(&self, names: Vec<String>) -> Self {
        StructNameSpace(self.inner.clone()).field_by_names(&names).into()
    }

    fn struct_rename_fields(&self, names: Vec<String>) -> Self {
        StructNameSpace(self.inner.clone()).rename_fields(names).into()
    }

    #[cfg(feature = "json")]
    fn struct_json_encode(&self) -> Self {
        StructNameSpace(self.inner.clone()).json_encode().into()
    }

    fn struct_with_fields(&self, fields: Vec<PyExpr>) -> PyResult<Self> {
        let fields = fields.to_exprs();
        let e = StructNameSpace(self.inner.clone())
            .with_fields(fields)
            .map_err(PyPolarsErr::from)?;
        Ok(e.into())
    }
}
