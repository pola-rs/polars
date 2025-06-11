use polars::prelude::{DataType, DataTypeExpr, Schema};
use pyo3::{PyResult, pyclass};

use super::PyExpr;
use crate::error::PyPolarsErr;
use crate::prelude::Wrap;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyDataTypeExpr {
    pub inner: DataTypeExpr,
}

impl From<DataTypeExpr> for PyDataTypeExpr {
    fn from(expr: DataTypeExpr) -> Self {
        PyDataTypeExpr { inner: expr }
    }
}

#[cfg(feature = "pymethods")]
#[pyo3::pymethods]
impl PyDataTypeExpr {
    #[staticmethod]
    pub fn from_dtype(datatype: Wrap<DataType>) -> Self {
        DataTypeExpr::Literal(datatype.0).into()
    }

    #[staticmethod]
    pub fn of_expr(expr: PyExpr) -> Self {
        DataTypeExpr::OfExpr(Box::new(expr.inner)).into()
    }

    pub fn collect_dtype(&self, schema: Wrap<Schema>) -> PyResult<Wrap<Schema>> {
        let dtype = self
            .clone()
            .inner
            .into_datatype(&schema.0)
            .map_err(PyPolarsErr::Polars)?;
        Ok(Wrap(Schema::from_iter([(PlSmallStr::EMPTY, dtype)])))
    }
}
