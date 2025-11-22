use polars::prelude::{DataType, DataTypeExpr, PlSmallStr, Schema};
use pyo3::{Bound, IntoPyObject, PyAny, PyResult, Python, pyclass};

use super::PyExpr;
use super::selector::{PySelector, parse_datatype_selector};
use crate::error::PyPolarsErr;
use crate::prelude::Wrap;

#[pyclass(frozen)]
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

    #[staticmethod]
    pub fn self_dtype() -> Self {
        DataTypeExpr::SelfDtype.into()
    }

    pub fn collect_dtype<'py>(
        &self,
        py: Python<'py>,
        schema: Wrap<Schema>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let dtype = self
            .clone()
            .inner
            .into_datatype(&schema.0)
            .map_err(PyPolarsErr::from)?;
        Wrap(dtype).into_pyobject(py)
    }

    pub fn inner_dtype(&self) -> Self {
        self.inner.clone().inner_dtype().into()
    }

    pub fn equals(&self, other: &Self) -> PyExpr {
        self.inner.clone().equals(other.inner.clone()).into()
    }

    pub fn display(&self) -> PyExpr {
        self.inner.clone().display().into()
    }

    pub fn matches(&self, selector: PySelector) -> PyResult<PyExpr> {
        let dtype_selector = parse_datatype_selector(selector)?;
        Ok(self.inner.clone().matches(dtype_selector).into())
    }

    #[staticmethod]
    pub fn struct_with_fields(fields: Vec<(String, PyDataTypeExpr)>) -> Self {
        let fields = fields
            .into_iter()
            .map(|(name, dt_expr)| (PlSmallStr::from_string(name), dt_expr.inner))
            .collect();
        DataTypeExpr::StructWithFields(fields).into()
    }

    pub fn wrap_in_list(&self) -> Self {
        self.inner.clone().wrap_in_list().into()
    }

    pub fn wrap_in_array(&self, width: usize) -> Self {
        self.inner.clone().wrap_in_array(width).into()
    }

    pub fn to_unsigned_integer(&self) -> Self {
        self.inner.clone().int().to_unsigned().into()
    }

    pub fn to_signed_integer(&self) -> Self {
        self.inner.clone().int().to_signed().into()
    }

    pub fn default_value(&self, n: usize, numeric_to_one: bool, num_list_values: usize) -> PyExpr {
        self.inner
            .clone()
            .default_value(n, numeric_to_one, num_list_values)
            .into()
    }

    pub fn list_inner_dtype(&self) -> Self {
        self.inner.clone().list().inner_dtype().into()
    }

    pub fn arr_inner_dtype(&self) -> Self {
        self.inner.clone().arr().inner_dtype().into()
    }

    pub fn arr_width(&self) -> PyExpr {
        self.inner.clone().arr().width().into()
    }

    pub fn arr_shape(&self) -> PyExpr {
        self.inner.clone().arr().shape().into()
    }

    pub fn struct_field_dtype_by_index(&self, index: i64) -> Self {
        self.inner
            .clone()
            .struct_()
            .field_dtype_by_index(index)
            .into()
    }

    pub fn struct_field_dtype_by_name(&self, name: &str) -> Self {
        self.inner
            .clone()
            .struct_()
            .field_dtype_by_name(name)
            .into()
    }

    pub fn struct_field_names(&self) -> PyExpr {
        self.inner.clone().struct_().field_names().into()
    }
}
