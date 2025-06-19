use polars::prelude::{DataType, DataTypeExpr, Schema};
use pyo3::{Bound, IntoPyObject, PyAny, PyResult, Python, pyclass};

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

    pub fn to_string(&self) -> PyExpr {
        self.inner.clone().to_string().into()
    }

    pub fn element_bitsize(&self) -> PyExpr {
        self.inner.clone().element_bitsize().into()
    }

    pub fn is_numeric(&self) -> PyExpr {
        self.inner.clone().is_numeric().into()
    }

    pub fn is_integer(&self) -> PyExpr {
        self.inner.clone().is_integer().into()
    }

    pub fn is_float(&self) -> PyExpr {
        self.inner.clone().is_float().into()
    }

    pub fn is_decimal(&self) -> PyExpr {
        self.inner.clone().is_decimal().into()
    }

    pub fn is_categorical(&self) -> PyExpr {
        self.inner.clone().is_categorical().into()
    }

    pub fn is_enum(&self) -> PyExpr {
        self.inner.clone().is_enum().into()
    }

    pub fn is_nested(&self) -> PyExpr {
        self.inner.clone().is_nested().into()
    }

    pub fn is_list(&self) -> PyExpr {
        self.inner.clone().is_list().into()
    }

    pub fn is_array(&self) -> PyExpr {
        self.inner.clone().is_array().into()
    }

    pub fn is_struct(&self) -> PyExpr {
        self.inner.clone().is_struct().into()
    }

    pub fn is_temporal(&self) -> PyExpr {
        self.inner.clone().is_temporal().into()
    }

    pub fn is_datetime(&self) -> PyExpr {
        self.inner.clone().is_datetime().into()
    }

    pub fn is_duration(&self) -> PyExpr {
        self.inner.clone().is_duration().into()
    }

    pub fn is_object(&self) -> PyExpr {
        self.inner.clone().is_object().into()
    }

    pub fn int_to_unsigned(&self) -> Self {
        self.inner.clone().int().to_unsigned().into()
    }

    pub fn int_to_signed(&self) -> Self {
        self.inner.clone().int().to_signed().into()
    }

    pub fn int_is_unsigned(&self) -> PyExpr {
        self.inner.clone().int().is_unsigned().into()
    }

    pub fn int_is_signed(&self) -> PyExpr {
        self.inner.clone().int().is_signed().into()
    }

    pub fn enum_num_categories(&self) -> PyExpr {
        self.inner.clone().enum_().num_categories().into()
    }

    pub fn enum_categories(&self) -> PyExpr {
        self.inner.clone().enum_().categories().into()
    }

    pub fn enum_get_category(&self, index: i64, raise_on_oob: bool) -> PyExpr {
        self.inner
            .clone()
            .enum_()
            .get_category(index, raise_on_oob)
            .into()
    }

    pub fn enum_index_of_category(&self, category: &str, raise_on_missing: bool) -> PyExpr {
        self.inner
            .clone()
            .enum_()
            .index_of_category(category, raise_on_missing)
            .into()
    }

    pub fn list_inner_dtype(&self) -> Self {
        self.inner.clone().list().inner_dtype().into()
    }

    pub fn arr_inner_dtype(&self) -> Self {
        self.inner.clone().arr().inner_dtype().into()
    }

    pub fn arr_has_width(&self, width: usize) -> PyExpr {
        self.inner.clone().arr().has_width(width).into()
    }

    pub fn arr_width(&self) -> PyExpr {
        self.inner.clone().arr().width().into()
    }

    pub fn arr_dimensions(&self) -> PyExpr {
        self.inner.clone().arr().dimensions().into()
    }

    pub fn struct_field_dtype_by_index(&self, index: i64) -> Self {
        self.inner.clone().struct_().field_dtype_by_index(index).into()
    }

    pub fn struct_field_dtype_by_name(&self, name: &str) -> Self {
        self.inner.clone().struct_().field_dtype_by_name(name).into()
    }

    pub fn struct_num_fields(&self) -> PyExpr {
        self.inner.clone().struct_().num_fields().into()
    }

    pub fn struct_field_names(&self) -> PyExpr {
        self.inner.clone().struct_().field_names().into()
    }

    pub fn struct_field_name(&self, index: i64, raise_on_oob: bool) -> PyExpr {
        self.inner
            .clone()
            .struct_()
            .field_name(index, raise_on_oob)
            .into()
    }

    pub fn struct_field_index(&self, field_name: &str, raise_on_missing: bool) -> PyExpr {
        self.inner
            .clone()
            .struct_()
            .field_index(field_name, raise_on_missing)
            .into()
    }
}
