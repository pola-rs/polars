use polars::lazy::dsl;
use pyo3::prelude::*;

use crate::prelude::*;
use crate::PyExpr;

#[pyfunction]
pub fn arange(start: PyExpr, end: PyExpr, step: i64) -> PyExpr {
    dsl::arange(start.inner, end.inner, step).into()
}

#[pyfunction]
pub fn int_range(start: PyExpr, end: PyExpr, step: i64, dtype: Wrap<DataType>) -> PyExpr {
    let dtype = dtype.0;

    let mut result = dsl::int_range(start.inner, end.inner, step);

    if dtype != DataType::Int64 {
        result = result.cast(dtype)
    }

    result.into()
}

#[pyfunction]
pub fn int_ranges(start: PyExpr, end: PyExpr, step: i64, dtype: Wrap<DataType>) -> PyExpr {
    let dtype = dtype.0;

    let mut result = dsl::int_ranges(start.inner, end.inner, step);

    if dtype != DataType::Int64 {
        result = result.cast(DataType::List(Box::new(dtype)))
    }

    result.into()
}
