use polars::prelude::*;
use pyo3::conversion::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::expr::datatype::PyDataTypeExpr;
use crate::series::PySeries;
use crate::{PyExpr, Wrap};

pub(crate) fn call_lambda_with_series(
    py: Python<'_>,
    s: &[Column],
    output_dtype: Option<DataType>,
    lambda: &PyObject,
) -> PolarsResult<Column> {
    // Set return_dtype in kwargs
    let dict = PyDict::new(py);
    let output_dtype = match output_dtype {
        None => None,
        Some(dt) => Some(Wrap(dt).into_pyobject(py)?),
    };
    dict.set_item("return_dtype", output_dtype)?;

    let series_objects = s
        .iter()
        .map(|c| PySeries::new(c.as_materialized_series().clone()).into_py_any(py))
        .collect::<PyResult<Vec<PyObject>>>()?;

    let result = lambda.call(py, (series_objects,), Some(&dict))?;
    Ok(result
        .extract::<PySeries>(py)
        .map(|s| s.series.into_column())?)
}

pub fn map_expr(
    pyexpr: &[PyExpr],
    lambda: PyObject,
    output_type: Option<PyDataTypeExpr>,
    is_elementwise: bool,
    returns_scalar: bool,
    is_ufunc: bool,
) -> PyExpr {
    let output_type = if is_ufunc {
        debug_assert!(output_type.is_none());
        Some(DataTypeExpr::Literal(DataType::Unknown(UnknownKind::Ufunc)))
    } else {
        output_type.map(|v| v.inner)
    };
    let func =
        python_dsl::PythonUdfExpression::new(lambda, output_type, is_elementwise, returns_scalar);
    let exprs = pyexpr.iter().map(|pe| pe.clone().inner).collect::<Vec<_>>();
    Expr::map_many_python(exprs, func).into()
}
