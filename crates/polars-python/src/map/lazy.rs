use std::mem::{ManuallyDrop, MaybeUninit};

use polars::prelude::*;
use polars_ffi::version_0::SeriesExport;
use pyo3::conversion::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::type_object::PyTypeInfo;
use pyo3::types::PyDict;

use crate::expr::datatype::PyDataTypeExpr;
use crate::py_modules::pyseries;
use crate::series::PySeries;
use crate::{PyExpr, Wrap};

pub(crate) fn call_lambda_with_series(
    py: Python<'_>,
    s: &[Column],
    output_dtype: Option<DataType>,
    lambda: &PyObject,
) -> PolarsResult<Option<Column>> {
    // Set return_dtype in kwargs
    let dict = PyDict::new(py);
    let output_dtype = match output_dtype {
        None => None,
        Some(dt) => Some(Wrap(dt).into_pyobject(py)?),
    };
    dict.set_item("return_dtype", output_dtype)?;

    let python_pyseries_type = pyseries(py);
    let rust_type = PySeries::type_object(py);

    // If the Python library and the Rust library are not compiled within a single compilation unit
    // we need to go via the FFI as the types are not compatible.
    let needs_ffi_translation = !rust_type.is(python_pyseries_type);

    let series_objects = if needs_ffi_translation {
        s.iter()
            .map(|c| {
                let mut export = ManuallyDrop::new(polars_ffi::version_0::export_series(
                    c.as_materialized_series(),
                ));
                let s_location = &raw mut export;
                pyseries(py)
                    .getattr(py, "_import")?
                    .call1(py, (s_location as usize,))
            })
            .collect::<PyResult<Vec<_>>>()?
    } else {
        s.iter()
            .map(|c| PySeries::new(c.as_materialized_series().clone()).into_py_any(py))
            .collect::<PyResult<Vec<PyObject>>>()?
    };

    let result = lambda.call(py, (series_objects,), Some(&dict))?;
    if result.is_none(py) {
        Ok(None)
    } else {
        if needs_ffi_translation {
            let mut export: MaybeUninit<SeriesExport> = MaybeUninit::uninit();
            pyseries(py)
                .call_method1(py, "_export", (&raw mut export as usize,))
                .unwrap();
            unsafe {
                let export = export.assume_init();
                polars_ffi::version_0::import_series(export).map(|s| Some(s.into_column()))
            }
        } else {
            Ok(result
                .extract::<PySeries>(py)
                .map(|s| s.series.into_column())
                .map(Some)?)
        }
    }
}

pub fn map_expr(
    pyexpr: &[PyExpr],
    lambda: PyObject,
    output_type: Option<PyDataTypeExpr>,
    is_elementwise: bool,
    returns_scalar: bool,
    map_groups: bool,
    is_ufunc: bool,
) -> PyExpr {
    let output_type = if is_ufunc {
        debug_assert!(output_type.is_none());
        Some(DataTypeExpr::Literal(DataType::Unknown(UnknownKind::Ufunc)))
    } else {
        output_type.map(|v| v.inner)
    };
    let func = python_dsl::PythonUdfExpression::new(
        lambda,
        output_type,
        is_elementwise,
        returns_scalar,
        map_groups,
    );
    let exprs = pyexpr.iter().map(|pe| pe.clone().inner).collect::<Vec<_>>();
    Expr::map_many_python(exprs, func).into()
}
