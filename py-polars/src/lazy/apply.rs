use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::lazy::dsl::PyExpr;
use crate::series::PySeries;

pub trait ToSeries {
    fn to_series(&self, py: Python, py_polars_module: &PyObject, name: &str) -> Series;
}

impl ToSeries for PyObject {
    fn to_series(&self, py: Python, py_polars_module: &PyObject, name: &str) -> Series {
        let py_pyseries = match self.getattr(py, "_s") {
            Ok(s) => s,
            // the lambda did not return a series, we try to create a new python Series
            _ => {
                let res = py_polars_module
                    .getattr(py, "Series")
                    .unwrap()
                    .call1(py, (name, PyList::new(py, [self])));

                match res {
                    Ok(python_s) => python_s.getattr(py, "_s").unwrap(),
                    Err(_) => {
                        panic!(
                            "expected a something that could convert to a `Series` but got: {}",
                            self.as_ref(py).get_type()
                        )
                    }
                }
            }
        };
        let pyseries = py_pyseries.extract::<PySeries>(py).unwrap();
        // Finally get the actual Series
        pyseries.series
    }
}

pub(crate) fn call_lambda_series_unary(
    py: Python,
    s: Series,
    lambda: &PyObject,
    polars_module: &PyObject,
) -> PyResult<PyObject> {
    let pypolars = polars_module.cast_as::<PyModule>(py).unwrap();

    // create a PySeries struct/object for Python
    let pyseries = PySeries::new(s);
    // Wrap this PySeries object in the python side Series wrapper
    let python_series_wrapper = pypolars
        .getattr("wrap_s")
        .unwrap()
        .call1((pyseries,))
        .unwrap();
    // call the lambda and get a python side Series wrapper
    lambda.call1(py, (python_series_wrapper,))
}

/// A python lambda taking two Series
pub(crate) fn call_lambda_series_binary(
    py: Python,
    a: Series,
    b: Series,
    lambda: &PyObject,
    polars_module: &PyObject,
) -> PolarsResult<PyObject> {
    let pypolars = polars_module.cast_as::<PyModule>(py).unwrap();
    // get the pypolars module
    // create a PySeries struct/object for Python
    let pyseries_a = PySeries::new(a);
    let pyseries_b = PySeries::new(b);

    // Wrap this PySeries object in the python side Series wrapper
    let python_series_wrapper_a = pypolars
        .getattr("wrap_s")
        .unwrap()
        .call1((pyseries_a,))
        .unwrap();
    let python_series_wrapper_b = pypolars
        .getattr("wrap_s")
        .unwrap()
        .call1((pyseries_b,))
        .unwrap();

    // call the lambda and get a python side Series wrapper
    let result_series_wrapper =
        match lambda.call1(py, (python_series_wrapper_a, python_series_wrapper_b)) {
            Ok(pyobj) => pyobj,
            Err(e) => {
                return Err(PolarsError::ComputeError(
                    format!("custom python function failed: {}", e.value(py)).into(),
                ))
            }
        };
    let pyseries = if let Ok(expr) = result_series_wrapper.getattr(py, "_pyexpr") {
        let pyexpr = expr.extract::<PyExpr>(py).unwrap();
        let expr = pyexpr.inner;
        let df = DataFrame::new_no_checks(vec![]);
        let out = df
            .lazy()
            .select([expr])
            .with_predicate_pushdown(false)
            .with_projection_pushdown(false)
            .collect()?;

        let s = out.select_at_idx(0).unwrap().clone();
        PySeries::new(s)
    } else {
        return Ok(result_series_wrapper);
    };

    // Finally get the actual Series
    Ok(pyseries.into_py(py))
}

pub(crate) fn call_lambda_series_slice(
    py: Python,
    s: &mut [Series],
    lambda: &PyObject,
    polars_module: &PyObject,
) -> PyObject {
    let pypolars = polars_module.cast_as::<PyModule>(py).unwrap();

    // create a PySeries struct/object for Python
    let iter = s.iter().map(|s| {
        let ps = PySeries::new(s.clone());

        // Wrap this PySeries object in the python side Series wrapper
        let python_series_wrapper = pypolars.getattr("wrap_s").unwrap().call1((ps,)).unwrap();

        python_series_wrapper
    });
    let wrapped_s = PyList::new(py, iter);

    // call the lambda and get a python side Series wrapper
    match lambda.call1(py, (wrapped_s,)) {
        Ok(pyobj) => pyobj,
        Err(e) => panic!("python apply failed: {}", e.value(py)),
    }
}
