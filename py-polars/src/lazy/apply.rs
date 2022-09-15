use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::lazy::dsl::PyExpr;
use crate::prelude::PyDataType;
use crate::py_modules::POLARS;
use crate::series::PySeries;

trait ToSeries {
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

fn get_output_type(obj: &PyAny) -> Option<DataType> {
    match obj.is_none() {
        true => None,
        false => Some(obj.extract::<PyDataType>().unwrap().into()),
    }
}

pub(crate) fn call_lambda_with_series(
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
pub(crate) fn binary_lambda(lambda: &PyObject, a: Series, b: Series) -> PolarsResult<Series> {
    Python::with_gil(|py| {
        // get the pypolars module
        let pypolars = PyModule::import(py, "polars").unwrap();
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
                Err(e) => panic!("custom python function failed: {}", e.value(py)),
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
                .with_aggregate_pushdown(false)
                .collect()?;

            let s = out.select_at_idx(0).unwrap().clone();
            PySeries::new(s)
        } else {
            return Ok(result_series_wrapper.to_series(py, &pypolars.into_py(py), ""));
        };

        // Finally get the actual Series
        Ok(pyseries.series)
    })
}

pub fn map_single(
    pyexpr: &PyExpr,
    lambda: PyObject,
    output_type: &PyAny,
    agg_list: bool,
) -> PyExpr {
    let output_type = get_output_type(output_type);

    let output_type2 = output_type.clone();
    let function = move |s: Series| {
        Python::with_gil(|py| {
            let output_type = output_type2.clone().unwrap_or(DataType::Unknown);

            // this is a python Series
            let out = call_lambda_with_series(py, s.clone(), &lambda, &POLARS)
                .map_err(|e| PolarsError::ComputeError(format!("{e}").into()))?;
            let s = out.to_series(py, &POLARS, s.name());

            if !matches!(output_type, DataType::Unknown) && s.dtype() != &output_type {
                Err(PolarsError::SchemaMisMatch(
                    format!("Expected output type: '{:?}', but got '{:?}'. Set 'return_dtype' to the proper datatype.", output_type, s.dtype()).into()))
            } else {
                Ok(s)
            }
        })
    };

    let output_map = GetOutput::map_field(move |fld| match output_type {
        Some(ref dt) => Field::new(fld.name(), dt.clone()),
        None => {
            let mut fld = fld.clone();
            fld.coerce(DataType::Unknown);
            fld
        }
    });
    if agg_list {
        pyexpr.clone().inner.map_list(function, output_map).into()
    } else {
        pyexpr.clone().inner.map(function, output_map).into()
    }
}

pub(crate) fn call_lambda_with_series_slice(
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

pub fn map_mul(
    pyexpr: &[PyExpr],
    py: Python,
    lambda: PyObject,
    output_type: &PyAny,
    apply_groups: bool,
) -> PyExpr {
    let output_type = get_output_type(output_type);

    // get the pypolars module
    // do the import outside of the function to prevent import side effects in a hot loop.
    let pypolars = PyModule::import(py, "polars").unwrap().to_object(py);

    let function = move |s: &mut [Series]| {
        Python::with_gil(|py| {
            // this is a python Series
            let out = call_lambda_with_series_slice(py, s, &lambda, &pypolars);

            // we return an error, because that will become a null value polars lazy apply list
            if apply_groups && out.is_none(py) {
                return Err(PolarsError::NoData("".into()));
            }

            Ok(out.to_series(py, &pypolars, ""))
        })
    };

    let exprs = pyexpr.iter().map(|pe| pe.clone().inner).collect::<Vec<_>>();

    let output_map = GetOutput::map_field(move |fld| match output_type {
        Some(ref dt) => Field::new(fld.name(), dt.clone()),
        None => fld.clone(),
    });
    if apply_groups {
        polars::lazy::dsl::apply_multiple(function, exprs, output_map).into()
    } else {
        polars::lazy::dsl::map_multiple(function, exprs, output_map).into()
    }
}
