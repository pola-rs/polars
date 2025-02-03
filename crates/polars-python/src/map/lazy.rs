use polars::prelude::*;
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::py_modules::polars;
use crate::series::PySeries;
use crate::{PyExpr, Wrap};

pub(crate) trait ToSeries {
    fn to_series(
        &self,
        py: Python,
        py_polars_module: &Py<PyModule>,
        name: &str,
    ) -> PolarsResult<Series>;
}

impl ToSeries for PyObject {
    fn to_series(
        &self,
        py: Python,
        py_polars_module: &Py<PyModule>,
        name: &str,
    ) -> PolarsResult<Series> {
        let py_pyseries = match self.getattr(py, "_s") {
            Ok(s) => s,
            // the lambda did not return a series, we try to create a new python Series
            _ => {
                let res = py_polars_module
                    .getattr(py, "Series")
                    .unwrap()
                    .call1(py, (name, PyList::new(py, [self]).unwrap()));

                match res {
                    Ok(python_s) => python_s.getattr(py, "_s").unwrap(),
                    Err(_) => {
                        polars_bail!(ComputeError:
                            "expected a something that could convert to a `Series` but got: {}",
                            self.bind(py).get_type()
                        )
                    },
                }
            },
        };
        let s = match py_pyseries.extract::<PySeries>(py) {
            Ok(pyseries) => pyseries.series,
            // This happens if the executed Polars is not from this source.
            // Currently only happens in PC-workers
            // For now use arrow to convert
            // Eventually we must use Polars' Series Export as that can deal with
            // multiple chunks
            Err(_) => {
                use arrow::ffi;
                let kwargs = PyDict::new(py);
                kwargs.set_item("in_place", true).unwrap();
                py_pyseries
                    .call_method(py, "rechunk", (), Some(&kwargs))
                    .map_err(|e| polars_err!(ComputeError: "could not rechunk: {e}"))?;

                // Prepare a pointer to receive the Array struct.
                let array = Box::new(ffi::ArrowArray::empty());
                let schema = Box::new(ffi::ArrowSchema::empty());

                let array_ptr = &*array as *const ffi::ArrowArray;
                let schema_ptr = &*schema as *const ffi::ArrowSchema;
                // SAFETY:
                // this is unsafe as it write to the pointers we just prepared
                py_pyseries
                    .call_method1(
                        py,
                        "_export_arrow_to_c",
                        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
                    )
                    .map_err(|e| polars_err!(ComputeError: "{e}"))?;

                unsafe {
                    let field = ffi::import_field_from_c(schema.as_ref())?;
                    let array = ffi::import_array_from_c(*array, field.dtype)?;
                    Series::from_arrow(field.name, array)?
                }
            },
        };
        Ok(s)
    }
}

pub(crate) fn call_lambda_with_series(
    py: Python,
    s: Series,
    lambda: &PyObject,
) -> PyResult<PyObject> {
    let pypolars = polars(py).bind(py);

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
pub(crate) fn binary_lambda(
    lambda: &PyObject,
    a: Series,
    b: Series,
) -> PolarsResult<Option<Series>> {
    Python::with_gil(|py| {
        // get the pypolars module
        let pypolars = polars(py).bind(py);
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
                Err(e) => polars_bail!(
                    ComputeError: "custom python function failed: {}", e.value(py),
                ),
            };
        let pyseries = if let Ok(expr) = result_series_wrapper.getattr(py, "_pyexpr") {
            let pyexpr = expr.extract::<PyExpr>(py).unwrap();
            let expr = pyexpr.inner;
            let df = DataFrame::empty();
            let out = df
                .lazy()
                .select([expr])
                .with_predicate_pushdown(false)
                .with_projection_pushdown(false)
                .collect()?;

            let s = out.select_at_idx(0).unwrap().clone();
            PySeries::new(s.take_materialized_series())
        } else {
            return Some(result_series_wrapper.to_series(py, pypolars.as_unbound(), ""))
                .transpose();
        };

        // Finally get the actual Series
        Ok(Some(pyseries.series))
    })
}

pub fn map_single(
    pyexpr: &PyExpr,
    lambda: PyObject,
    output_type: Option<Wrap<DataType>>,
    agg_list: bool,
    is_elementwise: bool,
    returns_scalar: bool,
) -> PyExpr {
    let output_type = output_type.map(|wrap| wrap.0);

    let func =
        python_udf::PythonUdfExpression::new(lambda, output_type, is_elementwise, returns_scalar);
    pyexpr.inner.clone().map_python(func, agg_list).into()
}

pub(crate) fn call_lambda_with_columns_slice(
    py: Python,
    s: &[Column],
    lambda: &PyObject,
    pypolars: &Py<PyModule>,
) -> PyObject {
    let pypolars = pypolars.bind(py);

    // create a PySeries struct/object for Python
    let iter = s.iter().map(|s| {
        let ps = PySeries::new(s.as_materialized_series().clone());

        // Wrap this PySeries object in the python side Series wrapper
        let python_series_wrapper = pypolars.getattr("wrap_s").unwrap().call1((ps,)).unwrap();

        python_series_wrapper
    });
    let wrapped_s = PyList::new(py, iter).unwrap();

    // call the lambda and get a python side Series wrapper
    match lambda.call1(py, (wrapped_s,)) {
        Ok(pyobj) => pyobj,
        Err(e) => panic!("python function failed: {}", e.value(py)),
    }
}

pub fn map_mul(
    pyexpr: &[PyExpr],
    py: Python,
    lambda: PyObject,
    output_type: Option<Wrap<DataType>>,
    map_groups: bool,
    returns_scalar: bool,
) -> PyExpr {
    // get the pypolars module
    // do the import outside of the function to prevent import side effects in a hot loop.
    let pypolars = polars(py).clone_ref(py);

    let function = move |s: &mut [Column]| {
        Python::with_gil(|py| {
            // this is a python Series
            let out = call_lambda_with_columns_slice(py, s, &lambda, &pypolars);

            // we return an error, because that will become a null value polars lazy apply list
            if map_groups && out.is_none(py) {
                return Ok(None);
            }

            Ok(Some(out.to_series(py, &pypolars, "")?.into_column()))
        })
    };

    let exprs = pyexpr.iter().map(|pe| pe.clone().inner).collect::<Vec<_>>();

    let output_map = GetOutput::map_field(move |fld| {
        Ok(match output_type {
            Some(ref dt) => Field::new(fld.name().clone(), dt.0.clone()),
            None => fld.clone(),
        })
    });
    if map_groups {
        polars::lazy::dsl::apply_multiple(function, exprs, output_map, returns_scalar).into()
    } else {
        polars::lazy::dsl::map_multiple(function, exprs, output_map).into()
    }
}
