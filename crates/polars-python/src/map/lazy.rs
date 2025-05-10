use std::mem::{ManuallyDrop, MaybeUninit};

use polars::prelude::*;
use polars_ffi::version_0::SeriesExport;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::py_modules::{pl_series, polars, polars_rs};
use crate::series::PySeries;
use crate::{PyExpr, Wrap};

pub(crate) trait ToSeries {
    fn to_series(
        &self,
        py: Python<'_>,
        py_polars_module: &Py<PyModule>,
        name: &str,
    ) -> PolarsResult<Series>;
}

impl ToSeries for PyObject {
    fn to_series(
        &self,
        py: Python<'_>,
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
            Err(_) => {
                let mut export: MaybeUninit<SeriesExport> = MaybeUninit::uninit();
                py_pyseries
                    .call_method1(py, "_export", (&raw mut export as usize,))
                    .unwrap();
                unsafe {
                    let export = export.assume_init();
                    polars_ffi::version_0::import_series(export)?
                }
            },
        };
        Ok(s)
    }
}

pub(crate) fn call_lambda_with_series(
    py: Python<'_>,
    s: &Series,
    lambda: &PyObject,
) -> PyResult<PyObject> {
    let pypolars = polars(py).bind(py);

    // create a PySeries struct/object for Python
    let pyseries = PySeries::new(s.clone());
    // Wrap this PySeries object in the python side Series wrapper
    let mut python_series_wrapper = pypolars
        .getattr("wrap_s")
        .unwrap()
        .call1((pyseries,))
        .unwrap();

    if !python_series_wrapper
        .getattr("_s")
        .unwrap()
        .is_instance(polars_rs(py).getattr(py, "PySeries").unwrap().bind(py))
        .unwrap()
    {
        let mut export = ManuallyDrop::new(polars_ffi::version_0::export_series(s));
        let plseries = pl_series(py).bind(py);

        let s_location = &raw mut export;
        python_series_wrapper = plseries
            .getattr("_import")
            .unwrap()
            .call1((s_location as usize,))
            .unwrap()
    }
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
        python_dsl::PythonUdfExpression::new(lambda, output_type, is_elementwise, returns_scalar);
    pyexpr.inner.clone().map_python(func, agg_list).into()
}

pub(crate) fn call_lambda_with_columns_slice(
    py: Python<'_>,
    s: &[Column],
    lambda: &PyObject,
    pypolars: &Py<PyModule>,
) -> PyObject {
    let pypolars = pypolars.bind(py);

    // create a PySeries struct/object for Python
    let iter = s.iter().map(|s| {
        let ps = PySeries::new(s.as_materialized_series().clone());

        // Wrap this PySeries object in the python side Series wrapper
        pypolars.getattr("wrap_s").unwrap().call1((ps,)).unwrap()
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
    py: Python<'_>,
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
