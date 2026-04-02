use polars_testing::asserts::{SeriesEqualOptions, assert_series_equal};
use pyo3::prelude::*;

use crate::PySeries;
use crate::error::PyPolarsErr;

#[pyfunction]
#[pyo3(signature = (left, right, *, check_dtypes, check_names, check_order, check_exact, rel_tol, abs_tol, categorical_as_str))]
pub fn assert_series_equal_py(
    left: &PySeries,
    right: &PySeries,
    check_dtypes: bool,
    check_names: bool,
    check_order: bool,
    check_exact: bool,
    rel_tol: f64,
    abs_tol: f64,
    categorical_as_str: bool,
) -> PyResult<()> {
    let left_series = &left.series.read();
    let right_series = &right.series.read();

    let options = SeriesEqualOptions {
        check_dtypes,
        check_names,
        check_order,
        check_exact,
        rel_tol,
        abs_tol,
        categorical_as_str,
    };

    assert_series_equal(left_series, right_series, options).map_err(|e| PyPolarsErr::from(e).into())
}
