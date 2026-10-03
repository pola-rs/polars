use polars_testing::asserts::{SeriesEqualOptions, assert_series_equal};
use pyo3::prelude::*;

use crate::PySeries;
use crate::utils::EnterPolarsExt;

#[pyfunction]
#[pyo3(signature = (left, right, *, check_dtypes, check_names, check_order, check_exact, rel_tol, abs_tol, categorical_as_str))]
pub fn assert_series_equal_py(
    py: Python<'_>,
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
    let options = SeriesEqualOptions {
        check_dtypes,
        check_names,
        check_order,
        check_exact,
        rel_tol,
        abs_tol,
        categorical_as_str,
    };

    py.enter_polars(|| assert_series_equal(&left.series.read(), &right.series.read(), options))
}
