use polars_testing::asserts::{DataFrameEqualOptions, assert_dataframe_equal, assert_schema_equal};
use pyo3::prelude::*;

use crate::error::PyPolarsErr;
use crate::{PyDataFrame, PySchema};

#[pyfunction]
#[pyo3(signature = (left, right, *, check_row_order, check_column_order, check_dtypes, check_exact, rel_tol, abs_tol, categorical_as_str))]
pub fn assert_dataframe_equal_py(
    left: &PyDataFrame,
    right: &PyDataFrame,
    check_row_order: bool,
    check_column_order: bool,
    check_dtypes: bool,
    check_exact: bool,
    rel_tol: f64,
    abs_tol: f64,
    categorical_as_str: bool,
) -> PyResult<()> {
    let left_df = &left.df.read();
    let right_df = &right.df.read();

    let options = DataFrameEqualOptions {
        check_row_order,
        check_column_order,
        check_dtypes,
        check_exact,
        rel_tol,
        abs_tol,
        categorical_as_str,
    };

    assert_dataframe_equal(left_df, right_df, options).map_err(|e| PyPolarsErr::from(e).into())
}

#[pyfunction]
#[pyo3(signature = (left_schema, right_schema, check_dtypes, check_column_order))]
pub fn assert_schema_equal_py(
    left_schema: PySchema,
    right_schema: PySchema,
    check_dtypes: bool,
    check_column_order: bool,
) -> PyResult<()> {
    assert_schema_equal(
        &left_schema.0,
        &right_schema.0,
        check_dtypes,
        check_column_order,
    )
    .map_err(|e| PyPolarsErr::from(e).into())
}
