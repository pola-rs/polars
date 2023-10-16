use polars_core;
use polars_core::fmt::FloatFmt;
use polars_core::prelude::IDX_DTYPE;
use polars_core::POOL;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::conversion::Wrap;

const VERSION: &str = env!("CARGO_PKG_VERSION");
#[pyfunction]
pub fn get_polars_version() -> &'static str {
    VERSION
}

#[pyfunction]
pub fn get_index_type(py: Python) -> PyObject {
    Wrap(IDX_DTYPE).to_object(py)
}

#[pyfunction]
pub fn threadpool_size() -> usize {
    POOL.current_num_threads()
}

#[pyfunction]
pub fn set_float_fmt(fmt: &str) -> PyResult<()> {
    let fmt = match fmt {
        "full" => FloatFmt::Full,
        "mixed" => FloatFmt::Mixed,
        e => {
            return Err(PyValueError::new_err(format!(
                "fmt must be one of {{'full', 'mixed'}}, got {e}",
            )))
        },
    };
    polars_core::fmt::set_float_fmt(fmt);
    Ok(())
}

#[pyfunction]
pub fn get_float_fmt() -> PyResult<String> {
    let strfmt = match polars_core::fmt::get_float_fmt() {
        FloatFmt::Full => "full",
        FloatFmt::Mixed => "mixed",
    };
    Ok(strfmt.to_string())
}

#[pyfunction]
pub fn set_float_precision(precision: Option<usize>) -> PyResult<()> {
    use polars_core::fmt::set_float_precision;
    set_float_precision(precision);
    Ok(())
}

#[pyfunction]
pub fn get_float_precision() -> PyResult<Option<usize>> {
    use polars_core::fmt::get_float_precision;
    Ok(get_float_precision())
}
