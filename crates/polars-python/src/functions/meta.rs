use polars_core::fmt::FloatFmt;
use polars_core::prelude::IDX_DTYPE;
use polars_core::POOL;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::conversion::Wrap;

#[pyfunction]
pub fn get_index_type(py: Python) -> PyObject {
    Wrap(IDX_DTYPE).to_object(py)
}

#[pyfunction]
pub fn thread_pool_size() -> usize {
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

#[pyfunction]
pub fn set_thousands_separator(sep: Option<char>) -> PyResult<()> {
    use polars_core::fmt::set_thousands_separator;
    set_thousands_separator(sep);
    Ok(())
}

#[pyfunction]
pub fn get_thousands_separator() -> PyResult<Option<String>> {
    use polars_core::fmt::get_thousands_separator;
    Ok(Some(get_thousands_separator()))
}

#[pyfunction]
pub fn set_decimal_separator(sep: Option<char>) -> PyResult<()> {
    use polars_core::fmt::set_decimal_separator;
    set_decimal_separator(sep);
    Ok(())
}

#[pyfunction]
pub fn get_decimal_separator() -> PyResult<Option<char>> {
    use polars_core::fmt::get_decimal_separator;
    Ok(Some(get_decimal_separator()))
}

#[pyfunction]
pub fn set_trim_decimal_zeros(trim: Option<bool>) -> PyResult<()> {
    use polars_core::fmt::set_trim_decimal_zeros;
    set_trim_decimal_zeros(trim);
    Ok(())
}

#[pyfunction]
pub fn get_trim_decimal_zeros() -> PyResult<Option<bool>> {
    use polars_core::fmt::get_trim_decimal_zeros;
    Ok(Some(get_trim_decimal_zeros()))
}
