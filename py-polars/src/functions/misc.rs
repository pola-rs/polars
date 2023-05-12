use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::prelude::DataType;

#[pyfunction]
pub fn dtype_str_repr(dtype: Wrap<DataType>) -> PyResult<String> {
    let dtype = dtype.0;
    Ok(dtype.to_string())
}
