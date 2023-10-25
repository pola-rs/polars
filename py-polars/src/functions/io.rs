use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::conversion::Wrap;
use crate::file::{get_either_file, EitherRustPythonFile};
use crate::prelude::DataType;
use crate::PyPolarsErr;

#[cfg(feature = "ipc")]
#[pyfunction]
pub fn read_ipc_schema(py: Python, py_f: PyObject) -> PyResult<PyObject> {
    use polars_core::export::arrow::io::ipc::read::read_file_metadata;
    let metadata = match get_either_file(py_f, false)? {
        EitherRustPythonFile::Rust(mut r) => {
            read_file_metadata(&mut r).map_err(PyPolarsErr::from)?
        },
        EitherRustPythonFile::Py(mut r) => read_file_metadata(&mut r).map_err(PyPolarsErr::from)?,
    };

    let dict = PyDict::new(py);
    for field in metadata.schema.fields {
        let dt: Wrap<DataType> = Wrap((&field.data_type).into());
        dict.set_item(field.name, dt.to_object(py))?;
    }
    Ok(dict.to_object(py))
}

#[cfg(feature = "parquet")]
#[pyfunction]
pub fn read_parquet_schema(py: Python, py_f: PyObject) -> PyResult<PyObject> {
    use polars_parquet::read::{infer_schema, read_metadata};

    let metadata = match get_either_file(py_f, false)? {
        EitherRustPythonFile::Rust(mut r) => read_metadata(&mut r).map_err(PyPolarsErr::from)?,
        EitherRustPythonFile::Py(mut r) => read_metadata(&mut r).map_err(PyPolarsErr::from)?,
    };
    let arrow_schema = infer_schema(&metadata).map_err(PyPolarsErr::from)?;

    let dict = PyDict::new(py);
    for field in arrow_schema.fields {
        let dt: Wrap<DataType> = Wrap((&field.data_type).into());
        dict.set_item(field.name, dt.to_object(py))?;
    }
    Ok(dict.to_object(py))
}
