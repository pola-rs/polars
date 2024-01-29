use polars_core::datatypes::create_enum_data_type;
use polars_core::export::arrow::array::Utf8ViewArray;
use polars_core::export::arrow::datatypes::Field;
use polars_core::prelude::{DTYPE_ENUM_KEY, DTYPE_ENUM_VALUE};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::conversion::Wrap;
use crate::file::{get_either_file, EitherRustPythonFile};
use crate::prelude::ArrowDataType;
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
    fields_to_pydict(&metadata.schema.fields, dict, py)?;
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
    fields_to_pydict(&arrow_schema.fields, dict, py)?;
    Ok(dict.to_object(py))
}

#[cfg(any(feature = "ipc", feature = "parquet"))]
fn fields_to_pydict(fields: &Vec<Field>, dict: &PyDict, py: Python) -> PyResult<()> {
    for field in fields {
        let dt = if field.metadata.get(DTYPE_ENUM_KEY) == Some(&DTYPE_ENUM_VALUE.into()) {
            Wrap(create_enum_data_type(Utf8ViewArray::new_empty(
                ArrowDataType::LargeUtf8,
            )))
        } else {
            Wrap((&field.data_type).into())
        };
        dict.set_item(&field.name, dt.to_object(py))?;
    }
    Ok(())
}
