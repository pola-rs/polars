use std::io::BufReader;

#[cfg(any(feature = "ipc", feature = "parquet"))]
use polars::prelude::ArrowSchema;
use polars_core::datatypes::create_enum_dtype;
use polars_core::export::arrow::array::Utf8ViewArray;
use polars_core::prelude::{DTYPE_ENUM_KEY, DTYPE_ENUM_VALUE};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::file::{get_either_file, EitherRustPythonFile};
use crate::prelude::ArrowDataType;

#[cfg(feature = "ipc")]
#[pyfunction]
pub fn read_ipc_schema(py: Python, py_f: PyObject) -> PyResult<PyObject> {
    use polars_core::export::arrow::io::ipc::read::read_file_metadata;
    let metadata = match get_either_file(py_f, false)? {
        EitherRustPythonFile::Rust(r) => {
            read_file_metadata(&mut BufReader::new(r)).map_err(PyPolarsErr::from)?
        },
        EitherRustPythonFile::Py(mut r) => read_file_metadata(&mut r).map_err(PyPolarsErr::from)?,
    };

    let dict = PyDict::new_bound(py);
    fields_to_pydict(&metadata.schema, &dict, py)?;
    Ok(dict.to_object(py))
}

#[cfg(feature = "parquet")]
#[pyfunction]
pub fn read_parquet_schema(py: Python, py_f: PyObject) -> PyResult<PyObject> {
    use polars_parquet::read::{infer_schema, read_metadata};

    let metadata = match get_either_file(py_f, false)? {
        EitherRustPythonFile::Rust(r) => {
            read_metadata(&mut BufReader::new(r)).map_err(PyPolarsErr::from)?
        },
        EitherRustPythonFile::Py(mut r) => read_metadata(&mut r).map_err(PyPolarsErr::from)?,
    };
    let arrow_schema = infer_schema(&metadata).map_err(PyPolarsErr::from)?;

    let dict = PyDict::new_bound(py);
    fields_to_pydict(&arrow_schema, &dict, py)?;
    Ok(dict.to_object(py))
}

#[cfg(any(feature = "ipc", feature = "parquet"))]
fn fields_to_pydict(schema: &ArrowSchema, dict: &Bound<'_, PyDict>, py: Python) -> PyResult<()> {
    for field in schema.iter_values() {
        let dt = if field.metadata.get(DTYPE_ENUM_KEY) == Some(&DTYPE_ENUM_VALUE.into()) {
            Wrap(create_enum_dtype(Utf8ViewArray::new_empty(
                ArrowDataType::LargeUtf8,
            )))
        } else {
            Wrap((&field.dtype).into())
        };
        dict.set_item(field.name.as_str(), dt.to_object(py))?;
    }
    Ok(())
}

#[cfg(feature = "clipboard")]
#[pyfunction]
pub fn read_clipboard_string() -> PyResult<String> {
    use arboard;
    let mut clipboard =
        arboard::Clipboard::new().map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
    let result = clipboard
        .get_text()
        .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
    Ok(result)
}

#[cfg(feature = "clipboard")]
#[pyfunction]
pub fn write_clipboard_string(s: &str) -> PyResult<()> {
    use arboard;
    let mut clipboard =
        arboard::Clipboard::new().map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
    clipboard
        .set_text(s)
        .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
    Ok(())
}
