use std::io::BufReader;

#[cfg(any(feature = "ipc", feature = "parquet"))]
use polars::prelude::ArrowSchema;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::file::{EitherRustPythonFile, get_either_file};

#[cfg(feature = "ipc")]
#[pyfunction]
pub fn read_ipc_schema(py: Python<'_>, py_f: PyObject) -> PyResult<Bound<'_, PyDict>> {
    use arrow::io::ipc::read::read_file_metadata;
    let metadata = match get_either_file(py_f, false)? {
        EitherRustPythonFile::Rust(r) => {
            read_file_metadata(&mut BufReader::new(r)).map_err(PyPolarsErr::from)?
        },
        EitherRustPythonFile::Py(mut r) => read_file_metadata(&mut r).map_err(PyPolarsErr::from)?,
    };

    let dict = PyDict::new(py);
    fields_to_pydict(&metadata.schema, &dict)?;
    Ok(dict)
}

#[cfg(feature = "parquet")]
#[pyfunction]
pub fn read_parquet_metadata(py: Python, py_f: PyObject) -> PyResult<Bound<PyDict>> {
    use polars_parquet::read::read_metadata;
    use polars_parquet::read::schema::read_custom_key_value_metadata;

    let metadata = match get_either_file(py_f, false)? {
        EitherRustPythonFile::Rust(r) => {
            read_metadata(&mut BufReader::new(r)).map_err(PyPolarsErr::from)?
        },
        EitherRustPythonFile::Py(mut r) => read_metadata(&mut r).map_err(PyPolarsErr::from)?,
    };

    let key_value_metadata = read_custom_key_value_metadata(metadata.key_value_metadata());
    let dict = PyDict::new(py);
    for (key, value) in key_value_metadata.into_iter() {
        dict.set_item(key.as_str(), value.as_str())?;
    }
    Ok(dict)
}

#[cfg(any(feature = "ipc", feature = "parquet"))]
fn fields_to_pydict(schema: &ArrowSchema, dict: &Bound<'_, PyDict>) -> PyResult<()> {
    for field in schema.iter_values() {
        let dt = Wrap(polars::prelude::DataType::from_arrow_field(field));
        dict.set_item(field.name.as_str(), &dt)?;
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
