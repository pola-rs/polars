use std::io::{BufReader, BufWriter, Cursor};

use polars::lazy::prelude::Expr;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::error::PyPolarsErr;
use crate::exceptions::ComputeError;
use crate::file::get_file_like;
use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        // Used in pickle/pickling
        let mut writer: Vec<u8> = vec![];
        ciborium::ser::into_writer(&self.inner, &mut writer)
            .map_err(|e| PyPolarsErr::Other(format!("{}", e)))?;

        Ok(PyBytes::new_bound(py, &writer).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        // Used in pickle/pickling
        match state.extract::<PyBackedBytes>(py) {
            Ok(s) => {
                let cursor = Cursor::new(&*s);
                self.inner = ciborium::de::from_reader(cursor)
                    .map_err(|e| PyPolarsErr::Other(format!("{}", e)))?;
                Ok(())
            },
            Err(e) => Err(e),
        }
    }

    /// Serialize into binary data.
    fn serialize_binary(&self, py_f: PyObject) -> PyResult<()> {
        let file = get_file_like(py_f, true)?;
        let writer = BufWriter::new(file);
        ciborium::into_writer(&self.inner, writer)
            .map_err(|err| ComputeError::new_err(err.to_string()))
    }

    /// Serialize into a JSON string.
    #[cfg(feature = "json")]
    fn serialize_json(&self, py_f: PyObject) -> PyResult<()> {
        let file = get_file_like(py_f, true)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &self.inner)
            .map_err(|err| ComputeError::new_err(err.to_string()))
    }

    /// Deserialize a file-like object containing binary data into an Expr.
    #[staticmethod]
    fn deserialize_binary(py_f: PyObject) -> PyResult<PyExpr> {
        let file = get_file_like(py_f, false)?;
        let reader = BufReader::new(file);
        let expr = ciborium::from_reader::<Expr, _>(reader)
            .map_err(|err| ComputeError::new_err(err.to_string()))?;
        Ok(expr.into())
    }

    /// Deserialize a file-like object containing JSON string data into an Expr.
    #[staticmethod]
    #[cfg(feature = "json")]
    fn deserialize_json(py_f: PyObject) -> PyResult<PyExpr> {
        // it is faster to first read to memory and then parse: https://github.com/serde-rs/json/issues/160
        // so don't bother with files.
        let mut json = String::new();
        let _ = get_file_like(py_f, false)?
            .read_to_string(&mut json)
            .unwrap();

        // SAFETY:
        // We skipped the serializing/deserializing of the static in lifetime in `DataType`
        // so we actually don't have a lifetime at all when serializing.

        // &str still has a lifetime. But it's ok, because we drop it immediately
        // in this scope.
        let json = unsafe { std::mem::transmute::<&'_ str, &'static str>(json.as_str()) };

        let inner: Expr = serde_json::from_str(json).map_err(|_| {
            let msg = "could not deserialize input into an expression";
            ComputeError::new_err(msg)
        })?;
        Ok(inner.into())
    }
}
