use std::io::{BufReader, BufWriter};

use polars::lazy::prelude::Expr;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::PyExpr;
use crate::error::PyPolarsErr;
use crate::exceptions::ComputeError;
use crate::file::get_file_like;

#[pymethods]
impl PyExpr {
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mut bytes: Vec<u8> = vec![];
        self.inner
            .serialize_compact_into(&mut bytes)
            .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn __setstate__(&mut self, state: &Bound<PyAny>) -> PyResult<()> {
        let bytes = state.extract::<PyBackedBytes>()?;
        self.inner = Expr::deserialize_compact_from(&mut &*bytes)
            .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
        Ok(())
    }

    /// Serialize into binary data.
    fn serialize_binary(&self, py_f: Py<PyAny>) -> PyResult<()> {
        let file = get_file_like(py_f, true)?;
        self.inner
            .serialize_binary_into(&mut BufWriter::new(file))
            .map_err(|err| ComputeError::new_err(err.to_string()))
    }

    /// Serialize into a JSON string.
    #[cfg(feature = "json")]
    fn serialize_json(&self, py_f: Py<PyAny>) -> PyResult<()> {
        let file = get_file_like(py_f, true)?;
        self.inner
            .serialize_json_into(&mut BufWriter::new(file))
            .map_err(|err| ComputeError::new_err(err.to_string()))
    }

    /// Deserialize a file-like object containing binary data into an Expr.
    #[staticmethod]
    fn deserialize_binary(py_f: Py<PyAny>) -> PyResult<PyExpr> {
        let file = get_file_like(py_f, false)?;
        let expr = Expr::deserialize_binary_from(&mut BufReader::new(file))
            .map_err(|err| ComputeError::new_err(err.to_string()))?;
        Ok(expr.into())
    }

    /// Deserialize a file-like object containing JSON string data into an Expr.
    #[staticmethod]
    #[cfg(feature = "json")]
    fn deserialize_json(py_f: Py<PyAny>) -> PyResult<PyExpr> {
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

        let inner = Expr::deserialize_json_from_str(json).map_err(|_| {
            let msg = "could not deserialize input into an expression";
            ComputeError::new_err(msg)
        })?;
        Ok(inner.into())
    }
}
