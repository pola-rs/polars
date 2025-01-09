use std::io::{BufReader, BufWriter};
use std::ops::Deref;

use polars::prelude::*;
use polars_io::mmap::ReaderBytes;
use pyo3::prelude::*;

use super::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::exceptions::ComputeError;
use crate::file::{get_file_like, get_mmap_bytes_reader};

#[pymethods]
impl PyDataFrame {
    /// Serialize into binary data.
    fn serialize_binary(&mut self, py: Python, py_f: PyObject) -> PyResult<()> {
        let file = get_file_like(py_f, true)?;
        let mut writer = BufWriter::new(file);

        py.allow_threads(|| {
            self.df
                .serialize_into_writer(&mut writer)
                .map_err(|e| PyPolarsErr::from(e).into())
        })
    }

    /// Deserialize a file-like object containing binary data into a DataFrame.
    #[staticmethod]
    fn deserialize_binary(py: Python, py_f: PyObject) -> PyResult<Self> {
        let file = get_file_like(py_f, false)?;
        let mut file = BufReader::new(file);

        py.allow_threads(|| {
            DataFrame::deserialize_from_reader(&mut file)
                .map_err(|e| PyPolarsErr::from(e).into())
                .map(|x| x.into())
        })
    }

    /// Serialize into a JSON string.
    #[cfg(feature = "json")]
    pub fn serialize_json(&mut self, py: Python, py_f: PyObject) -> PyResult<()> {
        let file = get_file_like(py_f, true)?;
        let writer = BufWriter::new(file);
        py.allow_threads(|| {
            serde_json::to_writer(writer, &self.df)
                .map_err(|err| ComputeError::new_err(err.to_string()))
        })
    }

    /// Deserialize a file-like object containing JSON string data into a DataFrame.
    #[staticmethod]
    #[cfg(feature = "json")]
    pub fn deserialize_json(py: Python, py_f: Bound<PyAny>) -> PyResult<Self> {
        let mut mmap_bytes_r = get_mmap_bytes_reader(&py_f)?;

        py.allow_threads(move || {
            let mmap_read: ReaderBytes = (&mut mmap_bytes_r).into();
            let bytes = mmap_read.deref();
            let df = serde_json::from_slice::<DataFrame>(bytes)
                .map_err(|err| ComputeError::new_err(err.to_string()))?;
            Ok(df.into())
        })
    }
}
