use std::io::{BufReader, BufWriter};
use std::ops::Deref;

use polars::prelude::*;
use polars_io::mmap::ReaderBytes;
use pyo3::prelude::*;

use super::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::exceptions::ComputeError;
use crate::file::{get_file_like, get_mmap_bytes_reader};
use crate::utils::EnterPolarsExt;

#[pymethods]
impl PyDataFrame {
    /// Serialize into binary data.
    fn serialize_binary(slf: Bound<'_, Self>, py_f: PyObject) -> PyResult<()> {
        let file = get_file_like(py_f, true)?;
        let mut writer = BufWriter::new(file);

        let mut slf_1 = slf.try_borrow_mut();
        let slf_1: Result<&mut PyDataFrame, _> = slf_1.as_deref_mut();
        let mut slf_2: Option<PyDataFrame> = (slf_1.is_err()).then(|| (*slf.borrow()).clone());

        let slf: &mut PyDataFrame = slf_1.unwrap_or_else(|_| slf_2.as_mut().unwrap());

        Ok(slf
            .df
            .serialize_into_writer(&mut writer)
            .map_err(PyPolarsErr::from)?)
    }

    /// Deserialize a file-like object containing binary data into a DataFrame.
    #[staticmethod]
    fn deserialize_binary(py: Python<'_>, py_f: PyObject) -> PyResult<Self> {
        let file = get_file_like(py_f, false)?;
        let mut file = BufReader::new(file);

        py.enter_polars_df(|| DataFrame::deserialize_from_reader(&mut file))
    }

    /// Serialize into a JSON string.
    #[cfg(feature = "json")]
    pub fn serialize_json(&mut self, py: Python<'_>, py_f: PyObject) -> PyResult<()> {
        let file = get_file_like(py_f, true)?;
        let writer = BufWriter::new(file);
        py.enter_polars(|| {
            serde_json::to_writer(writer, &self.df)
                .map_err(|err| ComputeError::new_err(err.to_string()))
        })
    }

    /// Deserialize a file-like object containing JSON string data into a DataFrame.
    #[staticmethod]
    #[cfg(feature = "json")]
    pub fn deserialize_json(py: Python<'_>, py_f: Bound<PyAny>) -> PyResult<Self> {
        let mut mmap_bytes_r = get_mmap_bytes_reader(&py_f)?;

        py.enter_polars(move || {
            let mmap_read: ReaderBytes = (&mut mmap_bytes_r).into();
            let bytes = mmap_read.deref();
            let df = serde_json::from_slice::<DataFrame>(bytes)
                .map_err(|err| ComputeError::new_err(err.to_string()))?;
            PyResult::Ok(df.into())
        })
    }
}
