use crossbeam_channel::Receiver;
use polars::frame::DataFrame;
use pyo3::prelude::*;

use crate::dataframe::PyDataFrame;

#[pyclass]
pub struct DataFrameIter {
    pub df_receiver: Receiver<DataFrame>,
    pub limit: Option<usize>,
    pub num_rows: usize,
}

#[pymethods]
impl DataFrameIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyDataFrame> {
        if slf.limit.is_some() && slf.num_rows >= slf.limit.unwrap() {
            return None;
        }
        let mut df = match slf.df_receiver.recv() {
            Ok(df) => df,
            Err(e) => {
                return None;
            }
        };

        if slf.limit.is_some() && slf.limit.unwrap() - slf.num_rows < df.height() {
            let limit = slf.limit.unwrap() - slf.num_rows;
            df = df.head(Some(limit));
        }
        slf.num_rows += df.height();
        Some(df.into())
    }
}
