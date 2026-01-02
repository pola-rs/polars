use polars::io::SerReader;
use polars::io::mmap::MmapBytesReader;
use polars::io::pcap::PcapReader;
use polars_core::prelude::*;
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::file::get_mmap_bytes_reader;

#[pyclass]
pub struct PyPcapReader {
    reader: PcapReader<Box<dyn MmapBytesReader>>,
}

#[pymethods]
impl PyPcapReader {
    #[new]
    #[pyo3(signature = (py_f, n_rows))]
    pub fn new(py_f: Bound<PyAny>, n_rows: Option<usize>) -> PyResult<Self> {
        let file = get_mmap_bytes_reader(&py_f)?;
        let reader = PcapReader::new(file).with_n_rows(n_rows);
        Ok(Self { reader })
    }

    pub fn schema(&self) -> Wrap<Schema> {
        let schema = Schema::from_iter(vec![
            Field::new("time_s".into(), DataType::Int64),
            Field::new("time_ns".into(), DataType::UInt32),
            Field::new("incl_len".into(), DataType::UInt32),
            Field::new("orig_len".into(), DataType::UInt32),
            Field::new("data".into(), DataType::Binary),
        ]);
        Wrap(schema)
    }

    pub fn next_batch(&mut self, batch_size: usize) -> PyResult<Option<PyDataFrame>> {
        self.reader
            .next_batch(batch_size)
            .map_err(|e| PyErr::from(PyPolarsErr::from(e)))
            .map(|opt_df| opt_df.map(PyDataFrame::new))
    }
}
