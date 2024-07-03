use std::path::PathBuf;
use std::sync::Mutex;

use polars::io::csv::read::OwnedBatchedCsvReader;
use polars::io::mmap::MmapBytesReader;
use polars::io::RowIndex;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;

use crate::error::PyPolarsErr;
use crate::{PyDataFrame, Wrap};

#[pyclass]
#[repr(transparent)]
pub struct PyBatchedCsv {
    reader: Mutex<OwnedBatchedCsvReader>,
}

#[pymethods]
#[allow(clippy::wrong_self_convention, clippy::should_implement_trait)]
impl PyBatchedCsv {
    #[staticmethod]
    #[pyo3(signature = (
        infer_schema_length, chunk_size, has_header, ignore_errors, n_rows, skip_rows,
        projection, separator, rechunk, columns, encoding, n_threads, path, overwrite_dtype,
        overwrite_dtype_slice, low_memory, comment_prefix, quote_char, null_values,
        missing_utf8_is_empty_string, try_parse_dates, skip_rows_after_header, row_index,
        sample_size, eol_char, raise_if_empty, truncate_ragged_lines, decimal_comma)
    )]
    fn new(
        infer_schema_length: Option<usize>,
        chunk_size: usize,
        has_header: bool,
        ignore_errors: bool,
        n_rows: Option<usize>,
        skip_rows: usize,
        projection: Option<Vec<usize>>,
        separator: &str,
        rechunk: bool,
        columns: Option<Vec<String>>,
        encoding: Wrap<CsvEncoding>,
        n_threads: Option<usize>,
        path: PathBuf,
        overwrite_dtype: Option<Vec<(PyBackedStr, Wrap<DataType>)>>,
        overwrite_dtype_slice: Option<Vec<Wrap<DataType>>>,
        low_memory: bool,
        comment_prefix: Option<&str>,
        quote_char: Option<&str>,
        null_values: Option<Wrap<NullValues>>,
        missing_utf8_is_empty_string: bool,
        try_parse_dates: bool,
        skip_rows_after_header: usize,
        row_index: Option<(String, IdxSize)>,
        sample_size: usize,
        eol_char: &str,
        raise_if_empty: bool,
        truncate_ragged_lines: bool,
        decimal_comma: bool,
    ) -> PyResult<PyBatchedCsv> {
        let null_values = null_values.map(|w| w.0);
        let eol_char = eol_char.as_bytes()[0];
        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: Arc::from(name.as_str()),
            offset,
        });
        let quote_char = if let Some(s) = quote_char {
            if s.is_empty() {
                None
            } else {
                Some(s.as_bytes()[0])
            }
        } else {
            None
        };

        let overwrite_dtype = overwrite_dtype.map(|overwrite_dtype| {
            overwrite_dtype
                .iter()
                .map(|(name, dtype)| {
                    let dtype = dtype.0.clone();
                    Field::new(name, dtype)
                })
                .collect::<Schema>()
        });

        let overwrite_dtype_slice = overwrite_dtype_slice.map(|overwrite_dtype| {
            overwrite_dtype
                .iter()
                .map(|dt| dt.0.clone())
                .collect::<Vec<_>>()
        });

        let file = std::fs::File::open(path).map_err(PyPolarsErr::from)?;
        let reader = Box::new(file) as Box<dyn MmapBytesReader>;
        let reader = CsvReadOptions::default()
            .with_infer_schema_length(infer_schema_length)
            .with_has_header(has_header)
            .with_n_rows(n_rows)
            .with_skip_rows(skip_rows)
            .with_ignore_errors(ignore_errors)
            .with_projection(projection.map(Arc::new))
            .with_rechunk(rechunk)
            .with_chunk_size(chunk_size)
            .with_columns(columns.map(Arc::from))
            .with_n_threads(n_threads)
            .with_dtype_overwrite(overwrite_dtype_slice.map(Arc::new))
            .with_low_memory(low_memory)
            .with_skip_rows_after_header(skip_rows_after_header)
            .with_row_index(row_index)
            .with_sample_size(sample_size)
            .with_raise_if_empty(raise_if_empty)
            .with_parse_options(
                CsvParseOptions::default()
                    .with_separator(separator.as_bytes()[0])
                    .with_encoding(encoding.0)
                    .with_missing_is_null(!missing_utf8_is_empty_string)
                    .with_comment_prefix(comment_prefix)
                    .with_null_values(null_values)
                    .with_try_parse_dates(try_parse_dates)
                    .with_quote_char(quote_char)
                    .with_eol_char(eol_char)
                    .with_truncate_ragged_lines(truncate_ragged_lines)
                    .with_decimal_comma(decimal_comma),
            )
            .into_reader_with_file_handle(reader);

        let reader = reader
            .batched(overwrite_dtype.map(Arc::new))
            .map_err(PyPolarsErr::from)?;

        Ok(PyBatchedCsv {
            reader: Mutex::new(reader),
        })
    }

    fn next_batches(&self, py: Python, n: usize) -> PyResult<Option<Vec<PyDataFrame>>> {
        let reader = &self.reader;
        let batches = py.allow_threads(move || {
            reader
                .lock()
                .map_err(|e| PyPolarsErr::Other(e.to_string()))?
                .next_batches(n)
                .map_err(PyPolarsErr::from)
        })?;

        // SAFETY: same memory layout
        let batches = unsafe {
            std::mem::transmute::<Option<Vec<DataFrame>>, Option<Vec<PyDataFrame>>>(batches)
        };
        Ok(batches)
    }
}
