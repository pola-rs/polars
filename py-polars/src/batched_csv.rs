use std::path::PathBuf;

use polars::io::mmap::MmapBytesReader;
use polars::io::RowCount;
use polars::prelude::*;
use pyo3::prelude::*;

use crate::prelude::read_impl::OwnedBatchedCsvReader;
use crate::{PyDataFrame, PyPolarsErr, Wrap};

#[pyclass]
#[repr(transparent)]
pub struct PyBatchedCsv {
    // option because we cannot get a self by value in pyo3
    pub reader: OwnedBatchedCsvReader,
}

#[pymethods]
#[allow(clippy::wrong_self_convention, clippy::should_implement_trait)]
impl PyBatchedCsv {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        infer_schema_length, chunk_size, has_header, ignore_errors, n_rows, skip_rows,
        projection, sep, rechunk, columns, encoding, n_threads, path, overwrite_dtype,
        overwrite_dtype_slice, low_memory, comment_char, quote_char, null_values,
        missing_utf8_is_empty_string, parse_dates, skip_rows_after_header, row_count,
        sample_size, eol_char)
    )]
    fn new(
        infer_schema_length: Option<usize>,
        chunk_size: usize,
        has_header: bool,
        ignore_errors: bool,
        n_rows: Option<usize>,
        skip_rows: usize,
        projection: Option<Vec<usize>>,
        sep: &str,
        rechunk: bool,
        columns: Option<Vec<String>>,
        encoding: Wrap<CsvEncoding>,
        n_threads: Option<usize>,
        path: PathBuf,
        overwrite_dtype: Option<Vec<(&str, Wrap<DataType>)>>,
        overwrite_dtype_slice: Option<Vec<Wrap<DataType>>>,
        low_memory: bool,
        comment_char: Option<&str>,
        quote_char: Option<&str>,
        null_values: Option<Wrap<NullValues>>,
        missing_utf8_is_empty_string: bool,
        parse_dates: bool,
        skip_rows_after_header: usize,
        row_count: Option<(String, IdxSize)>,
        sample_size: usize,
        eol_char: &str,
    ) -> PyResult<PyBatchedCsv> {
        let null_values = null_values.map(|w| w.0);
        let comment_char = comment_char.map(|s| s.as_bytes()[0]);
        let eol_char = eol_char.as_bytes()[0];
        let row_count = row_count.map(|(name, offset)| RowCount { name, offset });
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
            let fields = overwrite_dtype.iter().map(|(name, dtype)| {
                let dtype = dtype.0.clone();
                Field::new(name, dtype)
            });
            Schema::from(fields)
        });

        let overwrite_dtype_slice = overwrite_dtype_slice.map(|overwrite_dtype| {
            overwrite_dtype
                .iter()
                .map(|dt| dt.0.clone())
                .collect::<Vec<_>>()
        });

        let file = std::fs::File::open(path).map_err(PyPolarsErr::from)?;
        let reader = Box::new(file) as Box<dyn MmapBytesReader>;
        let reader = CsvReader::new(reader)
            .infer_schema(infer_schema_length)
            .has_header(has_header)
            .with_n_rows(n_rows)
            .with_delimiter(sep.as_bytes()[0])
            .with_skip_rows(skip_rows)
            .with_ignore_errors(ignore_errors)
            .with_projection(projection)
            .with_rechunk(rechunk)
            .with_chunk_size(chunk_size)
            .with_encoding(encoding.0)
            .with_columns(columns)
            .with_n_threads(n_threads)
            .with_dtypes_slice(overwrite_dtype_slice.as_deref())
            .with_missing_is_null(!missing_utf8_is_empty_string)
            .low_memory(low_memory)
            .with_comment_char(comment_char)
            .with_null_values(null_values)
            .with_parse_dates(parse_dates)
            .with_quote_char(quote_char)
            .with_end_of_line_char(eol_char)
            .with_skip_rows_after_header(skip_rows_after_header)
            .with_row_count(row_count)
            .sample_size(sample_size)
            .batched(overwrite_dtype.map(Arc::new))
            .map_err(PyPolarsErr::from)?;

        Ok(PyBatchedCsv { reader })
    }

    fn next_batches(&mut self, n: usize) -> PyResult<Option<Vec<PyDataFrame>>> {
        let batches = self.reader.next_batches(n).map_err(PyPolarsErr::from)?;
        Ok(batches.map(|batches| batches.into_iter().map(|out| out.1.into()).collect()))
    }
}
