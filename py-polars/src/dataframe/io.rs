use std::io::BufWriter;
use std::num::NonZeroUsize;
use std::ops::Deref;

#[cfg(feature = "avro")]
use polars::io::avro::AvroCompression;
use polars::io::mmap::{try_create_file, ReaderBytes};
use polars::io::RowIndex;
#[cfg(feature = "parquet")]
use polars_parquet::arrow::write::StatisticsOptions;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;

use super::*;
#[cfg(feature = "parquet")]
use crate::conversion::parse_parquet_compression;
use crate::conversion::Wrap;
use crate::file::{
    get_either_file, get_file_like, get_mmap_bytes_reader, get_mmap_bytes_reader_and_path,
    read_if_bytesio, EitherRustPythonFile,
};

#[pymethods]
impl PyDataFrame {
    #[staticmethod]
    #[cfg(feature = "csv")]
    #[pyo3(signature = (
    py_f, infer_schema_length, chunk_size, has_header, ignore_errors, n_rows,
    skip_rows, projection, separator, rechunk, columns, encoding, n_threads, path,
    overwrite_dtype, overwrite_dtype_slice, low_memory, comment_prefix, quote_char,
    null_values, missing_utf8_is_empty_string, try_parse_dates, skip_rows_after_header,
    row_index, sample_size, eol_char, raise_if_empty, truncate_ragged_lines, decimal_comma, schema)
)]
    pub fn read_csv(
        py: Python,
        mut py_f: Bound<PyAny>,
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
        path: Option<String>,
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
        schema: Option<Wrap<Schema>>,
    ) -> PyResult<Self> {
        let null_values = null_values.map(|w| w.0);
        let eol_char = eol_char.as_bytes()[0];
        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: Arc::from(name.as_str()),
            offset,
        });
        let quote_char = quote_char.and_then(|s| s.as_bytes().first().copied());

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

        py_f = read_if_bytesio(py_f);
        let mmap_bytes_r = get_mmap_bytes_reader(&py_f)?;
        let df = py.allow_threads(move || {
            CsvReadOptions::default()
                .with_path(path)
                .with_infer_schema_length(infer_schema_length)
                .with_has_header(has_header)
                .with_n_rows(n_rows)
                .with_skip_rows(skip_rows)
                .with_ignore_errors(ignore_errors)
                .with_projection(projection.map(Arc::new))
                .with_rechunk(rechunk)
                .with_chunk_size(chunk_size)
                .with_columns(columns.map(Arc::new))
                .with_n_threads(n_threads)
                .with_schema_overwrite(overwrite_dtype.map(Arc::new))
                .with_dtype_overwrite(overwrite_dtype_slice.map(Arc::new))
                .with_schema(schema.map(|schema| Arc::new(schema.0)))
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
                .into_reader_with_file_handle(mmap_bytes_r)
                .finish()
                .map_err(PyPolarsErr::from)
        })?;
        Ok(df.into())
    }

    #[staticmethod]
    #[cfg(feature = "parquet")]
    #[pyo3(signature = (py_f, columns, projection, n_rows, row_index, low_memory, parallel, use_statistics, rechunk))]
    pub fn read_parquet(
        py: Python,
        py_f: PyObject,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        n_rows: Option<usize>,
        row_index: Option<(String, IdxSize)>,
        low_memory: bool,
        parallel: Wrap<ParallelStrategy>,
        use_statistics: bool,
        rechunk: bool,
    ) -> PyResult<Self> {
        use EitherRustPythonFile::*;

        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: Arc::from(name.as_str()),
            offset,
        });
        let result = match get_either_file(py_f, false)? {
            Py(f) => {
                let buf = f.as_buffer();
                py.allow_threads(move || {
                    ParquetReader::new(buf)
                        .with_projection(projection)
                        .with_columns(columns)
                        .read_parallel(parallel.0)
                        .with_n_rows(n_rows)
                        .with_row_index(row_index)
                        .set_low_memory(low_memory)
                        .use_statistics(use_statistics)
                        .set_rechunk(rechunk)
                        .finish()
                })
            },
            Rust(f) => py.allow_threads(move || {
                ParquetReader::new(f.into_inner())
                    .with_projection(projection)
                    .with_columns(columns)
                    .read_parallel(parallel.0)
                    .with_n_rows(n_rows)
                    .with_row_index(row_index)
                    .use_statistics(use_statistics)
                    .set_rechunk(rechunk)
                    .finish()
            }),
        };
        let df = result.map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    #[cfg(feature = "json")]
    pub fn deserialize(py: Python, mut py_f: Bound<PyAny>) -> PyResult<Self> {
        use crate::file::read_if_bytesio;
        py_f = read_if_bytesio(py_f);
        let mmap_bytes_r = get_mmap_bytes_reader(&py_f)?;

        py.allow_threads(move || {
            let mmap_read: ReaderBytes = (&mmap_bytes_r).into();
            let bytes = mmap_read.deref();
            match serde_json::from_slice::<DataFrame>(bytes) {
                Ok(df) => Ok(df.into()),
                Err(e) => {
                    let msg = format!("{e}");
                    let e = PyPolarsErr::from(PolarsError::ComputeError(msg.into()));
                    Err(PyErr::from(e))
                },
            }
        })
    }

    #[staticmethod]
    #[cfg(feature = "json")]
    pub fn read_json(
        py: Python,
        mut py_f: Bound<PyAny>,
        infer_schema_length: Option<usize>,
        schema: Option<Wrap<Schema>>,
        schema_overrides: Option<Wrap<Schema>>,
    ) -> PyResult<Self> {
        assert!(infer_schema_length != Some(0));
        use crate::file::read_if_bytesio;
        py_f = read_if_bytesio(py_f);
        let mmap_bytes_r = get_mmap_bytes_reader(&py_f)?;

        py.allow_threads(move || {
            let mut builder = JsonReader::new(mmap_bytes_r)
                .with_json_format(JsonFormat::Json)
                .infer_schema_len(infer_schema_length.and_then(NonZeroUsize::new));

            if let Some(schema) = schema {
                builder = builder.with_schema(Arc::new(schema.0));
            }

            if let Some(schema) = schema_overrides.as_ref() {
                builder = builder.with_schema_overwrite(&schema.0);
            }

            let out = builder.finish().map_err(PyPolarsErr::from)?;
            Ok(out.into())
        })
    }

    #[staticmethod]
    #[cfg(feature = "json")]
    pub fn read_ndjson(
        py: Python,
        mut py_f: Bound<PyAny>,
        ignore_errors: bool,
        schema: Option<Wrap<Schema>>,
        schema_overrides: Option<Wrap<Schema>>,
    ) -> PyResult<Self> {
        py_f = read_if_bytesio(py_f);
        let mmap_bytes_r = get_mmap_bytes_reader(&py_f)?;

        let mut builder = JsonReader::new(mmap_bytes_r)
            .with_json_format(JsonFormat::JsonLines)
            .with_ignore_errors(ignore_errors);

        if let Some(schema) = schema {
            builder = builder.with_schema(Arc::new(schema.0));
        }

        if let Some(schema) = schema_overrides.as_ref() {
            builder = builder.with_schema_overwrite(&schema.0);
        }

        let out = py
            .allow_threads(move || builder.finish())
            .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
        Ok(out.into())
    }

    #[staticmethod]
    #[cfg(feature = "ipc")]
    #[pyo3(signature = (py_f, columns, projection, n_rows, row_index, memory_map))]
    pub fn read_ipc(
        py: Python,
        mut py_f: Bound<PyAny>,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        n_rows: Option<usize>,
        row_index: Option<(String, IdxSize)>,
        memory_map: bool,
    ) -> PyResult<Self> {
        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: Arc::from(name.as_str()),
            offset,
        });
        py_f = read_if_bytesio(py_f);
        let (mmap_bytes_r, mmap_path) = get_mmap_bytes_reader_and_path(&py_f)?;

        let mmap_path = if memory_map { mmap_path } else { None };
        let df = py.allow_threads(move || {
            IpcReader::new(mmap_bytes_r)
                .with_projection(projection)
                .with_columns(columns)
                .with_n_rows(n_rows)
                .with_row_index(row_index)
                .memory_mapped(mmap_path)
                .finish()
                .map_err(PyPolarsErr::from)
        })?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    #[cfg(feature = "ipc_streaming")]
    #[pyo3(signature = (py_f, columns, projection, n_rows, row_index, rechunk))]
    pub fn read_ipc_stream(
        py: Python,
        mut py_f: Bound<PyAny>,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        n_rows: Option<usize>,
        row_index: Option<(String, IdxSize)>,
        rechunk: bool,
    ) -> PyResult<Self> {
        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: Arc::from(name.as_str()),
            offset,
        });
        py_f = read_if_bytesio(py_f);
        let mmap_bytes_r = get_mmap_bytes_reader(&py_f)?;
        let df = py.allow_threads(move || {
            IpcStreamReader::new(mmap_bytes_r)
                .with_projection(projection)
                .with_columns(columns)
                .with_n_rows(n_rows)
                .with_row_index(row_index)
                .set_rechunk(rechunk)
                .finish()
                .map_err(PyPolarsErr::from)
        })?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    #[cfg(feature = "avro")]
    #[pyo3(signature = (py_f, columns, projection, n_rows))]
    pub fn read_avro(
        py: Python,
        py_f: PyObject,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        n_rows: Option<usize>,
    ) -> PyResult<Self> {
        use polars::io::avro::AvroReader;

        let file = get_file_like(py_f, false)?;
        let df = py.allow_threads(move || {
            AvroReader::new(file)
                .with_projection(projection)
                .with_columns(columns)
                .with_n_rows(n_rows)
                .finish()
                .map_err(PyPolarsErr::from)
        })?;
        Ok(PyDataFrame::new(df))
    }

    #[cfg(feature = "csv")]
    pub fn write_csv(
        &mut self,
        py: Python,
        py_f: PyObject,
        include_bom: bool,
        include_header: bool,
        separator: u8,
        line_terminator: String,
        quote_char: u8,
        batch_size: NonZeroUsize,
        datetime_format: Option<String>,
        date_format: Option<String>,
        time_format: Option<String>,
        float_precision: Option<usize>,
        null_value: Option<String>,
        quote_style: Option<Wrap<QuoteStyle>>,
    ) -> PyResult<()> {
        let null = null_value.unwrap_or_default();

        if let Ok(s) = py_f.extract::<PyBackedStr>(py) {
            let f = std::fs::File::create(&*s)?;
            py.allow_threads(|| {
                // No need for a buffered writer, because the csv writer does internal buffering.
                CsvWriter::new(f)
                    .include_bom(include_bom)
                    .include_header(include_header)
                    .with_separator(separator)
                    .with_line_terminator(line_terminator)
                    .with_quote_char(quote_char)
                    .with_batch_size(batch_size)
                    .with_datetime_format(datetime_format)
                    .with_date_format(date_format)
                    .with_time_format(time_format)
                    .with_float_precision(float_precision)
                    .with_null_value(null)
                    .with_quote_style(quote_style.map(|wrap| wrap.0).unwrap_or_default())
                    .finish(&mut self.df)
                    .map_err(PyPolarsErr::from)
            })?;
        } else {
            let mut buf = get_file_like(py_f, true)?;
            CsvWriter::new(&mut buf)
                .include_bom(include_bom)
                .include_header(include_header)
                .with_separator(separator)
                .with_line_terminator(line_terminator)
                .with_quote_char(quote_char)
                .with_batch_size(batch_size)
                .with_datetime_format(datetime_format)
                .with_date_format(date_format)
                .with_time_format(time_format)
                .with_float_precision(float_precision)
                .with_null_value(null)
                .with_quote_style(quote_style.map(|wrap| wrap.0).unwrap_or_default())
                .finish(&mut self.df)
                .map_err(PyPolarsErr::from)?;
        }

        Ok(())
    }

    #[cfg(feature = "parquet")]
    #[pyo3(signature = (py_f, compression, compression_level, statistics, row_group_size, data_page_size))]
    pub fn write_parquet(
        &mut self,
        py: Python,
        py_f: PyObject,
        compression: &str,
        compression_level: Option<i32>,
        statistics: Wrap<StatisticsOptions>,
        row_group_size: Option<usize>,
        data_page_size: Option<usize>,
    ) -> PyResult<()> {
        let compression = parse_parquet_compression(compression, compression_level)?;

        if let Ok(s) = py_f.extract::<PyBackedStr>(py) {
            let f = std::fs::File::create(&*s)?;
            py.allow_threads(|| {
                ParquetWriter::new(f)
                    .with_compression(compression)
                    .with_statistics(statistics.0)
                    .with_row_group_size(row_group_size)
                    .with_data_page_size(data_page_size)
                    .finish(&mut self.df)
                    .map_err(PyPolarsErr::from)
            })?;
        } else {
            let buf = get_file_like(py_f, true)?;
            ParquetWriter::new(buf)
                .with_compression(compression)
                .with_statistics(statistics.0)
                .with_row_group_size(row_group_size)
                .with_data_page_size(data_page_size)
                .finish(&mut self.df)
                .map_err(PyPolarsErr::from)?;
        }

        Ok(())
    }

    #[cfg(feature = "json")]
    pub fn serialize(&mut self, py_f: PyObject) -> PyResult<()> {
        let file = BufWriter::new(get_file_like(py_f, true)?);
        serde_json::to_writer(file, &self.df)
            .map_err(|e| polars_err!(ComputeError: "{e}"))
            .map_err(|e| PyPolarsErr::Other(format!("{e}")).into())
    }

    #[cfg(feature = "json")]
    pub fn write_json(&mut self, py_f: PyObject) -> PyResult<()> {
        let file = BufWriter::new(get_file_like(py_f, true)?);

        JsonWriter::new(file)
            .with_json_format(JsonFormat::Json)
            .finish(&mut self.df)
            .map_err(|e| PyPolarsErr::Other(format!("{e}")).into())
    }

    #[cfg(feature = "json")]
    pub fn write_ndjson(&mut self, py_f: PyObject) -> PyResult<()> {
        let file = BufWriter::new(get_file_like(py_f, true)?);

        let r = JsonWriter::new(file)
            .with_json_format(JsonFormat::JsonLines)
            .finish(&mut self.df);

        r.map_err(|e| PyPolarsErr::Other(format!("{e}")).into())
    }

    #[cfg(feature = "ipc")]
    pub fn write_ipc(
        &mut self,
        py: Python,
        py_f: PyObject,
        compression: Wrap<Option<IpcCompression>>,
        future: bool,
    ) -> PyResult<()> {
        if let Ok(s) = py_f.extract::<PyBackedStr>(py) {
            let s: &str = s.as_ref();
            let path = std::path::Path::new(s);
            let f = try_create_file(path).map_err(PyPolarsErr::from)?;
            py.allow_threads(|| {
                IpcWriter::new(f)
                    .with_compression(compression.0)
                    .with_pl_flavor(future)
                    .finish(&mut self.df)
                    .map_err(PyPolarsErr::from)
            })?;
        } else {
            let mut buf = get_file_like(py_f, true)?;

            IpcWriter::new(&mut buf)
                .with_compression(compression.0)
                .with_pl_flavor(future)
                .finish(&mut self.df)
                .map_err(PyPolarsErr::from)?;
        }
        Ok(())
    }

    #[cfg(feature = "ipc_streaming")]
    pub fn write_ipc_stream(
        &mut self,
        py: Python,
        py_f: PyObject,
        compression: Wrap<Option<IpcCompression>>,
    ) -> PyResult<()> {
        if let Ok(s) = py_f.extract::<PyBackedStr>(py) {
            let f = std::fs::File::create(&*s)?;
            py.allow_threads(|| {
                IpcStreamWriter::new(f)
                    .with_compression(compression.0)
                    .finish(&mut self.df)
                    .map_err(PyPolarsErr::from)
            })?;
        } else {
            let mut buf = get_file_like(py_f, true)?;

            IpcStreamWriter::new(&mut buf)
                .with_compression(compression.0)
                .finish(&mut self.df)
                .map_err(PyPolarsErr::from)?;
        }
        Ok(())
    }

    #[cfg(feature = "avro")]
    #[pyo3(signature = (py_f, compression, name))]
    pub fn write_avro(
        &mut self,
        py: Python,
        py_f: PyObject,
        compression: Wrap<Option<AvroCompression>>,
        name: String,
    ) -> PyResult<()> {
        use polars::io::avro::AvroWriter;

        if let Ok(s) = py_f.extract::<PyBackedStr>(py) {
            let f = std::fs::File::create(&*s)?;
            AvroWriter::new(f)
                .with_compression(compression.0)
                .with_name(name)
                .finish(&mut self.df)
                .map_err(PyPolarsErr::from)?;
        } else {
            let mut buf = get_file_like(py_f, true)?;
            AvroWriter::new(&mut buf)
                .with_compression(compression.0)
                .with_name(name)
                .finish(&mut self.df)
                .map_err(PyPolarsErr::from)?;
        }

        Ok(())
    }
}
