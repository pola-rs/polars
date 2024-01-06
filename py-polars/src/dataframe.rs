use std::io::{BufWriter, Cursor};
use std::ops::Deref;

use either::Either;
use numpy::IntoPyArray;
use polars::frame::row::{rows_to_schema_supertypes, Row};
use polars::frame::NullStrategy;
#[cfg(feature = "avro")]
use polars::io::avro::AvroCompression;
#[cfg(feature = "ipc")]
use polars::io::ipc::IpcCompression;
use polars::io::mmap::ReaderBytes;
use polars::io::RowCount;
use polars::prelude::*;
use polars_core::export::arrow::datatypes::IntegerType;
use polars_core::frame::explode::MeltArgs;
use polars_core::frame::*;
use polars_core::prelude::IndexOrder;
use polars_core::utils::arrow::compute::cast::CastOptions;
use polars_core::utils::try_get_supertype;
#[cfg(feature = "pivot")]
use polars_lazy::frame::pivot::{pivot, pivot_stable};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

#[cfg(feature = "parquet")]
use crate::conversion::parse_parquet_compression;
use crate::conversion::{ObjectValue, Wrap};
use crate::error::PyPolarsErr;
use crate::file::{get_either_file, get_file_like, get_mmap_bytes_reader, EitherRustPythonFile};
use crate::map::dataframe::{
    apply_lambda_unknown, apply_lambda_with_bool_out_type, apply_lambda_with_primitive_out_type,
    apply_lambda_with_string_out_type,
};
use crate::prelude::{dicts_to_rows, strings_to_smartstrings};
use crate::series::{PySeries, ToPySeries, ToSeries};
use crate::{arrow_interop, py_modules, PyExpr, PyLazyFrame};

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyDataFrame {
    pub df: DataFrame,
}

impl PyDataFrame {
    pub(crate) fn new(df: DataFrame) -> Self {
        PyDataFrame { df }
    }

    fn finish_from_rows(
        rows: Vec<Row>,
        infer_schema_length: Option<usize>,
        schema: Option<Schema>,
        schema_overrides_by_idx: Option<Vec<(usize, DataType)>>,
    ) -> PyResult<Self> {
        // Object builder must be registered, this is done on import.
        let mut final_schema =
            rows_to_schema_supertypes(&rows, infer_schema_length.map(|n| std::cmp::max(1, n)))
                .map_err(PyPolarsErr::from)?;

        // Erase scale from inferred decimals.
        for dtype in final_schema.iter_dtypes_mut() {
            if let DataType::Decimal(_, _) = dtype {
                *dtype = DataType::Decimal(None, None)
            }
        }

        // Integrate explicit/inferred schema.
        if let Some(schema) = schema {
            for (i, (name, dtype)) in schema.into_iter().enumerate() {
                if let Some((name_, dtype_)) = final_schema.get_at_index_mut(i) {
                    *name_ = name;

                    // If schema dtype is Unknown, overwrite with inferred datatype.
                    if !matches!(dtype, DataType::Unknown) {
                        *dtype_ = dtype;
                    }
                } else {
                    final_schema.with_column(name, dtype);
                }
            }
        }

        // Optional per-field overrides; these supersede default/inferred dtypes.
        if let Some(overrides) = schema_overrides_by_idx {
            for (i, dtype) in overrides {
                if let Some((_, dtype_)) = final_schema.get_at_index_mut(i) {
                    if !matches!(dtype, DataType::Unknown) {
                        *dtype_ = dtype;
                    }
                }
            }
        }
        let df =
            DataFrame::from_rows_and_schema(&rows, &final_schema).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    #[cfg(feature = "ipc_streaming")]
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        // Used in pickle/pickling
        let mut buf: Vec<u8> = vec![];
        IpcStreamWriter::new(&mut buf)
            .finish(&mut self.df.clone())
            .expect("ipc writer");
        Ok(PyBytes::new(py, &buf).to_object(py))
    }

    #[cfg(feature = "ipc_streaming")]
    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        // Used in pickle/pickling
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                let c = Cursor::new(s.as_bytes());
                let reader = IpcStreamReader::new(c);

                reader
                    .finish()
                    .map(|df| {
                        self.df = df;
                    })
                    .map_err(|e| PyPolarsErr::from(e).into())
            },
            Err(e) => Err(e),
        }
    }
}

impl From<DataFrame> for PyDataFrame {
    fn from(df: DataFrame) -> Self {
        PyDataFrame { df }
    }
}

#[pymethods]
#[allow(
    clippy::wrong_self_convention,
    clippy::should_implement_trait,
    clippy::len_without_is_empty
)]
impl PyDataFrame {
    pub fn into_raw_parts(&mut self) -> (usize, usize, usize) {
        // Used for polars-lazy python node. This takes the dataframe from
        // underneath of you, so don't use this anywhere else.
        let mut df = std::mem::take(&mut self.df);
        let cols = unsafe { std::mem::take(df.get_columns_mut()) };
        let (ptr, len, cap) = cols.into_raw_parts();
        (ptr as usize, len, cap)
    }

    #[new]
    pub fn __init__(columns: Vec<PySeries>) -> PyResult<Self> {
        let columns = columns.to_series();
        let df = DataFrame::new(columns).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn estimated_size(&self) -> usize {
        self.df.estimated_size()
    }

    pub fn dtype_strings(&self) -> Vec<String> {
        self.df
            .get_columns()
            .iter()
            .map(|s| format!("{}", s.dtype()))
            .collect()
    }

    #[staticmethod]
    #[cfg(feature = "csv")]
    #[pyo3(signature = (
        py_f, infer_schema_length, chunk_size, has_header, ignore_errors, n_rows,
        skip_rows, projection, separator, rechunk, columns, encoding, n_threads, path,
        overwrite_dtype, overwrite_dtype_slice, low_memory, comment_prefix, quote_char,
        null_values, missing_utf8_is_empty_string, try_parse_dates, skip_rows_after_header,
        row_count, sample_size, eol_char, raise_if_empty, truncate_ragged_lines, schema)
    )]
    pub fn read_csv(
        py_f: &PyAny,
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
        overwrite_dtype: Option<Vec<(&str, Wrap<DataType>)>>,
        overwrite_dtype_slice: Option<Vec<Wrap<DataType>>>,
        low_memory: bool,
        comment_prefix: Option<&str>,
        quote_char: Option<&str>,
        null_values: Option<Wrap<NullValues>>,
        missing_utf8_is_empty_string: bool,
        try_parse_dates: bool,
        skip_rows_after_header: usize,
        row_count: Option<(String, IdxSize)>,
        sample_size: usize,
        eol_char: &str,
        raise_if_empty: bool,
        truncate_ragged_lines: bool,
        schema: Option<Wrap<Schema>>,
    ) -> PyResult<Self> {
        let null_values = null_values.map(|w| w.0);
        let eol_char = eol_char.as_bytes()[0];
        let row_count = row_count.map(|(name, offset)| RowCount { name, offset });
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

        let mmap_bytes_r = get_mmap_bytes_reader(py_f)?;
        let df = CsvReader::new(mmap_bytes_r)
            .infer_schema(infer_schema_length)
            .has_header(has_header)
            .with_n_rows(n_rows)
            .with_separator(separator.as_bytes()[0])
            .with_skip_rows(skip_rows)
            .with_ignore_errors(ignore_errors)
            .with_projection(projection)
            .with_rechunk(rechunk)
            .with_chunk_size(chunk_size)
            .with_encoding(encoding.0)
            .with_columns(columns)
            .with_n_threads(n_threads)
            .with_path(path)
            .with_dtypes(overwrite_dtype.map(Arc::new))
            .with_dtypes_slice(overwrite_dtype_slice.as_deref())
            .with_schema(schema.map(|schema| Arc::new(schema.0)))
            .low_memory(low_memory)
            .with_null_values(null_values)
            .with_missing_is_null(!missing_utf8_is_empty_string)
            .with_comment_prefix(comment_prefix)
            .with_try_parse_dates(try_parse_dates)
            .with_quote_char(quote_char)
            .with_end_of_line_char(eol_char)
            .with_skip_rows_after_header(skip_rows_after_header)
            .with_row_count(row_count)
            .sample_size(sample_size)
            .raise_if_empty(raise_if_empty)
            .truncate_ragged_lines(truncate_ragged_lines)
            .finish()
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    #[staticmethod]
    #[cfg(feature = "parquet")]
    #[pyo3(signature = (py_f, columns, projection, n_rows, parallel, row_count, low_memory, use_statistics, rechunk))]
    pub fn read_parquet(
        py_f: PyObject,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        n_rows: Option<usize>,
        parallel: Wrap<ParallelStrategy>,
        row_count: Option<(String, IdxSize)>,
        low_memory: bool,
        use_statistics: bool,
        rechunk: bool,
    ) -> PyResult<Self> {
        use EitherRustPythonFile::*;

        let row_count = row_count.map(|(name, offset)| RowCount { name, offset });
        let result = match get_either_file(py_f, false)? {
            Py(f) => {
                let buf = f.as_buffer();
                ParquetReader::new(buf)
                    .with_projection(projection)
                    .with_columns(columns)
                    .read_parallel(parallel.0)
                    .with_n_rows(n_rows)
                    .with_row_count(row_count)
                    .set_low_memory(low_memory)
                    .use_statistics(use_statistics)
                    .set_rechunk(rechunk)
                    .finish()
            },
            Rust(f) => ParquetReader::new(f.into_inner())
                .with_projection(projection)
                .with_columns(columns)
                .read_parallel(parallel.0)
                .with_n_rows(n_rows)
                .with_row_count(row_count)
                .use_statistics(use_statistics)
                .set_rechunk(rechunk)
                .finish(),
        };
        let df = result.map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    #[cfg(feature = "ipc")]
    #[pyo3(signature = (py_f, columns, projection, n_rows, row_count, memory_map))]
    pub fn read_ipc(
        py_f: &PyAny,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        n_rows: Option<usize>,
        row_count: Option<(String, IdxSize)>,
        memory_map: bool,
    ) -> PyResult<Self> {
        let row_count = row_count.map(|(name, offset)| RowCount { name, offset });
        let mmap_bytes_r = get_mmap_bytes_reader(py_f)?;
        let df = IpcReader::new(mmap_bytes_r)
            .with_projection(projection)
            .with_columns(columns)
            .with_n_rows(n_rows)
            .with_row_count(row_count)
            .memory_mapped(memory_map)
            .finish()
            .map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    #[cfg(feature = "ipc_streaming")]
    #[pyo3(signature = (py_f, columns, projection, n_rows, row_count, rechunk))]
    pub fn read_ipc_stream(
        py_f: &PyAny,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        n_rows: Option<usize>,
        row_count: Option<(String, IdxSize)>,
        rechunk: bool,
    ) -> PyResult<Self> {
        let row_count = row_count.map(|(name, offset)| RowCount { name, offset });
        let mmap_bytes_r = get_mmap_bytes_reader(py_f)?;
        let df = IpcStreamReader::new(mmap_bytes_r)
            .with_projection(projection)
            .with_columns(columns)
            .with_n_rows(n_rows)
            .with_row_count(row_count)
            .set_rechunk(rechunk)
            .finish()
            .map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    #[cfg(feature = "avro")]
    #[pyo3(signature = (py_f, columns, projection, n_rows))]
    pub fn read_avro(
        py_f: PyObject,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        n_rows: Option<usize>,
    ) -> PyResult<Self> {
        use polars::io::avro::AvroReader;

        let file = get_file_like(py_f, false)?;
        let df = AvroReader::new(file)
            .with_projection(projection)
            .with_columns(columns)
            .with_n_rows(n_rows)
            .finish()
            .map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
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

        if let Ok(s) = py_f.extract::<&str>(py) {
            let f = std::fs::File::create(s)?;
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

    #[staticmethod]
    #[cfg(feature = "json")]
    pub fn read_json(
        py_f: &PyAny,
        infer_schema_length: Option<usize>,
        schema: Option<Wrap<Schema>>,
        schema_overrides: Option<Wrap<Schema>>,
    ) -> PyResult<Self> {
        // memmap the file first.
        let mmap_bytes_r = get_mmap_bytes_reader(py_f)?;
        let mmap_read: ReaderBytes = (&mmap_bytes_r).into();
        let bytes = mmap_read.deref();

        // Happy path is our column oriented json as that is most performant,
        // on failure we try the arrow json reader instead, which is row-oriented.
        match serde_json::from_slice::<DataFrame>(bytes) {
            Ok(df) => Ok(df.into()),
            Err(e) => {
                let msg = format!("{e}");
                if msg.contains("successful parse invalid data") {
                    let e = PyPolarsErr::from(PolarsError::ComputeError(msg.into()));
                    Err(PyErr::from(e))
                } else {
                    let mut builder = JsonReader::new(mmap_bytes_r)
                        .with_json_format(JsonFormat::Json)
                        .infer_schema_len(infer_schema_length);

                    if let Some(schema) = schema {
                        builder = builder.with_schema(Arc::new(schema.0));
                    }

                    if let Some(schema) = schema_overrides.as_ref() {
                        builder = builder.with_schema_overwrite(&schema.0);
                    }

                    let out = builder
                        .finish()
                        .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
                    Ok(out.into())
                }
            },
        }
    }

    #[staticmethod]
    #[cfg(feature = "json")]
    pub fn read_ndjson(
        py_f: &PyAny,
        ignore_errors: bool,
        schema: Option<Wrap<Schema>>,
        schema_overrides: Option<Wrap<Schema>>,
    ) -> PyResult<Self> {
        let mmap_bytes_r = get_mmap_bytes_reader(py_f)?;

        let mut builder = JsonReader::new(mmap_bytes_r)
            .with_json_format(JsonFormat::JsonLines)
            .with_ignore_errors(ignore_errors);

        if let Some(schema) = schema {
            builder = builder.with_schema(Arc::new(schema.0));
        }

        if let Some(schema) = schema_overrides.as_ref() {
            builder = builder.with_schema_overwrite(&schema.0);
        }

        let out = builder
            .finish()
            .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
        Ok(out.into())
    }

    #[cfg(feature = "json")]
    pub fn write_json(&mut self, py_f: PyObject, pretty: bool, row_oriented: bool) -> PyResult<()> {
        let file = BufWriter::new(get_file_like(py_f, true)?);

        let r = match (pretty, row_oriented) {
            (_, true) => JsonWriter::new(file)
                .with_json_format(JsonFormat::Json)
                .finish(&mut self.df),
            (true, _) => serde_json::to_writer_pretty(file, &self.df)
                .map_err(|e| polars_err!(ComputeError: "{e}")),
            (false, _) => {
                serde_json::to_writer(file, &self.df).map_err(|e| polars_err!(ComputeError: "{e}"))
            },
        };
        r.map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
        Ok(())
    }

    #[cfg(feature = "json")]
    pub fn write_ndjson(&mut self, py_f: PyObject) -> PyResult<()> {
        let file = BufWriter::new(get_file_like(py_f, true)?);

        let r = JsonWriter::new(file)
            .with_json_format(JsonFormat::JsonLines)
            .finish(&mut self.df);

        r.map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
        Ok(())
    }

    #[staticmethod]
    pub fn from_arrow_record_batches(rb: Vec<&PyAny>) -> PyResult<Self> {
        let df = arrow_interop::to_rust::to_rust_df(&rb)?;
        Ok(Self::from(df))
    }

    // somehow from_rows did not work
    #[staticmethod]
    pub fn read_rows(
        rows: Vec<Wrap<Row>>,
        infer_schema_length: Option<usize>,
        schema: Option<Wrap<Schema>>,
    ) -> PyResult<Self> {
        // SAFETY: Wrap<T> is transparent.
        let rows = unsafe { std::mem::transmute::<Vec<Wrap<Row>>, Vec<Row>>(rows) };
        Self::finish_from_rows(rows, infer_schema_length, schema.map(|wrap| wrap.0), None)
    }

    #[staticmethod]
    pub fn read_dicts(
        dicts: &PyAny,
        infer_schema_length: Option<usize>,
        schema: Option<Wrap<Schema>>,
        schema_overrides: Option<Wrap<Schema>>,
    ) -> PyResult<Self> {
        // If given, read dict fields in schema order.
        let mut schema_columns = PlIndexSet::new();
        if let Some(s) = &schema {
            schema_columns.extend(s.0.iter_names().map(|n| n.to_string()))
        }
        let (rows, names) = dicts_to_rows(dicts, infer_schema_length, schema_columns)?;
        let mut schema_overrides_by_idx: Vec<(usize, DataType)> = Vec::new();
        if let Some(overrides) = schema_overrides {
            for (idx, name) in names.iter().enumerate() {
                if let Some(dtype) = overrides.0.get(name) {
                    schema_overrides_by_idx.push((idx, dtype.clone()));
                }
            }
        }
        let mut pydf = Self::finish_from_rows(
            rows,
            infer_schema_length,
            schema.map(|wrap| wrap.0),
            Some(schema_overrides_by_idx),
        )?;
        unsafe {
            for (s, name) in pydf.df.get_columns_mut().iter_mut().zip(&names) {
                s.rename(name);
            }
        }
        let length = names.len();
        if names.into_iter().collect::<PlHashSet<_>>().len() != length {
            let err = PolarsError::Duplicate("duplicate column names found".into());
            Err(PyPolarsErr::Polars(err))?;
        }

        Ok(pydf)
    }

    #[staticmethod]
    pub fn read_dict(py: Python, dict: &PyDict) -> PyResult<Self> {
        let cols = dict
            .into_iter()
            .map(|(key, val)| {
                let name = key.extract::<&str>()?;

                let s = if val.is_instance_of::<PyDict>() {
                    let df = Self::read_dict(py, val.extract::<&PyDict>()?)?;
                    df.df.into_struct(name).into_series()
                } else {
                    let obj = py_modules::SERIES.call1(py, (name, val))?;

                    let pyseries_obj = obj.getattr(py, "_s")?;
                    let pyseries = pyseries_obj.extract::<PySeries>(py)?;
                    pyseries.series
                };
                Ok(s)
            })
            .collect::<PyResult<Vec<_>>>()?;

        let df = DataFrame::new(cols).map_err(PyPolarsErr::from)?;
        Ok(df.into())
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
        batch_size: usize,
        datetime_format: Option<String>,
        date_format: Option<String>,
        time_format: Option<String>,
        float_precision: Option<usize>,
        null_value: Option<String>,
        quote_style: Option<Wrap<QuoteStyle>>,
    ) -> PyResult<()> {
        let null = null_value.unwrap_or_default();

        if let Ok(s) = py_f.extract::<&str>(py) {
            let f = std::fs::File::create(s)?;
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

    #[cfg(feature = "ipc")]
    pub fn write_ipc(
        &mut self,
        py: Python,
        py_f: PyObject,
        compression: Wrap<Option<IpcCompression>>,
    ) -> PyResult<()> {
        if let Ok(s) = py_f.extract::<&str>(py) {
            let f = std::fs::File::create(s)?;
            py.allow_threads(|| {
                IpcWriter::new(f)
                    .with_compression(compression.0)
                    .finish(&mut self.df)
                    .map_err(PyPolarsErr::from)
            })?;
        } else {
            let mut buf = get_file_like(py_f, true)?;

            IpcWriter::new(&mut buf)
                .with_compression(compression.0)
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
        if let Ok(s) = py_f.extract::<&str>(py) {
            let f = std::fs::File::create(s)?;
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

    #[cfg(feature = "object")]
    pub fn row_tuple(&self, idx: i64) -> PyResult<PyObject> {
        let idx = if idx < 0 {
            (self.df.height() as i64 + idx) as usize
        } else {
            idx as usize
        };
        if idx >= self.df.height() {
            return Err(PyPolarsErr::from(polars_err!(oob = idx, self.df.height())).into());
        }
        let out = Python::with_gil(|py| {
            PyTuple::new(
                py,
                self.df.get_columns().iter().map(|s| match s.dtype() {
                    DataType::Object(_, _) => {
                        let obj: Option<&ObjectValue> = s.get_object(idx).map(|any| any.into());
                        obj.to_object(py)
                    },
                    _ => Wrap(s.get(idx).unwrap()).into_py(py),
                }),
            )
            .into_py(py)
        });
        Ok(out)
    }

    #[cfg(feature = "object")]
    pub fn row_tuples(&self) -> PyObject {
        Python::with_gil(|py| {
            let df = &self.df;
            PyList::new(
                py,
                (0..df.height()).map(|idx| {
                    PyTuple::new(
                        py,
                        self.df.get_columns().iter().map(|s| match s.dtype() {
                            DataType::Null => py.None(),
                            DataType::Object(_, _) => {
                                let obj: Option<&ObjectValue> =
                                    s.get_object(idx).map(|any| any.into());
                                obj.to_object(py)
                            },
                            // safety: we are in bounds.
                            _ => unsafe { Wrap(s.get_unchecked(idx)).into_py(py) },
                        }),
                    )
                }),
            )
            .into_py(py)
        })
    }

    pub fn to_numpy(&self, py: Python, order: Wrap<IndexOrder>) -> Option<PyObject> {
        let mut st = None;
        for s in self.df.iter() {
            let dt_i = s.dtype();
            match st {
                None => st = Some(dt_i.clone()),
                Some(ref mut st) => {
                    *st = try_get_supertype(st, dt_i).ok()?;
                },
            }
        }
        let st = st?;

        #[rustfmt::skip]
        let pyarray = match st {
            DataType::UInt32 => self.df.to_ndarray::<UInt32Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            DataType::UInt64 => self.df.to_ndarray::<UInt64Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            DataType::Int32 => self.df.to_ndarray::<Int32Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            DataType::Int64 => self.df.to_ndarray::<Int64Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            DataType::Float32 => self.df.to_ndarray::<Float32Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            DataType::Float64 => self.df.to_ndarray::<Float64Type>(order.0).ok()?.into_pyarray(py).into_py(py),
            _ => return None,
        };
        Some(pyarray)
    }

    #[cfg(feature = "parquet")]
    #[pyo3(signature = (py_f, compression, compression_level, statistics, row_group_size, data_page_size))]
    pub fn write_parquet(
        &mut self,
        py: Python,
        py_f: PyObject,
        compression: &str,
        compression_level: Option<i32>,
        statistics: bool,
        row_group_size: Option<usize>,
        data_page_size: Option<usize>,
    ) -> PyResult<()> {
        let compression = parse_parquet_compression(compression, compression_level)?;

        if let Ok(s) = py_f.extract::<&str>(py) {
            let f = std::fs::File::create(s)?;
            py.allow_threads(|| {
                ParquetWriter::new(f)
                    .with_compression(compression)
                    .with_statistics(statistics)
                    .with_row_group_size(row_group_size)
                    .with_data_page_size(data_page_size)
                    .finish(&mut self.df)
                    .map_err(PyPolarsErr::from)
            })?;
        } else {
            let buf = get_file_like(py_f, true)?;
            ParquetWriter::new(buf)
                .with_compression(compression)
                .with_statistics(statistics)
                .with_row_group_size(row_group_size)
                .with_data_page_size(data_page_size)
                .finish(&mut self.df)
                .map_err(PyPolarsErr::from)?;
        }

        Ok(())
    }

    pub fn to_arrow(&mut self) -> PyResult<Vec<PyObject>> {
        self.df.align_chunks();
        Python::with_gil(|py| {
            let pyarrow = py.import("pyarrow")?;
            let names = self.df.get_column_names();

            let rbs = self
                .df
                .iter_chunks()
                .map(|rb| arrow_interop::to_py::to_py_rb(&rb, &names, py, pyarrow))
                .collect::<PyResult<_>>()?;
            Ok(rbs)
        })
    }

    pub fn to_pandas(&mut self) -> PyResult<Vec<PyObject>> {
        self.df.as_single_chunk_par();
        Python::with_gil(|py| {
            let pyarrow = py.import("pyarrow")?;
            let names = self.df.get_column_names();
            let cat_columns = self
                .df
                .get_columns()
                .iter()
                .enumerate()
                .filter(|(_i, s)| matches!(s.dtype(), DataType::Categorical(_, _)))
                .map(|(i, _)| i)
                .collect::<Vec<_>>();

            let rbs = self
                .df
                .iter_chunks()
                .map(|rb| {
                    let mut rb = rb.into_arrays();
                    for i in &cat_columns {
                        let arr = rb.get_mut(*i).unwrap();
                        let out = polars_core::export::arrow::compute::cast::cast(
                            &**arr,
                            &ArrowDataType::Dictionary(
                                IntegerType::Int64,
                                Box::new(ArrowDataType::LargeUtf8),
                                false,
                            ),
                            CastOptions::default(),
                        )
                        .unwrap();
                        *arr = out;
                    }
                    let rb = ArrowChunk::new(rb);

                    arrow_interop::to_py::to_py_rb(&rb, &names, py, pyarrow)
                })
                .collect::<PyResult<_>>()?;
            Ok(rbs)
        })
    }

    pub fn add(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df + &s.series).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn sub(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df - &s.series).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn div(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df / &s.series).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn mul(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df * &s.series).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn rem(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df % &s.series).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn add_df(&self, s: &Self) -> PyResult<Self> {
        let df = (&self.df + &s.df).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn sub_df(&self, s: &Self) -> PyResult<Self> {
        let df = (&self.df - &s.df).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn div_df(&self, s: &Self) -> PyResult<Self> {
        let df = (&self.df / &s.df).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn mul_df(&self, s: &Self) -> PyResult<Self> {
        let df = (&self.df * &s.df).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn rem_df(&self, s: &Self) -> PyResult<Self> {
        let df = (&self.df % &s.df).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn sample_n(
        &self,
        n: &PySeries,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let df = self
            .df
            .sample_n(&n.series, with_replacement, shuffle, seed)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn sample_frac(
        &self,
        frac: &PySeries,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let df = self
            .df
            .sample_frac(&frac.series, with_replacement, shuffle, seed)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn rechunk(&self) -> Self {
        self.df.agg_chunks().into()
    }

    /// Format `DataFrame` as String
    pub fn as_str(&self) -> String {
        format!("{:?}", self.df)
    }

    pub fn get_columns(&self) -> Vec<PySeries> {
        let cols = self.df.get_columns().to_vec();
        cols.to_pyseries()
    }

    /// Get column names
    pub fn columns(&self) -> Vec<&str> {
        self.df.get_column_names()
    }

    /// set column names
    pub fn set_column_names(&mut self, names: Vec<&str>) -> PyResult<()> {
        self.df
            .set_column_names(&names)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    /// Get datatypes
    pub fn dtypes(&self, py: Python) -> PyObject {
        let iter = self
            .df
            .iter()
            .map(|s| Wrap(s.dtype().clone()).to_object(py));
        PyList::new(py, iter).to_object(py)
    }

    pub fn n_chunks(&self) -> usize {
        self.df.n_chunks()
    }

    pub fn shape(&self) -> (usize, usize) {
        self.df.shape()
    }

    pub fn height(&self) -> usize {
        self.df.height()
    }

    pub fn width(&self) -> usize {
        self.df.width()
    }

    pub fn hstack(&self, columns: Vec<PySeries>) -> PyResult<Self> {
        let columns = columns.to_series();
        let df = self.df.hstack(&columns).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn hstack_mut(&mut self, columns: Vec<PySeries>) -> PyResult<()> {
        let columns = columns.to_series();
        self.df.hstack_mut(&columns).map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn vstack(&self, other: &PyDataFrame) -> PyResult<Self> {
        let df = self.df.vstack(&other.df).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn vstack_mut(&mut self, other: &PyDataFrame) -> PyResult<()> {
        self.df.vstack_mut(&other.df).map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn extend(&mut self, other: &PyDataFrame) -> PyResult<()> {
        self.df.extend(&other.df).map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn drop_in_place(&mut self, name: &str) -> PyResult<PySeries> {
        let s = self.df.drop_in_place(name).map_err(PyPolarsErr::from)?;
        Ok(PySeries { series: s })
    }

    pub fn select_at_idx(&self, idx: usize) -> Option<PySeries> {
        self.df.select_at_idx(idx).map(|s| PySeries::new(s.clone()))
    }

    pub fn get_column_index(&self, name: &str) -> Option<usize> {
        self.df.get_column_index(name)
    }

    pub fn get_column(&self, name: &str) -> PyResult<PySeries> {
        let series = self
            .df
            .column(name)
            .map(|s| PySeries::new(s.clone()))
            .map_err(PyPolarsErr::from)?;
        Ok(series)
    }

    pub fn select(&self, selection: Vec<&str>) -> PyResult<Self> {
        let df = self.df.select(selection).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn gather(&self, indices: Wrap<Vec<IdxSize>>) -> PyResult<Self> {
        let indices = indices.0;
        let indices = IdxCa::from_vec("", indices);
        let df = self.df.take(&indices).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn take_with_series(&self, indices: &PySeries) -> PyResult<Self> {
        let idx = indices.series.idx().map_err(PyPolarsErr::from)?;
        let df = self.df.take(idx).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn replace(&mut self, column: &str, new_col: PySeries) -> PyResult<()> {
        self.df
            .replace(column, new_col.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn replace_column(&mut self, index: usize, new_column: PySeries) -> PyResult<()> {
        self.df
            .replace_column(index, new_column.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn insert_column(&mut self, index: usize, column: PySeries) -> PyResult<()> {
        self.df
            .insert_column(index, column.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn slice(&self, offset: i64, length: Option<usize>) -> Self {
        let df = self
            .df
            .slice(offset, length.unwrap_or_else(|| self.df.height()));
        df.into()
    }

    pub fn head(&self, n: usize) -> Self {
        let df = self.df.head(Some(n));
        PyDataFrame::new(df)
    }

    pub fn tail(&self, n: usize) -> Self {
        let df = self.df.tail(Some(n));
        PyDataFrame::new(df)
    }

    pub fn is_unique(&self) -> PyResult<PySeries> {
        let mask = self.df.is_unique().map_err(PyPolarsErr::from)?;
        Ok(mask.into_series().into())
    }

    pub fn is_duplicated(&self) -> PyResult<PySeries> {
        let mask = self.df.is_duplicated().map_err(PyPolarsErr::from)?;
        Ok(mask.into_series().into())
    }

    pub fn equals(&self, other: &PyDataFrame, null_equal: bool) -> bool {
        if null_equal {
            self.df.equals_missing(&other.df)
        } else {
            self.df.equals(&other.df)
        }
    }

    pub fn with_row_number(&self, name: &str, offset: Option<IdxSize>) -> PyResult<Self> {
        let df = self
            .df
            .with_row_count(name, offset)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn group_by_map_groups(
        &self,
        by: Vec<&str>,
        lambda: PyObject,
        maintain_order: bool,
    ) -> PyResult<Self> {
        let gb = if maintain_order {
            self.df.group_by_stable(&by)
        } else {
            self.df.group_by(&by)
        }
        .map_err(PyPolarsErr::from)?;

        let function = move |df: DataFrame| {
            Python::with_gil(|py| {
                let pypolars = PyModule::import(py, "polars").unwrap();
                let pydf = PyDataFrame::new(df);
                let python_df_wrapper =
                    pypolars.getattr("wrap_df").unwrap().call1((pydf,)).unwrap();

                // Call the lambda and get a python-side DataFrame wrapper.
                let result_df_wrapper = match lambda.call1(py, (python_df_wrapper,)) {
                    Ok(pyobj) => pyobj,
                    Err(e) => panic!("UDF failed: {}", e.value(py)),
                };
                let py_pydf = result_df_wrapper.getattr(py, "_df").expect(
                    "Could not get DataFrame attribute '_df'. Make sure that you return a DataFrame object.",
                );

                let pydf = py_pydf.extract::<PyDataFrame>(py).unwrap();
                Ok(pydf.df)
            })
        };
        // We don't use `py.allow_threads(|| gb.par_apply(..)` because that segfaulted
        // due to code related to Pyo3 or rayon, cannot reproduce it in native polars.
        // So we lose parallelism, but it doesn't really matter because we are GIL bound anyways
        // and this function should not be used in idiomatic polars anyway.
        let df = gb.apply(function).map_err(PyPolarsErr::from)?;

        Ok(df.into())
    }

    pub fn clone(&self) -> Self {
        PyDataFrame::new(self.df.clone())
    }

    pub fn melt(
        &self,
        id_vars: Vec<&str>,
        value_vars: Vec<&str>,
        value_name: Option<&str>,
        variable_name: Option<&str>,
    ) -> PyResult<Self> {
        let args = MeltArgs {
            id_vars: strings_to_smartstrings(id_vars),
            value_vars: strings_to_smartstrings(value_vars),
            value_name: value_name.map(|s| s.into()),
            variable_name: variable_name.map(|s| s.into()),
            streamable: false,
        };

        let df = self.df.melt2(args).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[cfg(feature = "pivot")]
    pub fn pivot_expr(
        &self,
        values: Vec<String>,
        index: Vec<String>,
        columns: Vec<String>,
        maintain_order: bool,
        sort_columns: bool,
        aggregate_expr: Option<PyExpr>,
        separator: Option<&str>,
    ) -> PyResult<Self> {
        let fun = if maintain_order { pivot_stable } else { pivot };
        let agg_expr = aggregate_expr.map(|expr| expr.inner);
        let df = fun(
            &self.df,
            values,
            index,
            columns,
            sort_columns,
            agg_expr,
            separator,
        )
        .map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn partition_by(
        &self,
        by: Vec<String>,
        maintain_order: bool,
        include_key: bool,
    ) -> PyResult<Vec<Self>> {
        let out = if maintain_order {
            self.df.partition_by_stable(by, include_key)
        } else {
            self.df.partition_by(by, include_key)
        }
        .map_err(PyPolarsErr::from)?;

        // SAFETY: PyDataFrame is a repr(transparent) DataFrame.
        Ok(unsafe { std::mem::transmute::<Vec<DataFrame>, Vec<PyDataFrame>>(out) })
    }

    pub fn lazy(&self) -> PyLazyFrame {
        self.df.clone().lazy().into()
    }

    pub fn max_horizontal(&self) -> PyResult<Option<PySeries>> {
        let s = self.df.max_horizontal().map_err(PyPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn min_horizontal(&self) -> PyResult<Option<PySeries>> {
        let s = self.df.min_horizontal().map_err(PyPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn sum_horizontal(&self, ignore_nulls: bool) -> PyResult<Option<PySeries>> {
        let null_strategy = if ignore_nulls {
            NullStrategy::Ignore
        } else {
            NullStrategy::Propagate
        };
        let s = self
            .df
            .sum_horizontal(null_strategy)
            .map_err(PyPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn mean_horizontal(&self, ignore_nulls: bool) -> PyResult<Option<PySeries>> {
        let null_strategy = if ignore_nulls {
            NullStrategy::Ignore
        } else {
            NullStrategy::Propagate
        };
        let s = self
            .df
            .mean_horizontal(null_strategy)
            .map_err(PyPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    #[pyo3(signature = (columns, separator, drop_first=false))]
    pub fn to_dummies(
        &self,
        columns: Option<Vec<String>>,
        separator: Option<&str>,
        drop_first: bool,
    ) -> PyResult<Self> {
        let df = match columns {
            Some(cols) => self.df.columns_to_dummies(
                cols.iter().map(|x| x as &str).collect(),
                separator,
                drop_first,
            ),
            None => self.df.to_dummies(separator, drop_first),
        }
        .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn null_count(&self) -> Self {
        let df = self.df.null_count();
        df.into()
    }

    #[pyo3(signature = (lambda, output_type, inference_size))]
    pub fn map_rows(
        &mut self,
        lambda: &PyAny,
        output_type: Option<Wrap<DataType>>,
        inference_size: usize,
    ) -> PyResult<(PyObject, bool)> {
        Python::with_gil(|py| {
            // needed for series iter
            self.df.as_single_chunk_par();
            let df = &self.df;

            use apply_lambda_with_primitive_out_type as apply;
            #[rustfmt::skip]
            let out = match output_type.map(|dt| dt.0) {
                Some(DataType::Int32) => apply::<Int32Type>(df, py, lambda, 0, None).into_series(),
                Some(DataType::Int64) => apply::<Int64Type>(df, py, lambda, 0, None).into_series(),
                Some(DataType::UInt32) => apply::<UInt32Type>(df, py, lambda, 0, None).into_series(),
                Some(DataType::UInt64) => apply::<UInt64Type>(df, py, lambda, 0, None).into_series(),
                Some(DataType::Float32) => apply::<Float32Type>(df, py, lambda, 0, None).into_series(),
                Some(DataType::Float64) => apply::<Float64Type>(df, py, lambda, 0, None).into_series(),
                Some(DataType::Date) => apply::<Int32Type>(df, py, lambda, 0, None).into_date().into_series(),
                Some(DataType::Datetime(tu, tz)) => apply::<Int64Type>(df, py, lambda, 0, None).into_datetime(tu, tz).into_series(),
                Some(DataType::Boolean) => apply_lambda_with_bool_out_type(df, py, lambda, 0, None).into_series(),
                Some(DataType::String) => apply_lambda_with_string_out_type(df, py, lambda, 0, None).into_series(),
                _ => return apply_lambda_unknown(df, py, lambda, inference_size),
            };

            Ok((PySeries::from(out).into_py(py), false))
        })
    }

    pub fn shrink_to_fit(&mut self) {
        self.df.shrink_to_fit();
    }

    pub fn hash_rows(&mut self, k0: u64, k1: u64, k2: u64, k3: u64) -> PyResult<PySeries> {
        let hb = ahash::RandomState::with_seeds(k0, k1, k2, k3);
        let hash = self.df.hash_rows(Some(hb)).map_err(PyPolarsErr::from)?;
        Ok(hash.into_series().into())
    }

    #[pyo3(signature = (keep_names_as, column_names))]
    pub fn transpose(&self, keep_names_as: Option<&str>, column_names: &PyAny) -> PyResult<Self> {
        let new_col_names = if let Ok(name) = column_names.extract::<Vec<String>>() {
            Some(Either::Right(name))
        } else if let Ok(name) = column_names.extract::<String>() {
            Some(Either::Left(name))
        } else {
            None
        };
        Ok(self
            .df
            .transpose(keep_names_as, new_col_names)
            .map_err(PyPolarsErr::from)?
            .into())
    }
    pub fn upsample(
        &self,
        by: Vec<String>,
        index_column: &str,
        every: &str,
        offset: &str,
        stable: bool,
    ) -> PyResult<Self> {
        let out = if stable {
            self.df.upsample_stable(
                by,
                index_column,
                Duration::parse(every),
                Duration::parse(offset),
            )
        } else {
            self.df.upsample(
                by,
                index_column,
                Duration::parse(every),
                Duration::parse(offset),
            )
        };
        let out = out.map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    pub fn to_struct(&self, name: &str) -> PySeries {
        let s = self.df.clone().into_struct(name);
        s.into_series().into()
    }

    pub fn unnest(&self, columns: Vec<String>) -> PyResult<Self> {
        let df = self.df.unnest(columns).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn clear(&self) -> Self {
        self.df.clear().into()
    }
}
