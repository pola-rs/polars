use std::io::BufWriter;
use std::ops::Deref;

use numpy::IntoPyArray;
use polars::frame::row::{rows_to_schema_supertypes, Row};
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
use polars_core::prelude::QuantileInterpolOptions;
use polars_core::utils::arrow::compute::cast::CastOptions;
use polars_core::utils::try_get_supertype;
#[cfg(feature = "pivot")]
use polars_lazy::frame::pivot::{pivot, pivot_stable};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::apply::dataframe::{
    apply_lambda_unknown, apply_lambda_with_bool_out_type, apply_lambda_with_primitive_out_type,
    apply_lambda_with_utf8_out_type,
};
#[cfg(feature = "parquet")]
use crate::conversion::parse_parquet_compression;
use crate::conversion::{ObjectValue, Wrap};
use crate::error::PyPolarsErr;
use crate::file::{get_either_file, get_file_like, get_mmap_bytes_reader, EitherRustPythonFile};
use crate::lazy::dataframe::PyLazyFrame;
use crate::prelude::dicts_to_rows;
use crate::series::{to_pyseries_collection, to_series_collection, PySeries};
use crate::{arrow_interop, py_modules, PyExpr};

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
        schema_overwrite: Option<Schema>,
    ) -> PyResult<Self> {
        // object builder must be registered.
        #[cfg(feature = "object")]
        crate::object::register_object_builder();

        let schema =
            rows_to_schema_supertypes(&rows, infer_schema_length.map(|n| std::cmp::max(1, n)))
                .map_err(PyPolarsErr::from)?;
        // replace inferred nulls with boolean
        let fields = schema.iter_fields().map(|mut fld| match fld.data_type() {
            DataType::Null => {
                fld.coerce(DataType::Boolean);
                fld
            }
            _ => fld,
        });
        let mut schema = Schema::from(fields);

        if let Some(schema_overwrite) = schema_overwrite {
            for (i, (name, dtype)) in schema_overwrite.into_iter().enumerate() {
                if let Some((name_, dtype_)) = schema.get_index_mut(i) {
                    *name_ = name;

                    // if user sets dtype unknown, we use the inferred datatype
                    if !matches!(dtype, DataType::Unknown) {
                        *dtype_ = dtype;
                    }
                } else {
                    schema.with_column(name, dtype);
                }
            }
        }

        let df = DataFrame::from_rows_and_schema(&rows, &schema).map_err(PyPolarsErr::from)?;
        Ok(df.into())
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
        // used for polars-lazy python node. This takes the dataframe from underneath of you, so
        // don't use this anywhere else.
        let mut df = std::mem::take(&mut self.df);
        let cols = std::mem::take(df.get_columns_mut());
        let (ptr, len, cap) = cols.into_raw_parts();
        (ptr as usize, len, cap)
    }

    #[new]
    pub fn __init__(columns: Vec<PySeries>) -> PyResult<Self> {
        let columns = to_series_collection(columns);
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
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "csv-file")]
    #[pyo3(signature = (
        py_f, infer_schema_length, chunk_size, has_header, ignore_errors, n_rows,
        skip_rows, projection, sep, rechunk, columns, encoding, n_threads, path,
        overwrite_dtype, overwrite_dtype_slice, low_memory, comment_char, quote_char,
        null_values, missing_utf8_is_empty_string, parse_dates, skip_rows_after_header,
        row_count, sample_size, eol_char)
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
        sep: &str,
        rechunk: bool,
        columns: Option<Vec<String>>,
        encoding: Wrap<CsvEncoding>,
        n_threads: Option<usize>,
        path: Option<String>,
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
    ) -> PyResult<Self> {
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

        let mmap_bytes_r = get_mmap_bytes_reader(py_f)?;
        let df = CsvReader::new(mmap_bytes_r)
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
            .with_path(path)
            .with_dtypes(overwrite_dtype.as_ref())
            .with_dtypes_slice(overwrite_dtype_slice.as_deref())
            .low_memory(low_memory)
            .with_null_values(null_values)
            .with_missing_is_null(!missing_utf8_is_empty_string)
            .with_comment_char(comment_char)
            .with_parse_dates(parse_dates)
            .with_quote_char(quote_char)
            .with_end_of_line_char(eol_char)
            .with_skip_rows_after_header(skip_rows_after_header)
            .with_row_count(row_count)
            .sample_size(sample_size)
            .finish()
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    #[staticmethod]
    #[cfg(feature = "parquet")]
    #[pyo3(signature = (py_f, columns, projection, n_rows, parallel, row_count, low_memory))]
    pub fn read_parquet(
        py_f: PyObject,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        n_rows: Option<usize>,
        parallel: Wrap<ParallelStrategy>,
        row_count: Option<(String, IdxSize)>,
        low_memory: bool,
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
                    .finish()
            }
            Rust(f) => ParquetReader::new(f.into_inner())
                .with_projection(projection)
                .with_columns(columns)
                .read_parallel(parallel.0)
                .with_n_rows(n_rows)
                .with_row_count(row_count)
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
    #[pyo3(signature = (py_f, compression))]
    pub fn write_avro(
        &mut self,
        py: Python,
        py_f: PyObject,
        compression: Wrap<Option<AvroCompression>>,
    ) -> PyResult<()> {
        use polars::io::avro::AvroWriter;

        if let Ok(s) = py_f.extract::<&str>(py) {
            let f = std::fs::File::create(s).unwrap();
            AvroWriter::new(f)
                .with_compression(compression.0)
                .finish(&mut self.df)
                .map_err(PyPolarsErr::from)?;
        } else {
            let mut buf = get_file_like(py_f, true)?;
            AvroWriter::new(&mut buf)
                .with_compression(compression.0)
                .finish(&mut self.df)
                .map_err(PyPolarsErr::from)?;
        }

        Ok(())
    }

    #[staticmethod]
    #[cfg(feature = "json")]
    pub fn read_json(py_f: &PyAny, json_lines: bool) -> PyResult<Self> {
        let mmap_bytes_r = get_mmap_bytes_reader(py_f)?;
        if json_lines {
            let out = JsonReader::new(mmap_bytes_r)
                .with_json_format(JsonFormat::JsonLines)
                .finish()
                .map_err(|e| PyPolarsErr::Other(format!("{e:?}")))?;
            Ok(out.into())
        } else {
            // memmap the file first
            let mmap_bytes_r = get_mmap_bytes_reader(py_f)?;
            let mmap_read: ReaderBytes = (&mmap_bytes_r).into();
            let bytes = mmap_read.deref();

            // Happy path is our column oriented json as that is most performant
            // on failure we try
            match serde_json::from_slice::<DataFrame>(bytes) {
                Ok(df) => Ok(df.into()),
                // try arrow json reader instead
                // this is row oriented
                Err(_) => {
                    let out = JsonReader::new(mmap_bytes_r)
                        .with_json_format(JsonFormat::Json)
                        .finish()
                        .map_err(|e| PyPolarsErr::Other(format!("{e:?}")))?;
                    Ok(out.into())
                }
            }
        }
    }

    #[staticmethod]
    #[cfg(feature = "json")]
    pub fn read_ndjson(py_f: &PyAny) -> PyResult<Self> {
        let mmap_bytes_r = get_mmap_bytes_reader(py_f)?;

        let out = JsonReader::new(mmap_bytes_r)
            .with_json_format(JsonFormat::JsonLines)
            .finish()
            .map_err(|e| PyPolarsErr::Other(format!("{e:?}")))?;
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
                .map_err(|e| PolarsError::ComputeError(format!("{e:?}").into())),
            (false, _) => serde_json::to_writer(file, &self.df)
                .map_err(|e| PolarsError::ComputeError(format!("{e:?}").into())),
        };
        r.map_err(|e| PyPolarsErr::Other(format!("{e:?}")))?;
        Ok(())
    }

    #[cfg(feature = "json")]
    pub fn write_ndjson(&mut self, py_f: PyObject) -> PyResult<()> {
        let file = BufWriter::new(get_file_like(py_f, true)?);

        let r = JsonWriter::new(file)
            .with_json_format(JsonFormat::JsonLines)
            .finish(&mut self.df);

        r.map_err(|e| PyPolarsErr::Other(format!("{e:?}")))?;
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
        schema_overwrite: Option<Wrap<Schema>>,
    ) -> PyResult<Self> {
        // safety:
        // wrap is transparent
        let rows: Vec<Row> = unsafe { std::mem::transmute(rows) };
        Self::finish_from_rows(
            rows,
            infer_schema_length,
            schema_overwrite.map(|wrap| wrap.0),
        )
    }

    #[staticmethod]
    pub fn read_dicts(
        dicts: &PyAny,
        infer_schema_length: Option<usize>,
        schema_overwrite: Option<Wrap<Schema>>,
    ) -> PyResult<Self> {
        // if given, read dict fields in schema order
        let mut schema_columns = PlIndexSet::new();
        if let Some(schema) = &schema_overwrite {
            schema_columns.extend(schema.0.iter_names().map(|n| n.to_string()))
        }
        let (rows, names) = dicts_to_rows(dicts, infer_schema_length, schema_columns)?;

        let mut pydf = Self::finish_from_rows(
            rows,
            infer_schema_length,
            schema_overwrite.map(|wrap| wrap.0),
        )?;

        pydf.df
            .get_columns_mut()
            .iter_mut()
            .zip(&names)
            .for_each(|(s, name)| {
                s.rename(name);
            });
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

                let s = if val.is_instance_of::<PyDict>()? {
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

    #[allow(clippy::too_many_arguments)]
    pub fn write_csv(
        &mut self,
        py: Python,
        py_f: PyObject,
        has_header: bool,
        sep: u8,
        quote: u8,
        batch_size: usize,
        datetime_format: Option<String>,
        date_format: Option<String>,
        time_format: Option<String>,
        float_precision: Option<usize>,
        null_value: Option<String>,
    ) -> PyResult<()> {
        let null = null_value.unwrap_or_default();

        if let Ok(s) = py_f.extract::<&str>(py) {
            py.allow_threads(|| {
                let f = std::fs::File::create(s).unwrap();
                // no need for a buffered writer, because the csv writer does internal buffering
                CsvWriter::new(f)
                    .has_header(has_header)
                    .with_delimiter(sep)
                    .with_quoting_char(quote)
                    .with_batch_size(batch_size)
                    .with_datetime_format(datetime_format)
                    .with_date_format(date_format)
                    .with_time_format(time_format)
                    .with_float_precision(float_precision)
                    .with_null_value(null)
                    .finish(&mut self.df)
                    .map_err(PyPolarsErr::from)
            })?;
        } else {
            let mut buf = get_file_like(py_f, true)?;
            CsvWriter::new(&mut buf)
                .has_header(has_header)
                .with_delimiter(sep)
                .with_quoting_char(quote)
                .with_batch_size(batch_size)
                .with_datetime_format(datetime_format)
                .with_date_format(date_format)
                .with_time_format(time_format)
                .with_float_precision(float_precision)
                .with_null_value(null)
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
            py.allow_threads(|| {
                let f = std::fs::File::create(s).unwrap();
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

    #[cfg(feature = "object")]
    pub fn row_tuple(&self, idx: i64) -> PyResult<PyObject> {
        let idx = if idx < 0 {
            (self.df.height() as i64 + idx) as usize
        } else {
            idx as usize
        };
        if idx >= self.df.height() {
            Err(PolarsError::ComputeError("Index out of bounds.".into()))
                .map_err(PyPolarsErr::from)?;
        }

        let out = Python::with_gil(|py| {
            PyTuple::new(
                py,
                self.df.get_columns().iter().map(|s| match s.dtype() {
                    DataType::Object(_) => {
                        let obj: Option<&ObjectValue> = s.get_object(idx).map(|any| any.into());
                        obj.to_object(py)
                    }
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
                            DataType::Object(_) => {
                                let obj: Option<&ObjectValue> =
                                    s.get_object(idx).map(|any| any.into());
                                obj.to_object(py)
                            }
                            // safety: we are in bounds.
                            _ => unsafe { Wrap(s.get_unchecked(idx)).into_py(py) },
                        }),
                    )
                }),
            )
            .into_py(py)
        })
    }

    pub fn to_numpy(&self, py: Python) -> Option<PyObject> {
        let mut st = None;
        for s in self.df.iter() {
            let dt_i = s.dtype();
            match st {
                None => st = Some(dt_i.clone()),
                Some(ref mut st) => {
                    *st = try_get_supertype(st, dt_i).ok()?;
                }
            }
        }
        let st = st?;

        match st {
            DataType::UInt32 => self
                .df
                .to_ndarray::<UInt32Type>()
                .ok()
                .map(|arr| arr.into_pyarray(py).into_py(py)),
            DataType::UInt64 => self
                .df
                .to_ndarray::<UInt64Type>()
                .ok()
                .map(|arr| arr.into_pyarray(py).into_py(py)),
            DataType::Int32 => self
                .df
                .to_ndarray::<Int32Type>()
                .ok()
                .map(|arr| arr.into_pyarray(py).into_py(py)),
            DataType::Int64 => self
                .df
                .to_ndarray::<Int64Type>()
                .ok()
                .map(|arr| arr.into_pyarray(py).into_py(py)),
            DataType::Float32 => self
                .df
                .to_ndarray::<Float32Type>()
                .ok()
                .map(|arr| arr.into_pyarray(py).into_py(py)),
            DataType::Float64 => self
                .df
                .to_ndarray::<Float64Type>()
                .ok()
                .map(|arr| arr.into_pyarray(py).into_py(py)),
            _ => None,
        }
    }

    #[cfg(feature = "parquet")]
    #[pyo3(signature = (py_f, compression, compression_level, statistics, row_group_size))]
    pub fn write_parquet(
        &mut self,
        py: Python,
        py_f: PyObject,
        compression: &str,
        compression_level: Option<i32>,
        statistics: bool,
        row_group_size: Option<usize>,
    ) -> PyResult<()> {
        let compression = parse_parquet_compression(compression, compression_level)?;

        if let Ok(s) = py_f.extract::<&str>(py) {
            let f = std::fs::File::create(s).unwrap();
            py.allow_threads(|| {
                ParquetWriter::new(f)
                    .with_compression(compression)
                    .with_statistics(statistics)
                    .with_row_group_size(row_group_size)
                    .finish(&mut self.df)
                    .map_err(PyPolarsErr::from)
            })?;
        } else {
            let buf = get_file_like(py_f, true)?;
            ParquetWriter::new(buf)
                .with_compression(compression)
                .with_statistics(statistics)
                .with_row_group_size(row_group_size)
                .finish(&mut self.df)
                .map_err(PyPolarsErr::from)?;
        }

        Ok(())
    }

    pub fn to_arrow(&mut self) -> PyResult<Vec<PyObject>> {
        self.df.rechunk();
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
                .filter(|(_i, s)| matches!(s.dtype(), DataType::Categorical(_)))
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
        n: usize,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let df = self
            .df
            .sample_n(n, with_replacement, shuffle, seed)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn sample_frac(
        &self,
        frac: f64,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let df = self
            .df
            .sample_frac(frac, with_replacement, shuffle, seed)
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
        let cols = self.df.get_columns().clone();
        to_pyseries_collection(cols)
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

    pub fn hstack_mut(&mut self, columns: Vec<PySeries>) -> PyResult<()> {
        let columns = to_series_collection(columns);
        self.df.hstack_mut(&columns).map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn hstack(&self, columns: Vec<PySeries>) -> PyResult<Self> {
        let columns = to_series_collection(columns);
        let df = self.df.hstack(&columns).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn extend(&mut self, df: &PyDataFrame) -> PyResult<()> {
        self.df.extend(&df.df).map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn vstack_mut(&mut self, df: &PyDataFrame) -> PyResult<()> {
        self.df.vstack_mut(&df.df).map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn vstack(&mut self, df: &PyDataFrame) -> PyResult<Self> {
        let df = self.df.vstack(&df.df).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn drop_in_place(&mut self, name: &str) -> PyResult<PySeries> {
        let s = self.df.drop_in_place(name).map_err(PyPolarsErr::from)?;
        Ok(PySeries { series: s })
    }

    pub fn drop_nulls(&self, subset: Option<Vec<String>>) -> PyResult<Self> {
        let df = self
            .df
            .drop_nulls(subset.as_ref().map(|s| s.as_ref()))
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn drop(&self, name: &str) -> PyResult<Self> {
        let df = self.df.drop(name).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn select_at_idx(&self, idx: usize) -> Option<PySeries> {
        self.df.select_at_idx(idx).map(|s| PySeries::new(s.clone()))
    }

    pub fn find_idx_by_name(&self, name: &str) -> Option<usize> {
        self.df.find_idx_by_name(name)
    }

    pub fn column(&self, name: &str) -> PyResult<PySeries> {
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

    pub fn take(&self, indices: Wrap<Vec<IdxSize>>) -> PyResult<Self> {
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

    pub fn sort(&self, by_column: &str, reverse: bool, nulls_last: bool) -> PyResult<Self> {
        let df = self
            .df
            .sort_with_options(
                by_column,
                SortOptions {
                    descending: reverse,
                    nulls_last,
                    multithreaded: true,
                },
            )
            .map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn replace(&mut self, column: &str, new_col: PySeries) -> PyResult<()> {
        self.df
            .replace(column, new_col.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn replace_at_idx(&mut self, index: usize, new_col: PySeries) -> PyResult<()> {
        self.df
            .replace_at_idx(index, new_col.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn insert_at_idx(&mut self, index: usize, new_col: PySeries) -> PyResult<()> {
        self.df
            .insert_at_idx(index, new_col.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn slice(&self, offset: usize, length: Option<usize>) -> Self {
        let df = self
            .df
            .slice(offset as i64, length.unwrap_or_else(|| self.df.height()));
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

    pub fn frame_equal(&self, other: &PyDataFrame, null_equal: bool) -> bool {
        if null_equal {
            self.df.frame_equal_missing(&other.df)
        } else {
            self.df.frame_equal(&other.df)
        }
    }

    pub fn with_row_count(&self, name: &str, offset: Option<IdxSize>) -> PyResult<Self> {
        let df = self
            .df
            .with_row_count(name, offset)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn groupby_apply(&self, by: Vec<&str>, lambda: PyObject) -> PyResult<Self> {
        let gb = self.df.groupby(&by).map_err(PyPolarsErr::from)?;
        let function = move |df: DataFrame| {
            Python::with_gil(|py| {
                // get the pypolars module
                let pypolars = PyModule::import(py, "polars").unwrap();

                // create a PyDataFrame struct/object for Python
                let pydf = PyDataFrame::new(df);

                // Wrap this PySeries object in the python side DataFrame wrapper
                let python_df_wrapper =
                    pypolars.getattr("wrap_df").unwrap().call1((pydf,)).unwrap();

                // call the lambda and get a python side DataFrame wrapper
                let result_df_wrapper = match lambda.call1(py, (python_df_wrapper,)) {
                    Ok(pyobj) => pyobj,
                    Err(e) => panic!("UDF failed: {}", e.value(py)),
                };
                // unpack the wrapper in a PyDataFrame
                let py_pydf = result_df_wrapper.getattr(py, "_df").expect(
                "Could net get DataFrame attribute '_df'. Make sure that you return a DataFrame object.",
            );
                // Downcast to Rust
                let pydf = py_pydf.extract::<PyDataFrame>(py).unwrap();
                // Finally get the actual DataFrame
                Ok(pydf.df)
            })
        };
        // We don't use `py.allow_threads(|| gb.par_apply(..)` because that segfaulted
        // due to code related to Pyo3 or rayon, cannot reproduce it in native polars
        // so we lose parallelism, but it doesn't really matter because we are GIL bound anyways
        // and this function should not be used in idiomatic polars anyway.
        let df = gb.apply(function).map_err(PyPolarsErr::from)?;

        Ok(df.into())
    }

    pub fn clone(&self) -> Self {
        PyDataFrame::new(self.df.clone())
    }

    pub fn melt(
        &self,
        id_vars: Vec<String>,
        value_vars: Vec<String>,
        value_name: Option<String>,
        variable_name: Option<String>,
    ) -> PyResult<Self> {
        let args = MeltArgs {
            id_vars,
            value_vars,
            value_name,
            variable_name,
        };

        let df = self.df.melt2(args).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[cfg(feature = "pivot")]
    #[allow(clippy::too_many_arguments)]
    pub fn pivot_expr(
        &self,
        values: Vec<String>,
        index: Vec<String>,
        columns: Vec<String>,
        aggregate_expr: PyExpr,
        maintain_order: bool,
        sort_columns: bool,
        separator: Option<&str>,
    ) -> PyResult<Self> {
        let fun = match maintain_order {
            true => pivot_stable,
            false => pivot,
        };
        let df = fun(
            &self.df,
            values,
            index,
            columns,
            aggregate_expr.inner,
            sort_columns,
            separator,
        )
        .map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn partition_by(&self, groups: Vec<String>, stable: bool) -> PyResult<Vec<Self>> {
        let out = if stable {
            self.df.partition_by_stable(groups)
        } else {
            self.df.partition_by(groups)
        }
        .map_err(PyPolarsErr::from)?;
        // Safety:
        // Repr mem layout
        Ok(unsafe { std::mem::transmute::<Vec<DataFrame>, Vec<PyDataFrame>>(out) })
    }

    pub fn shift(&self, periods: i64) -> Self {
        self.df.shift(periods).into()
    }

    #[pyo3(signature = (maintain_order, subset, keep))]
    pub fn unique(
        &self,
        py: Python,
        maintain_order: bool,
        subset: Option<Vec<String>>,
        keep: Wrap<UniqueKeepStrategy>,
    ) -> PyResult<Self> {
        let df = py.allow_threads(|| {
            let subset = subset.as_ref().map(|v| v.as_ref());
            match maintain_order {
                true => self.df.unique_stable(subset, keep.0),
                false => self.df.unique(subset, keep.0),
            }
            .map_err(PyPolarsErr::from)
        })?;
        Ok(df.into())
    }

    pub fn lazy(&self) -> PyLazyFrame {
        self.df.clone().lazy().into()
    }

    pub fn max(&self) -> Self {
        self.df.max().into()
    }

    pub fn min(&self) -> Self {
        self.df.min().into()
    }

    pub fn sum(&self) -> Self {
        self.df.sum().into()
    }

    pub fn mean(&self) -> Self {
        self.df.mean().into()
    }

    pub fn std(&self, ddof: u8) -> Self {
        self.df.std(ddof).into()
    }

    pub fn var(&self, ddof: u8) -> Self {
        self.df.var(ddof).into()
    }

    pub fn median(&self) -> Self {
        self.df.median().into()
    }

    pub fn hmean(&self, null_strategy: Wrap<NullStrategy>) -> PyResult<Option<PySeries>> {
        let s = self.df.hmean(null_strategy.0).map_err(PyPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn hmax(&self) -> PyResult<Option<PySeries>> {
        let s = self.df.hmax().map_err(PyPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn hmin(&self) -> PyResult<Option<PySeries>> {
        let s = self.df.hmin().map_err(PyPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn hsum(&self, null_strategy: Wrap<NullStrategy>) -> PyResult<Option<PySeries>> {
        let s = self.df.hsum(null_strategy.0).map_err(PyPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn quantile(
        &self,
        quantile: f64,
        interpolation: Wrap<QuantileInterpolOptions>,
    ) -> PyResult<Self> {
        let df = self
            .df
            .quantile(quantile, interpolation.0)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn to_dummies(
        &self,
        columns: Option<Vec<String>>,
        separator: Option<&str>,
    ) -> PyResult<Self> {
        let df = match columns {
            Some(cols) => self
                .df
                .columns_to_dummies(cols.iter().map(|x| x as &str).collect(), separator),
            None => self.df.to_dummies(separator),
        }
        .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn null_count(&self) -> Self {
        let df = self.df.null_count();
        df.into()
    }

    #[pyo3(signature = (lambda, output_type, inference_size))]
    pub fn apply(
        &self,
        lambda: &PyAny,
        output_type: Option<Wrap<DataType>>,
        inference_size: usize,
    ) -> PyResult<(PyObject, bool)> {
        Python::with_gil(|py| {
            let df = &self.df;

            let output_type = output_type.map(|dt| dt.0);
            let out = match output_type {
                Some(DataType::Int32) => {
                    apply_lambda_with_primitive_out_type::<Int32Type>(df, py, lambda, 0, None)
                        .into_series()
                }
                Some(DataType::Int64) => {
                    apply_lambda_with_primitive_out_type::<Int64Type>(df, py, lambda, 0, None)
                        .into_series()
                }
                Some(DataType::UInt32) => {
                    apply_lambda_with_primitive_out_type::<UInt32Type>(df, py, lambda, 0, None)
                        .into_series()
                }
                Some(DataType::UInt64) => {
                    apply_lambda_with_primitive_out_type::<UInt64Type>(df, py, lambda, 0, None)
                        .into_series()
                }
                Some(DataType::Float32) => {
                    apply_lambda_with_primitive_out_type::<Float32Type>(df, py, lambda, 0, None)
                        .into_series()
                }
                Some(DataType::Float64) => {
                    apply_lambda_with_primitive_out_type::<Float64Type>(df, py, lambda, 0, None)
                        .into_series()
                }
                Some(DataType::Boolean) => {
                    apply_lambda_with_bool_out_type(df, py, lambda, 0, None).into_series()
                }
                Some(DataType::Date) => {
                    apply_lambda_with_primitive_out_type::<Int32Type>(df, py, lambda, 0, None)
                        .into_date()
                        .into_series()
                }
                Some(DataType::Datetime(tu, tz)) => {
                    apply_lambda_with_primitive_out_type::<Int64Type>(df, py, lambda, 0, None)
                        .into_datetime(tu, tz)
                        .into_series()
                }
                Some(DataType::Utf8) => {
                    apply_lambda_with_utf8_out_type(df, py, lambda, 0, None).into_series()
                }
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

    pub fn transpose(&self, include_header: bool, names: &str) -> PyResult<Self> {
        let mut df = self.df.transpose().map_err(PyPolarsErr::from)?;
        if include_header {
            let s = Utf8Chunked::from_iter_values(
                names,
                self.df.get_columns().iter().map(|s| s.name()),
            )
            .into_series();
            df.insert_at_idx(0, s).unwrap();
        }
        Ok(df.into())
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

    pub fn unnest(&self, names: Vec<String>) -> PyResult<Self> {
        let df = self.df.unnest(names).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }
}
