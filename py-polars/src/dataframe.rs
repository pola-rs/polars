use numpy::IntoPyArray;
use pyo3::types::{PyList, PyTuple};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::io::{BufReader, Cursor, Read};

use polars::frame::groupby::GroupBy;
use polars::prelude::*;

use crate::apply::dataframe::{
    apply_lambda_unknown, apply_lambda_with_bool_out_type, apply_lambda_with_primitive_out_type,
    apply_lambda_with_utf8_out_type,
};
use crate::conversion::{ObjectValue, Wrap};
use crate::file::get_mmap_bytes_reader;
use crate::lazy::dataframe::PyLazyFrame;
use crate::prelude::{dicts_to_rows, str_to_null_strategy};
use crate::{
    arrow_interop,
    error::PyPolarsErr,
    file::{get_either_file, get_file_like, EitherRustPythonFile},
    series::{to_pyseries_collection, to_series_collection, PySeries},
};
use polars::frame::row::{rows_to_schema, Row};
use polars::io::RowCount;
use polars_core::export::arrow::datatypes::IntegerType;
use polars_core::frame::groupby::PivotAgg;
use polars_core::frame::ArrowChunk;
use polars_core::prelude::QuantileInterpolOptions;
use polars_core::utils::arrow::compute::cast::CastOptions;
use polars_core::utils::get_supertype;

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

    fn finish_from_rows(rows: Vec<Row>) -> PyResult<Self> {
        // replace inferred nulls with boolean
        let schema = rows_to_schema(&rows);
        let fields = schema.iter_fields().map(|mut fld| match fld.data_type() {
            DataType::Null => {
                fld.coerce(DataType::Boolean);
                fld
            }
            _ => fld,
        });
        let schema = Schema::from(fields);

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
    #[new]
    pub fn __init__(columns: Vec<PySeries>) -> PyResult<Self> {
        let columns = to_series_collection(columns);
        let df = DataFrame::new(columns).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
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
        encoding: &str,
        n_threads: Option<usize>,
        path: Option<String>,
        overwrite_dtype: Option<Vec<(&str, Wrap<DataType>)>>,
        overwrite_dtype_slice: Option<Vec<Wrap<DataType>>>,
        low_memory: bool,
        comment_char: Option<&str>,
        quote_char: Option<&str>,
        null_values: Option<Wrap<NullValues>>,
        parse_dates: bool,
        skip_rows_after_header: usize,
        row_count: Option<(String, u32)>,
    ) -> PyResult<Self> {
        let null_values = null_values.map(|w| w.0);
        let comment_char = comment_char.map(|s| s.as_bytes()[0]);

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
        let encoding = match encoding {
            "utf8" => CsvEncoding::Utf8,
            "utf8-lossy" => CsvEncoding::LossyUtf8,
            e => {
                return Err(
                    PyPolarsErr::Other(format!("encoding not {} not implemented.", e)).into(),
                )
            }
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
            .with_ignore_parser_errors(ignore_errors)
            .with_projection(projection)
            .with_rechunk(rechunk)
            .with_chunk_size(chunk_size)
            .with_encoding(encoding)
            .with_columns(columns)
            .with_n_threads(n_threads)
            .with_path(path)
            .with_dtypes(overwrite_dtype.as_ref())
            .with_dtypes_slice(overwrite_dtype_slice.as_deref())
            .low_memory(low_memory)
            .with_comment_char(comment_char)
            .with_null_values(null_values)
            .with_parse_dates(parse_dates)
            .with_quote_char(quote_char)
            .with_skip_rows_after_header(skip_rows_after_header)
            .with_row_count(row_count)
            .finish()
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    #[staticmethod]
    #[cfg(feature = "parquet")]
    pub fn read_parquet(
        py_f: PyObject,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        n_rows: Option<usize>,
        parallel: bool,
        row_count: Option<(String, u32)>,
    ) -> PyResult<Self> {
        use EitherRustPythonFile::*;

        let row_count = row_count.map(|(name, offset)| RowCount { name, offset });
        let result = match get_either_file(py_f, false)? {
            Py(f) => {
                let buf = f.as_buffer();
                ParquetReader::new(buf)
                    .with_projection(projection)
                    .with_columns(columns)
                    .read_parallel(parallel)
                    .with_n_rows(n_rows)
                    .with_row_count(row_count)
                    .finish()
            }
            Rust(f) => ParquetReader::new(f)
                .with_projection(projection)
                .with_columns(columns)
                .read_parallel(parallel)
                .with_n_rows(n_rows)
                .with_row_count(row_count)
                .finish(),
        };
        let df = result.map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    #[cfg(feature = "ipc")]
    pub fn read_ipc(
        py_f: PyObject,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        n_rows: Option<usize>,
        row_count: Option<(String, u32)>,
    ) -> PyResult<Self> {
        let row_count = row_count.map(|(name, offset)| RowCount { name, offset });
        let file = get_file_like(py_f, false)?;
        let df = IpcReader::new(file)
            .with_projection(projection)
            .with_columns(columns)
            .with_n_rows(n_rows)
            .with_row_count(row_count)
            .finish()
            .map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    #[cfg(feature = "avro")]
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
    pub fn to_avro(&mut self, py_f: PyObject, compression: &str) -> PyResult<()> {
        use polars::io::avro::{AvroCompression, AvroWriter};
        let compression = match compression {
            "uncompressed" => None,
            "snappy" => Some(AvroCompression::Snappy),
            "deflate" => Some(AvroCompression::Deflate),
            s => return Err(PyPolarsErr::Other(format!("compression {} not supported", s)).into()),
        };

        let mut buf = get_file_like(py_f, true)?;
        AvroWriter::new(&mut buf)
            .with_compression(compression)
            .finish(&mut self.df)
            .map_err(PyPolarsErr::from)?;

        Ok(())
    }

    #[staticmethod]
    #[cfg(feature = "json")]
    pub fn read_json(py_f: PyObject, json_lines: bool) -> PyResult<Self> {
        if json_lines {
            let f = get_file_like(py_f, false)?;
            let f = BufReader::new(f);
            let out = JsonReader::new(f)
                .with_json_format(JsonFormat::JsonLines)
                .finish()
                .map_err(|e| PyPolarsErr::Other(format!("{:?}", e)))?;
            Ok(out.into())
        } else {
            // it is faster to first read to memory and then parse: https://github.com/serde-rs/json/issues/160
            // so don't bother with files.
            let mut json = String::new();
            let _ = get_file_like(py_f, false)?
                .read_to_string(&mut json)
                .unwrap();

            // Happy path is our column oriented json as that is fasted
            // on failure we try
            match serde_json::from_str::<DataFrame>(&json) {
                Ok(df) => Ok(df.into()),
                // try arrow json reader instead
                Err(_) => {
                    let f = Cursor::new(json);

                    let out = JsonReader::new(f)
                        .with_json_format(JsonFormat::Json)
                        .finish()
                        .map_err(|e| PyPolarsErr::Other(format!("{:?}", e)))?;
                    Ok(out.into())
                }
            }
        }
    }

    #[cfg(feature = "json")]
    pub fn to_json(
        &mut self,
        py_f: PyObject,
        pretty: bool,
        row_oriented: bool,
        json_lines: bool,
    ) -> PyResult<()> {
        let file = get_file_like(py_f, true)?;

        let r = match (pretty, row_oriented, json_lines) {
            (_, true, true) => panic!("{}", "only one of {row_oriented, json_lines} should be set"),
            (_, _, true) => JsonWriter::new(file)
                .with_json_format(JsonFormat::JsonLines)
                .finish(&mut self.df),
            (_, true, false) => JsonWriter::new(file)
                .with_json_format(JsonFormat::Json)
                .finish(&mut self.df),
            (true, _, _) => serde_json::to_writer_pretty(file, &self.df)
                .map_err(|e| PolarsError::ComputeError(format!("{:?}", e).into())),
            (false, _, _) => serde_json::to_writer(file, &self.df)
                .map_err(|e| PolarsError::ComputeError(format!("{:?}", e).into())),
        };
        r.map_err(|e| PyPolarsErr::Other(format!("{:?}", e)))?;
        Ok(())
    }

    #[staticmethod]
    pub fn from_arrow_record_batches(rb: Vec<&PyAny>) -> PyResult<Self> {
        let df = arrow_interop::to_rust::to_rust_df(&rb)?;
        Ok(Self::from(df))
    }

    // somehow from_rows did not work
    #[staticmethod]
    pub fn read_rows(rows: Vec<Wrap<Row>>) -> PyResult<Self> {
        // safety:
        // wrap is transparent
        let rows: Vec<Row> = unsafe { std::mem::transmute(rows) };
        Self::finish_from_rows(rows)
    }

    #[staticmethod]
    pub fn read_dicts(dicts: &PyAny) -> PyResult<Self> {
        let (rows, names) = dicts_to_rows(dicts)?;
        let mut pydf = Self::finish_from_rows(rows)?;
        pydf.df
            .set_column_names(&names)
            .map_err(PyPolarsErr::from)?;
        Ok(pydf)
    }

    pub fn to_csv(&mut self, py_f: PyObject, has_header: bool, sep: u8) -> PyResult<()> {
        let mut buf = get_file_like(py_f, true)?;
        CsvWriter::new(&mut buf)
            .has_header(has_header)
            .with_delimiter(sep)
            .finish(&mut self.df)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    #[cfg(feature = "ipc")]
    pub fn to_ipc(&mut self, py_f: PyObject, compression: &str) -> PyResult<()> {
        let compression = match compression {
            "uncompressed" => None,
            "lz4" => Some(IpcCompression::LZ4),
            "zstd" => Some(IpcCompression::ZSTD),
            s => return Err(PyPolarsErr::Other(format!("compression {} not supported", s)).into()),
        };
        let mut buf = get_file_like(py_f, true)?;

        IpcWriter::new(&mut buf)
            .with_compression(compression)
            .finish(&mut self.df)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn row_tuple(&self, idx: i64) -> PyObject {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let idx = if idx < 0 {
            (self.df.height() as i64 + idx) as usize
        } else {
            idx as usize
        };
        PyTuple::new(
            py,
            self.df.get_columns().iter().map(|s| match s.dtype() {
                DataType::Object(_) => {
                    let obj: Option<&ObjectValue> = s.get_object(idx).map(|any| any.into());
                    obj.to_object(py)
                }
                _ => Wrap(s.get(idx)).into_py(py),
            }),
        )
        .into_py(py)
    }

    pub fn row_tuples(&self) -> PyObject {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let df = &self.df;
        PyList::new(
            py,
            (0..df.height()).map(|idx| {
                PyTuple::new(
                    py,
                    self.df.get_columns().iter().map(|s| match s.dtype() {
                        DataType::Object(_) => {
                            let obj: Option<&ObjectValue> = s.get_object(idx).map(|any| any.into());
                            obj.to_object(py)
                        }
                        _ => Wrap(s.get(idx)).into_py(py),
                    }),
                )
            }),
        )
        .into_py(py)
    }

    pub fn to_numpy(&self, py: Python) -> Option<PyObject> {
        let mut st = DataType::Int8;
        for s in self.df.iter() {
            let dt_i = s.dtype();
            st = get_supertype(&st, dt_i).ok()?;
        }

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
    pub fn to_parquet(
        &mut self,
        py_f: PyObject,
        compression: &str,
        statistics: bool,
    ) -> PyResult<()> {
        let compression = match compression {
            "uncompressed" => ParquetCompression::Uncompressed,
            "snappy" => ParquetCompression::Snappy,
            "gzip" => ParquetCompression::Gzip,
            "lzo" => ParquetCompression::Lzo,
            "brotli" => ParquetCompression::Brotli,
            "lz4" => ParquetCompression::Lz4,
            "zstd" => ParquetCompression::Zstd,
            s => return Err(PyPolarsErr::Other(format!("compression {} not supported", s)).into()),
        };
        let buf = get_file_like(py_f, true)?;

        ParquetWriter::new(buf)
            .with_compression(compression)
            .with_statistics(statistics)
            .finish(&mut self.df)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn to_arrow(&mut self) -> PyResult<Vec<PyObject>> {
        self.df.rechunk();
        let gil = Python::acquire_gil();
        let py = gil.python();
        let pyarrow = py.import("pyarrow")?;
        let names = self.df.get_column_names();

        let rbs = self
            .df
            .iter_chunks()
            .map(|rb| arrow_interop::to_py::to_py_rb(&rb, &names, py, pyarrow))
            .collect::<PyResult<_>>()?;
        Ok(rbs)
    }

    pub fn to_pandas(&mut self) -> PyResult<Vec<PyObject>> {
        self.df.rechunk();
        let gil = Python::acquire_gil();
        let py = gil.python();
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

        use polars_core::export::arrow::array::ArrayRef;
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
                    let out = Arc::from(out) as ArrayRef;
                    *arr = out;
                }
                let rb = ArrowChunk::new(rb);

                arrow_interop::to_py::to_py_rb(&rb, &names, py, pyarrow)
            })
            .collect::<PyResult<_>>()?;
        Ok(rbs)
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

    pub fn sample_n(&self, n: usize, with_replacement: bool, seed: u64) -> PyResult<Self> {
        let df = self
            .df
            .sample_n(n, with_replacement, seed)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn sample_frac(&self, frac: f64, with_replacement: bool, seed: u64) -> PyResult<Self> {
        let df = self
            .df
            .sample_frac(frac, with_replacement, seed)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn rechunk(&mut self) -> Self {
        self.df.agg_chunks().into()
    }

    /// Format `DataFrame` as String
    pub fn as_str(&self) -> String {
        format!("{:?}", self.df)
    }

    pub fn fill_null(&self, strategy: &str) -> PyResult<Self> {
        let strat = match strategy {
            "backward" => FillNullStrategy::Backward,
            "forward" => FillNullStrategy::Forward,
            "min" => FillNullStrategy::Min,
            "max" => FillNullStrategy::Max,
            "mean" => FillNullStrategy::Mean,
            "one" => FillNullStrategy::One,
            "zero" => FillNullStrategy::Zero,
            s => return Err(PyPolarsErr::Other(format!("Strategy {} not supported", s)).into()),
        };
        let df = self.df.fill_null(strat).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn join(
        &self,
        other: &PyDataFrame,
        left_on: Vec<&str>,
        right_on: Vec<&str>,
        how: &str,
        suffix: String,
    ) -> PyResult<Self> {
        let how = match how {
            "left" => JoinType::Left,
            "inner" => JoinType::Inner,
            "outer" => JoinType::Outer,
            "asof" => JoinType::AsOf(AsOfOptions {
                strategy: AsofStrategy::Backward,
                left_by: None,
                right_by: None,
                tolerance: None,
                tolerance_str: None,
            }),
            "cross" => JoinType::Cross,
            _ => panic!("not supported"),
        };

        let df = self
            .df
            .join(&other.df, left_on, right_on, how, Some(suffix))
            .map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
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

    pub fn with_column(&mut self, s: PySeries) -> PyResult<Self> {
        let mut df = self.df.clone();
        df.with_column(s.series).map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    /// Get datatypes
    pub fn dtypes(&self, py: Python) -> PyObject {
        let iter = self
            .df
            .iter()
            .map(|s| Wrap(s.dtype().clone()).to_object(py));
        PyList::new(py, iter).to_object(py)
    }

    pub fn n_chunks(&self) -> PyResult<usize> {
        let n = self.df.n_chunks().map_err(PyPolarsErr::from)?;
        Ok(n)
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
        let df = self.df.select(&selection).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn filter(&self, mask: &PySeries) -> PyResult<Self> {
        let filter_series = &mask.series;
        if let Ok(ca) = filter_series.bool() {
            let df = self.df.filter(ca).map_err(PyPolarsErr::from)?;
            Ok(PyDataFrame::new(df))
        } else {
            Err(PyRuntimeError::new_err("Expected a boolean mask"))
        }
    }

    pub fn take(&self, indices: Wrap<Vec<u32>>) -> PyResult<Self> {
        let indices = indices.0;
        let indices = UInt32Chunked::from_vec("", indices);
        let df = self.df.take(&indices).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn take_with_series(&self, indices: &PySeries) -> PyResult<Self> {
        let idx = indices.series.u32().map_err(PyPolarsErr::from)?;
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
                },
            )
            .map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn sort_in_place(&mut self, by_column: &str, reverse: bool) -> PyResult<()> {
        self.df
            .sort_in_place([by_column], reverse)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn replace(&mut self, column: &str, new_col: PySeries) -> PyResult<()> {
        self.df
            .replace(column, new_col.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    pub fn rename(&mut self, column: &str, new_col: &str) -> PyResult<()> {
        self.df.rename(column, new_col).map_err(PyPolarsErr::from)?;
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

    pub fn slice(&self, offset: usize, length: usize) -> Self {
        let df = self.df.slice(offset as i64, length);
        df.into()
    }

    pub fn head(&self, length: Option<usize>) -> Self {
        let df = self.df.head(length);
        PyDataFrame::new(df)
    }

    pub fn tail(&self, length: Option<usize>) -> Self {
        let df = self.df.tail(length);
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

    pub fn with_row_count(&self, name: &str, offset: Option<u32>) -> PyResult<Self> {
        let df = self
            .df
            .with_row_count(name, offset)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn groupby(&self, by: Vec<&str>, select: Option<Vec<String>>, agg: &str) -> PyResult<Self> {
        let gb = Python::with_gil(|py| py.allow_threads(|| self.df.groupby(&by)))
            .map_err(PyPolarsErr::from)?;
        let selection = match select.as_ref() {
            Some(s) => gb.select(s),
            None => gb,
        };
        finish_groupby(selection, agg)
    }

    pub fn groupby_agg(
        &self,
        by: Vec<&str>,
        column_to_agg: Vec<(&str, Vec<&str>)>,
    ) -> PyResult<Self> {
        let gb = self.df.groupby(&by).map_err(PyPolarsErr::from)?;
        let df = gb.agg(&column_to_agg).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn groupby_apply(&self, by: Vec<&str>, lambda: PyObject) -> PyResult<Self> {
        let gb = self.df.groupby(&by).map_err(PyPolarsErr::from)?;
        let function = move |df: DataFrame| {
            let gil = Python::acquire_gil();
            let py = gil.python();
            // get the pypolars module
            let pypolars = PyModule::import(py, "polars").unwrap();

            // create a PyDataFrame struct/object for Python
            let pydf = PyDataFrame::new(df);

            // Wrap this PySeries object in the python side DataFrame wrapper
            let python_df_wrapper = pypolars.getattr("wrap_df").unwrap().call1((pydf,)).unwrap();

            // call the lambda and get a python side DataFrame wrapper
            let result_df_wrapper = match lambda.call1(py, (python_df_wrapper,)) {
                Ok(pyobj) => pyobj,
                Err(e) => panic!("UDF failed: {}", e.pvalue(py)),
            };
            // unpack the wrapper in a PyDataFrame
            let py_pydf = result_df_wrapper.getattr(py, "_df").expect(
                "Could net get DataFrame attribute '_df'. Make sure that you return a DataFrame object.",
            );
            // Downcast to Rust
            let pydf = py_pydf.extract::<PyDataFrame>(py).unwrap();
            // Finally get the actual DataFrame
            Ok(pydf.df)
        };
        // We don't use `py.allow_threads(|| gb.par_apply(..)` because that segfaulted
        // due to code related to Pyo3 or rayon, cannot reproduce it in native polars
        // so we lose parallelism, but it doesn't really matter because we are GIL bound anyways
        // and this function should not be used in ideomatic polars anyway.
        let df = gb.apply(function).map_err(PyPolarsErr::from)?;

        Ok(df.into())
    }

    pub fn groupby_quantile(
        &self,
        by: Vec<&str>,
        select: Vec<String>,
        quantile: f64,
        interpolation: &str,
    ) -> PyResult<Self> {
        let interpol = match interpolation {
            "nearest" => QuantileInterpolOptions::Nearest,
            "lower" => QuantileInterpolOptions::Lower,
            "higher" => QuantileInterpolOptions::Higher,
            "midpoint" => QuantileInterpolOptions::Midpoint,
            "linear" => QuantileInterpolOptions::Linear,
            _ => panic!("not supported"),
        };
        let gb = self.df.groupby(&by).map_err(PyPolarsErr::from)?;
        let selection = gb.select(&select);
        let df = selection.quantile(quantile, interpol);
        let df = df.map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn pivot(
        &self,
        by: Vec<String>,
        pivot_column: Vec<String>,
        values_column: Vec<String>,
        agg: &str,
    ) -> PyResult<Self> {
        let mut gb = self.df.groupby(&by).map_err(PyPolarsErr::from)?;
        let pivot = gb.pivot(pivot_column, values_column);
        let df = match agg {
            "first" => pivot.first(),
            "min" => pivot.min(),
            "max" => pivot.max(),
            "mean" => pivot.mean(),
            "median" => pivot.median(),
            "sum" => pivot.sum(),
            "count" => pivot.count(),
            "last" => pivot.last(),
            a => Err(PolarsError::ComputeError(
                format!("agg fn {} does not exists", a).into(),
            )),
        };
        let df = df.map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn clone(&self) -> Self {
        PyDataFrame::new(self.df.clone())
    }

    pub fn melt(&self, id_vars: Vec<String>, value_vars: Vec<String>) -> PyResult<Self> {
        let df = self
            .df
            .melt(id_vars, value_vars)
            .map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn pivot2(
        &self,
        values: Vec<String>,
        index: Vec<String>,
        columns: Vec<String>,
        aggregate_fn: Wrap<PivotAgg>,
        maintain_order: bool,
    ) -> PyResult<Self> {
        let fun = match maintain_order {
            true => DataFrame::pivot,
            false => DataFrame::pivot_stable,
        };
        let df =
            fun(&self.df, values, index, columns, aggregate_fn.0).map_err(PyPolarsErr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn shift(&self, periods: i64) -> Self {
        self.df.shift(periods).into()
    }

    pub fn distinct(
        &self,
        py: Python,
        maintain_order: bool,
        subset: Option<Vec<String>>,
        keep: Wrap<DistinctKeepStrategy>,
    ) -> PyResult<Self> {
        let df = py.allow_threads(|| {
            let subset = subset.as_ref().map(|v| v.as_ref());
            match maintain_order {
                true => self.df.distinct_stable(subset, keep.0),
                false => self.df.distinct(subset, keep.0),
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

    pub fn std(&self) -> Self {
        self.df.std().into()
    }

    pub fn var(&self) -> Self {
        self.df.var().into()
    }

    pub fn median(&self) -> Self {
        self.df.median().into()
    }

    pub fn hmean(&self, null_strategy: &str) -> PyResult<Option<PySeries>> {
        let strategy = str_to_null_strategy(null_strategy)?;
        let s = self.df.hmean(strategy).map_err(PyPolarsErr::from)?;
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

    pub fn hsum(&self, null_strategy: &str) -> PyResult<Option<PySeries>> {
        let strategy = str_to_null_strategy(null_strategy)?;
        let s = self.df.hsum(strategy).map_err(PyPolarsErr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn quantile(&self, quantile: f64, interpolation: &str) -> PyResult<Self> {
        let interpol = match interpolation {
            "nearest" => QuantileInterpolOptions::Nearest,
            "lower" => QuantileInterpolOptions::Lower,
            "higher" => QuantileInterpolOptions::Higher,
            "midpoint" => QuantileInterpolOptions::Midpoint,
            "linear" => QuantileInterpolOptions::Linear,
            _ => panic!("not supported"),
        };
        let df = self
            .df
            .quantile(quantile, interpol)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn to_dummies(&self) -> PyResult<Self> {
        let df = self.df.to_dummies().map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    pub fn null_count(&self) -> Self {
        let df = self.df.null_count();
        df.into()
    }

    pub fn apply(
        &self,
        lambda: &PyAny,
        output_type: Option<Wrap<DataType>>,
        inference_size: usize,
    ) -> PyResult<(PyObject, bool)> {
        let gil = Python::acquire_gil();
        let py = gil.python();
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
    }

    pub fn shrink_to_fit(&mut self) {
        self.df.shrink_to_fit();
    }

    pub fn hash_rows(&self, k0: u64, k1: u64, k2: u64, k3: u64) -> PyResult<PySeries> {
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

fn finish_groupby(gb: GroupBy, agg: &str) -> PyResult<PyDataFrame> {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let df = py.allow_threads(|| match agg {
        "min" => gb.min(),
        "max" => gb.max(),
        "mean" => gb.mean(),
        "first" => gb.first(),
        "last" => gb.last(),
        "sum" => gb.sum(),
        "count" => gb.count(),
        "n_unique" => gb.n_unique(),
        "median" => gb.median(),
        "agg_list" => gb.agg_list(),
        "groups" => gb.groups(),
        "std" => gb.std(),
        "var" => gb.var(),
        a => Err(PolarsError::ComputeError(
            format!("agg fn {} does not exists", a).into(),
        )),
    });

    let df = df.map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame::new(df))
}
