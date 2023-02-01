use std::borrow::Cow;
use std::collections::HashMap;
use std::io::BufWriter;
use std::path::PathBuf;

use polars::io::RowCount;
#[cfg(feature = "csv-file")]
use polars::lazy::frame::LazyCsvReader;
#[cfg(feature = "json")]
use polars::lazy::frame::LazyJsonLineReader;
use polars::lazy::frame::{AllowedOptimizations, LazyFrame, LazyGroupBy};
use polars::lazy::prelude::col;
use polars::prelude::{ClosedWindow, CsvEncoding, DataFrame, Field, JoinType, Schema};
use polars::time::*;
use polars_core::cloud;
use polars_core::frame::explode::MeltArgs;
use polars_core::frame::UniqueKeepStrategy;
use polars_core::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::arrow_interop::to_rust::pyarrow_schema_to_rust;
use crate::conversion::Wrap;
use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::file::get_file_like;
use crate::lazy::dsl::PyExpr;
use crate::lazy::utils::py_exprs_to_exprs;
use crate::prelude::*;
use crate::py_modules::POLARS;

#[pyclass]
#[repr(transparent)]
pub struct PyLazyGroupBy {
    // option because we cannot get a self by value in pyo3
    pub lgb: Option<LazyGroupBy>,
}
/// Extract CloudOptions from a Python object.
fn extract_cloud_options(url: &str, py_object: PyObject) -> PyResult<cloud::CloudOptions> {
    let untyped_options = Python::with_gil(|py| py_object.extract::<HashMap<String, String>>(py))
        .expect("Expected a dictionary for cloud_options");
    Ok(
        cloud::CloudOptions::from_untyped_config(url, untyped_options)
            .map_err(PyPolarsErr::from)?,
    )
}

#[pymethods]
impl PyLazyGroupBy {
    pub fn agg(&mut self, aggs: Vec<PyExpr>) -> PyLazyFrame {
        let lgb = self.lgb.take().unwrap();
        let aggs = py_exprs_to_exprs(aggs);
        lgb.agg(aggs).into()
    }

    pub fn head(&mut self, n: usize) -> PyLazyFrame {
        let lgb = self.lgb.take().unwrap();
        lgb.head(Some(n)).into()
    }

    pub fn tail(&mut self, n: usize) -> PyLazyFrame {
        let lgb = self.lgb.take().unwrap();
        lgb.tail(Some(n)).into()
    }

    pub fn apply(
        &mut self,
        lambda: PyObject,
        schema: Option<Wrap<Schema>>,
    ) -> PyResult<PyLazyFrame> {
        let lgb = self.lgb.take().unwrap();
        let schema = match schema {
            Some(schema) => Arc::new(schema.0),
            None => LazyFrame::from(lgb.logical_plan.clone())
                .schema()
                .map_err(PyPolarsErr::from)?,
        };

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
                let result_df_wrapper = lambda.call1(py, (python_df_wrapper,)).map_err(|e| {
                    PolarsError::ComputeError(
                        format!("User provided python function failed: {e}").into(),
                    )
                })?;
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
        Ok(lgb.apply(function, schema).into())
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyLazyFrame {
    // option because we cannot get a self by value in pyo3
    pub ldf: LazyFrame,
}

impl PyLazyFrame {
    fn get_schema(&self) -> PyResult<SchemaRef> {
        let schema = self.ldf.schema().map_err(PyPolarsErr::from)?;
        Ok(schema)
    }
}

impl From<LazyFrame> for PyLazyFrame {
    fn from(ldf: LazyFrame) -> Self {
        PyLazyFrame { ldf }
    }
}

#[pymethods]
#[allow(clippy::should_implement_trait)]
impl PyLazyFrame {
    #[cfg(all(feature = "json", feature = "serde_json"))]
    pub fn write_json(&self, py_f: PyObject) -> PyResult<()> {
        let file = BufWriter::new(get_file_like(py_f, true)?);
        serde_json::to_writer(file, &self.ldf.logical_plan)
            .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
        Ok(())
    }

    #[staticmethod]
    #[cfg(feature = "json")]
    pub fn read_json(py_f: PyObject) -> PyResult<Self> {
        // it is faster to first read to memory and then parse: https://github.com/serde-rs/json/issues/160
        // so don't bother with files.
        let mut json = String::new();
        let _ = get_file_like(py_f, false)?
            .read_to_string(&mut json)
            .unwrap();

        // Safety
        // we skipped the serializing/deserializing of the static in lifetime in `DataType`
        // so we actually don't have a lifetime at all when serializing.

        // &str still has a lifetime. Bit its ok, because we drop it immediately
        // in this scope
        let json = unsafe { std::mem::transmute::<&'_ str, &'static str>(json.as_str()) };

        let lp = serde_json::from_str::<LogicalPlan>(json)
            .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
        Ok(LazyFrame::from(lp).into())
    }

    #[staticmethod]
    #[cfg(feature = "json")]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, infer_schema_length, batch_size, n_rows, low_memory, rechunk, row_count))]
    pub fn new_from_ndjson(
        path: String,
        infer_schema_length: Option<usize>,
        batch_size: Option<usize>,
        n_rows: Option<usize>,
        low_memory: bool,
        rechunk: bool,
        row_count: Option<(String, IdxSize)>,
    ) -> PyResult<Self> {
        let row_count = row_count.map(|(name, offset)| RowCount { name, offset });

        let lf = LazyJsonLineReader::new(path)
            .with_infer_schema_length(infer_schema_length)
            .with_batch_size(batch_size)
            .with_n_rows(n_rows)
            .low_memory(low_memory)
            .with_rechunk(rechunk)
            .with_row_count(row_count)
            .finish()
            .map_err(PyPolarsErr::from)?;
        Ok(lf.into())
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "csv-file")]
    #[pyo3(signature = (path, sep, has_header, ignore_errors, skip_rows, n_rows, cache, overwrite_dtype,
        low_memory, comment_char, quote_char, null_values, missing_utf8_is_empty_string,
        infer_schema_length, with_schema_modify, rechunk, skip_rows_after_header,
        encoding, row_count, parse_dates, eol_char,
    )
    )]
    pub fn new_from_csv(
        path: String,
        sep: &str,
        has_header: bool,
        ignore_errors: bool,
        skip_rows: usize,
        n_rows: Option<usize>,
        cache: bool,
        overwrite_dtype: Option<Vec<(&str, Wrap<DataType>)>>,
        low_memory: bool,
        comment_char: Option<&str>,
        quote_char: Option<&str>,
        null_values: Option<Wrap<NullValues>>,
        missing_utf8_is_empty_string: bool,
        infer_schema_length: Option<usize>,
        with_schema_modify: Option<PyObject>,
        rechunk: bool,
        skip_rows_after_header: usize,
        encoding: Wrap<CsvEncoding>,
        row_count: Option<(String, IdxSize)>,
        parse_dates: bool,
        eol_char: &str,
    ) -> PyResult<Self> {
        let null_values = null_values.map(|w| w.0);
        let comment_char = comment_char.map(|s| s.as_bytes()[0]);
        let quote_char = quote_char.map(|s| s.as_bytes()[0]);
        let delimiter = sep.as_bytes()[0];
        let eol_char = eol_char.as_bytes()[0];
        let row_count = row_count.map(|(name, offset)| RowCount { name, offset });

        let overwrite_dtype = overwrite_dtype.map(|overwrite_dtype| {
            let fields = overwrite_dtype
                .into_iter()
                .map(|(name, dtype)| Field::new(name, dtype.0));
            Schema::from(fields)
        });
        let mut r = LazyCsvReader::new(path)
            .with_infer_schema_length(infer_schema_length)
            .with_delimiter(delimiter)
            .has_header(has_header)
            .with_ignore_errors(ignore_errors)
            .with_skip_rows(skip_rows)
            .with_n_rows(n_rows)
            .with_cache(cache)
            .with_dtype_overwrite(overwrite_dtype.as_ref())
            .low_memory(low_memory)
            .with_comment_char(comment_char)
            .with_quote_char(quote_char)
            .with_end_of_line_char(eol_char)
            .with_rechunk(rechunk)
            .with_skip_rows_after_header(skip_rows_after_header)
            .with_encoding(encoding.0)
            .with_row_count(row_count)
            .with_parse_dates(parse_dates)
            .with_null_values(null_values)
            .with_missing_is_null(!missing_utf8_is_empty_string);

        if let Some(lambda) = with_schema_modify {
            let f = |schema: Schema| {
                let iter = schema.iter_names();
                Python::with_gil(|py| {
                    let names = PyList::new(py, iter);

                    let out = lambda.call1(py, (names,)).expect("python function failed");
                    let new_names = out
                        .extract::<Vec<String>>(py)
                        .expect("python function should return List[str]");
                    assert_eq!(new_names.len(), schema.len(), "The length of the new names list should be equal to the original column length");

                    let fields = schema
                        .iter_dtypes()
                        .zip(new_names)
                        .map(|(dtype, name)| Field::from_owned(name, dtype.clone()));
                    Ok(Schema::from(fields))
                })
            };
            r = r.with_schema_modify(f).map_err(PyPolarsErr::from)?
        }

        Ok(r.finish().map_err(PyPolarsErr::from)?.into())
    }

    #[cfg(feature = "parquet")]
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, n_rows, cache, parallel, rechunk, row_count,
        low_memory, cloud_options)
    )]
    pub fn new_from_parquet(
        path: String,
        n_rows: Option<usize>,
        cache: bool,
        parallel: Wrap<ParallelStrategy>,
        rechunk: bool,
        row_count: Option<(String, IdxSize)>,
        low_memory: bool,
        cloud_options: Option<PyObject>,
    ) -> PyResult<Self> {
        let cloud_options = cloud_options
            .map(|po| extract_cloud_options(&path, po))
            .transpose()?;
        let row_count = row_count.map(|(name, offset)| RowCount { name, offset });
        let args = ScanArgsParquet {
            n_rows,
            cache,
            parallel: parallel.0,
            rechunk,
            row_count,
            low_memory,
            cloud_options,
        };
        let lf = LazyFrame::scan_parquet(path, args).map_err(PyPolarsErr::from)?;
        Ok(lf.into())
    }

    #[cfg(feature = "ipc")]
    #[staticmethod]
    #[pyo3(signature = (path, n_rows, cache, rechunk, row_count, memory_map))]
    pub fn new_from_ipc(
        path: String,
        n_rows: Option<usize>,
        cache: bool,
        rechunk: bool,
        row_count: Option<(String, IdxSize)>,
        memory_map: bool,
    ) -> PyResult<Self> {
        let row_count = row_count.map(|(name, offset)| RowCount { name, offset });
        let args = ScanArgsIpc {
            n_rows,
            cache,
            rechunk,
            row_count,
            memmap: memory_map,
        };
        let lf = LazyFrame::scan_ipc(path, args).map_err(PyPolarsErr::from)?;
        Ok(lf.into())
    }

    #[staticmethod]
    pub fn scan_from_python_function_arrow_schema(
        schema: &PyList,
        scan_fn: Vec<u8>,
        pyarrow: bool,
    ) -> PyResult<Self> {
        let schema = pyarrow_schema_to_rust(schema)?;
        Ok(LazyFrame::scan_from_python_function(schema, scan_fn, pyarrow).into())
    }

    #[staticmethod]
    pub fn scan_from_python_function_pl_schema(
        schema: Vec<(&str, Wrap<DataType>)>,
        scan_fn: Vec<u8>,
        pyarrow: bool,
    ) -> PyResult<Self> {
        let schema = Schema::from_iter(schema.into_iter().map(|(name, dt)| Field::new(name, dt.0)));
        Ok(LazyFrame::scan_from_python_function(schema, scan_fn, pyarrow).into())
    }

    pub fn describe_plan(&self) -> String {
        self.ldf.describe_plan()
    }

    pub fn describe_optimized_plan(&self) -> PyResult<String> {
        let result = self
            .ldf
            .describe_optimized_plan()
            .map_err(PyPolarsErr::from)?;
        Ok(result)
    }
    pub fn to_dot(&self, optimized: bool) -> PyResult<String> {
        let result = self.ldf.to_dot(optimized).map_err(PyPolarsErr::from)?;
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn optimization_toggle(
        &self,
        type_coercion: bool,
        predicate_pushdown: bool,
        projection_pushdown: bool,
        simplify_expr: bool,
        slice_pushdown: bool,
        cse: bool,
        streaming: bool,
    ) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        let mut ldf = ldf
            .with_type_coercion(type_coercion)
            .with_predicate_pushdown(predicate_pushdown)
            .with_simplify_expr(simplify_expr)
            .with_slice_pushdown(slice_pushdown)
            .with_streaming(streaming)
            .with_projection_pushdown(projection_pushdown);

        #[cfg(feature = "cse")]
        {
            ldf = ldf.with_common_subplan_elimination(cse);
        }

        ldf.into()
    }

    pub fn sort(&self, by_column: &str, reverse: bool, nulls_last: bool) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.sort(
            by_column,
            SortOptions {
                descending: reverse,
                nulls_last,
                multithreaded: true,
            },
        )
        .into()
    }

    pub fn sort_by_exprs(
        &self,
        by_column: Vec<PyExpr>,
        reverse: Vec<bool>,
        nulls_last: bool,
    ) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        let exprs = py_exprs_to_exprs(by_column);
        ldf.sort_by_exprs(exprs, reverse, nulls_last).into()
    }
    pub fn cache(&self) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.cache().into()
    }

    pub fn profile(&self, py: Python) -> PyResult<(PyDataFrame, PyDataFrame)> {
        // if we don't allow threads and we have udfs trying to acquire the gil from different
        // threads we deadlock.
        let (df, time_df) = py.allow_threads(|| {
            let ldf = self.ldf.clone();
            ldf.profile().map_err(PyPolarsErr::from)
        })?;
        Ok((df.into(), time_df.into()))
    }

    pub fn collect(&self, py: Python) -> PyResult<PyDataFrame> {
        // if we don't allow threads and we have udfs trying to acquire the gil from different
        // threads we deadlock.
        let df = py.allow_threads(|| {
            let ldf = self.ldf.clone();
            ldf.collect().map_err(PyPolarsErr::from)
        })?;
        Ok(df.into())
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "streaming")]
    #[pyo3(signature = (path, compression, compression_level, statistics, row_group_size, data_pagesize_limit, maintain_order))]
    pub fn sink_parquet(
        &self,
        py: Python,
        path: PathBuf,
        compression: &str,
        compression_level: Option<i32>,
        statistics: bool,
        row_group_size: Option<usize>,
        data_pagesize_limit: Option<usize>,
        maintain_order: bool,
    ) -> PyResult<()> {
        let compression = parse_parquet_compression(compression, compression_level)?;

        let options = ParquetWriteOptions {
            compression,
            statistics,
            row_group_size,
            data_pagesize_limit,
            maintain_order,
        };

        // if we don't allow threads and we have udfs trying to acquire the gil from different
        // threads we deadlock.
        py.allow_threads(|| {
            let ldf = self.ldf.clone();
            ldf.sink_parquet(path, options).map_err(PyPolarsErr::from)
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "streaming")]
    #[pyo3(signature = (path, compression, maintain_order))]
    pub fn sink_ipc(
        &self,
        py: Python,
        path: PathBuf,
        compression: Option<Wrap<IpcCompression>>,
        maintain_order: bool,
    ) -> PyResult<()> {
        let options = IpcWriterOptions {
            compression: compression.map(|c| c.0),
            maintain_order,
        };

        // if we don't allow threads and we have udfs trying to acquire the gil from different
        // threads we deadlock.
        py.allow_threads(|| {
            let ldf = self.ldf.clone();
            ldf.sink_ipc(path, options).map_err(PyPolarsErr::from)
        })?;
        Ok(())
    }

    pub fn fetch(&self, py: Python, n_rows: usize) -> PyResult<PyDataFrame> {
        let ldf = self.ldf.clone();
        let df = py.allow_threads(|| ldf.fetch(n_rows).map_err(PyPolarsErr::from))?;
        Ok(df.into())
    }

    pub fn filter(&mut self, predicate: PyExpr) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.filter(predicate.inner).into()
    }

    pub fn select(&mut self, exprs: Vec<PyExpr>) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        let exprs = py_exprs_to_exprs(exprs);
        ldf.select(exprs).into()
    }

    pub fn groupby(&mut self, by: Vec<PyExpr>, maintain_order: bool) -> PyLazyGroupBy {
        let ldf = self.ldf.clone();
        let by = py_exprs_to_exprs(by);
        let lazy_gb = if maintain_order {
            ldf.groupby_stable(by)
        } else {
            ldf.groupby(by)
        };

        PyLazyGroupBy { lgb: Some(lazy_gb) }
    }

    pub fn groupby_rolling(
        &mut self,
        index_column: String,
        period: &str,
        offset: &str,
        closed: Wrap<ClosedWindow>,
        by: Vec<PyExpr>,
    ) -> PyLazyGroupBy {
        let closed_window = closed.0;
        let ldf = self.ldf.clone();
        let by = by
            .into_iter()
            .map(|pyexpr| pyexpr.inner)
            .collect::<Vec<_>>();
        let lazy_gb = ldf.groupby_rolling(
            by,
            RollingGroupOptions {
                index_column,
                period: Duration::parse(period),
                offset: Duration::parse(offset),
                closed_window,
            },
        );

        PyLazyGroupBy { lgb: Some(lazy_gb) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn groupby_dynamic(
        &mut self,
        index_column: String,
        every: &str,
        period: &str,
        offset: &str,
        truncate: bool,
        include_boundaries: bool,
        closed: Wrap<ClosedWindow>,
        by: Vec<PyExpr>,
        start_by: Wrap<StartBy>,
    ) -> PyLazyGroupBy {
        let closed_window = closed.0;
        let by = by
            .into_iter()
            .map(|pyexpr| pyexpr.inner)
            .collect::<Vec<_>>();
        let ldf = self.ldf.clone();
        let lazy_gb = ldf.groupby_dynamic(
            by,
            DynamicGroupOptions {
                index_column,
                every: Duration::parse(every),
                period: Duration::parse(period),
                offset: Duration::parse(offset),
                truncate,
                include_boundaries,
                closed_window,
                start_by: start_by.0,
            },
        );

        PyLazyGroupBy { lgb: Some(lazy_gb) }
    }

    pub fn with_context(&self, contexts: Vec<PyLazyFrame>) -> PyLazyFrame {
        let contexts = contexts.into_iter().map(|ldf| ldf.ldf).collect::<Vec<_>>();
        self.ldf.clone().with_context(contexts).into()
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "asof_join")]
    #[pyo3(signature = (other, left_on, right_on, left_by, right_by, allow_parallel, force_parallel, suffix, strategy, tolerance, tolerance_str))]
    pub fn join_asof(
        &self,
        other: PyLazyFrame,
        left_on: PyExpr,
        right_on: PyExpr,
        left_by: Option<Vec<String>>,
        right_by: Option<Vec<String>>,
        allow_parallel: bool,
        force_parallel: bool,
        suffix: String,
        strategy: Wrap<AsofStrategy>,
        tolerance: Option<Wrap<AnyValue<'_>>>,
        tolerance_str: Option<String>,
    ) -> PyResult<Self> {
        let ldf = self.ldf.clone();
        let other = other.ldf;
        let left_on = left_on.inner;
        let right_on = right_on.inner;
        Ok(ldf
            .join_builder()
            .with(other)
            .left_on([left_on])
            .right_on([right_on])
            .allow_parallel(allow_parallel)
            .force_parallel(force_parallel)
            .how(JoinType::AsOf(AsOfOptions {
                strategy: strategy.0,
                left_by,
                right_by,
                tolerance: tolerance.map(|t| t.0.into_static().unwrap()),
                tolerance_str,
            }))
            .suffix(suffix)
            .finish()
            .into())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn join(
        &self,
        other: PyLazyFrame,
        left_on: Vec<PyExpr>,
        right_on: Vec<PyExpr>,
        allow_parallel: bool,
        force_parallel: bool,
        how: Wrap<JoinType>,
        suffix: String,
    ) -> PyResult<Self> {
        let ldf = self.ldf.clone();
        let other = other.ldf;
        let left_on = left_on
            .into_iter()
            .map(|pyexpr| pyexpr.inner)
            .collect::<Vec<_>>();
        let right_on = right_on
            .into_iter()
            .map(|pyexpr| pyexpr.inner)
            .collect::<Vec<_>>();

        Ok(ldf
            .join_builder()
            .with(other)
            .left_on(left_on)
            .right_on(right_on)
            .allow_parallel(allow_parallel)
            .force_parallel(force_parallel)
            .how(how.0)
            .suffix(suffix)
            .finish()
            .into())
    }

    pub fn with_column(&mut self, expr: PyExpr) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.with_column(expr.inner).into()
    }

    pub fn with_columns(&mut self, exprs: Vec<PyExpr>) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.with_columns(py_exprs_to_exprs(exprs)).into()
    }

    pub fn rename(&mut self, existing: Vec<String>, new: Vec<String>) -> PyLazyFrame {
        let ldf = self.ldf.clone();
        ldf.rename(existing, new).into()
    }

    pub fn reverse(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.reverse().into()
    }

    pub fn shift(&self, periods: i64) -> Self {
        let ldf = self.ldf.clone();
        ldf.shift(periods).into()
    }

    pub fn shift_and_fill(&self, periods: i64, fill_value: PyExpr) -> Self {
        let ldf = self.ldf.clone();
        ldf.shift_and_fill(periods, fill_value.inner).into()
    }

    pub fn fill_nan(&self, fill_value: PyExpr) -> Self {
        let ldf = self.ldf.clone();
        ldf.fill_nan(fill_value.inner).into()
    }

    pub fn min(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.min().into()
    }

    pub fn max(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.max().into()
    }

    pub fn sum(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.sum().into()
    }

    pub fn mean(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.mean().into()
    }

    pub fn std(&self, ddof: u8) -> Self {
        let ldf = self.ldf.clone();
        ldf.std(ddof).into()
    }

    pub fn var(&self, ddof: u8) -> Self {
        let ldf = self.ldf.clone();
        ldf.var(ddof).into()
    }

    pub fn median(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.median().into()
    }

    pub fn quantile(&self, quantile: PyExpr, interpolation: Wrap<QuantileInterpolOptions>) -> Self {
        let ldf = self.ldf.clone();
        ldf.quantile(quantile.inner, interpolation.0).into()
    }

    pub fn explode(&self, column: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.clone();
        let column = py_exprs_to_exprs(column);
        ldf.explode(column).into()
    }

    #[pyo3(signature = (maintain_order, subset, keep))]
    pub fn unique(
        &self,
        maintain_order: bool,
        subset: Option<Vec<String>>,
        keep: Wrap<UniqueKeepStrategy>,
    ) -> Self {
        let ldf = self.ldf.clone();
        match maintain_order {
            true => ldf.unique_stable(subset, keep.0),
            false => ldf.unique(subset, keep.0),
        }
        .into()
    }

    pub fn drop_nulls(&self, subset: Option<Vec<String>>) -> Self {
        let ldf = self.ldf.clone();
        ldf.drop_nulls(subset.map(|v| v.into_iter().map(|s| col(&s)).collect()))
            .into()
    }

    pub fn slice(&self, offset: i64, len: Option<IdxSize>) -> Self {
        let ldf = self.ldf.clone();
        ldf.slice(offset, len.unwrap_or(IdxSize::MAX)).into()
    }

    pub fn tail(&self, n: IdxSize) -> Self {
        let ldf = self.ldf.clone();
        ldf.tail(n).into()
    }

    pub fn melt(
        &self,
        id_vars: Vec<String>,
        value_vars: Vec<String>,
        value_name: Option<String>,
        variable_name: Option<String>,
    ) -> Self {
        let args = MeltArgs {
            id_vars,
            value_vars,
            value_name,
            variable_name,
        };

        let ldf = self.ldf.clone();
        ldf.melt(args).into()
    }

    pub fn with_row_count(&self, name: &str, offset: Option<IdxSize>) -> Self {
        let ldf = self.ldf.clone();
        ldf.with_row_count(name, offset).into()
    }

    #[pyo3(signature = (lambda, predicate_pushdown, projection_pushdown, slice_pushdown, streamable, schema, validate_output))]
    #[allow(clippy::too_many_arguments)]
    pub fn map(
        &self,
        lambda: PyObject,
        predicate_pushdown: bool,
        projection_pushdown: bool,
        slice_pushdown: bool,
        streamable: bool,
        schema: Option<Wrap<Schema>>,
        validate_output: bool,
    ) -> Self {
        let opt = AllowedOptimizations {
            predicate_pushdown,
            projection_pushdown,
            slice_pushdown,
            streaming: streamable,
            ..Default::default()
        };
        let schema = schema.map(|schema| Arc::new(schema.0));
        let schema2 = schema.clone();

        let function = move |df: DataFrame| {
            Python::with_gil(|py| {
                let opt_schema = schema2.clone();

                let expected_schema = if let Some(schema) = opt_schema.as_ref() {
                    Cow::Borrowed(schema.as_ref())
                }
                // only materialize if we validate the output
                else if validate_output {
                    Cow::Owned(df.schema())
                }
                // do not materialize the schema, we will ignore it.
                else {
                    Cow::Owned(Schema::default())
                };

                // create a PyDataFrame struct/object for Python
                let pydf = PyDataFrame::new(df);
                // Wrap this PyDataFrame object in the python side DataFrame wrapper
                let python_df_wrapper = POLARS
                    .getattr(py, "wrap_df")
                    .unwrap()
                    .call1(py, (pydf,))
                    .unwrap();
                // call the lambda and get a python side Series wrapper

                let result_df_wrapper = lambda.call1(py, (python_df_wrapper,)).map_err(|e| {
                    PolarsError::ComputeError(
                        format!("User provided python function failed: {e}").into(),
                    )
                })?;
                // unpack the wrapper in a PyDataFrame
                let py_pydf = result_df_wrapper.getattr(py, "_df").map_err(|_| {
                    let pytype = result_df_wrapper.as_ref(py).get_type();
                    PolarsError::ComputeError(
                        format!(
                            "Expected 'LazyFrame.map' to return a 'DataFrame', got a '{pytype}'",
                        )
                        .into(),
                    )
                })?;

                // Downcast to Rust
                let pydf = py_pydf.extract::<PyDataFrame>(py).unwrap();
                // Finally get the actual DataFrame
                let df = pydf.df;

                if validate_output {
                    let output_schema = df.schema();
                    if expected_schema.as_ref() != &output_schema {
                        return Err(PolarsError::ComputeError(
                            format!("The output schema of 'LazyFrame.map' is incorrect. Expected: {expected_schema:?}\n\
                        Got: {output_schema:?}").into()
                        ));
                    }
                }
                Ok(df)
            })
        };

        let ldf = self.ldf.clone();

        let udf_schema =
            schema.map(move |s| Arc::new(move |_: &Schema| Ok(s.clone())) as Arc<dyn UdfSchema>);
        ldf.map(function, opt, udf_schema, None).into()
    }

    pub fn drop_columns(&self, cols: Vec<String>) -> Self {
        let ldf = self.ldf.clone();
        ldf.drop_columns(cols).into()
    }

    pub fn clone(&self) -> PyLazyFrame {
        self.ldf.clone().into()
    }

    pub fn columns(&self) -> PyResult<Vec<String>> {
        Ok(self.get_schema()?.iter_names().cloned().collect())
    }

    pub fn dtypes(&self, py: Python) -> PyResult<PyObject> {
        let schema = self.get_schema()?;
        let iter = schema
            .iter_dtypes()
            .map(|dt| Wrap(dt.clone()).to_object(py));
        Ok(PyList::new(py, iter).to_object(py))
    }

    pub fn schema(&self, py: Python) -> PyResult<PyObject> {
        let schema = self.get_schema()?;
        let schema_dict = PyDict::new(py);

        schema.iter_fields().for_each(|fld| {
            schema_dict
                .set_item(fld.name(), Wrap(fld.data_type().clone()))
                .unwrap()
        });
        Ok(schema_dict.to_object(py))
    }

    pub fn unnest(&self, cols: Vec<String>) -> PyLazyFrame {
        self.ldf.clone().unnest(cols).into()
    }

    pub fn width(&self) -> PyResult<usize> {
        Ok(self.get_schema()?.len())
    }

    #[cfg(feature = "merge_sorted")]
    pub fn merge_sorted(&self, other: PyLazyFrame, key: &str) -> PyResult<Self> {
        let out = self
            .ldf
            .clone()
            .merge_sorted(other.ldf, key)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }
}
