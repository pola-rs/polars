use std::borrow::Cow;
use std::collections::HashMap;
use std::io::BufWriter;
use std::path::PathBuf;

use polars::io::RowCount;
#[cfg(feature = "csv")]
use polars::lazy::frame::LazyCsvReader;
#[cfg(feature = "json")]
use polars::lazy::frame::LazyJsonLineReader;
use polars::lazy::frame::{AllowedOptimizations, LazyFrame};
use polars::lazy::prelude::col;
use polars::prelude::{ClosedWindow, CsvEncoding, DataFrame, Field, JoinType, Schema};
use polars::time::*;
use polars_core::cloud;
use polars_core::frame::explode::MeltArgs;
use polars_core::frame::hash_join::JoinValidation;
use polars_core::frame::UniqueKeepStrategy;
use polars_core::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::arrow_interop::to_rust::pyarrow_schema_to_rust;
use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::expr::ToExprs;
use crate::file::get_file_like;
use crate::prelude::*;
use crate::py_modules::POLARS;
use crate::{PyDataFrame, PyExpr, PyLazyGroupBy};

/// Extract CloudOptions from a Python object.
fn extract_cloud_options(url: &str, py_object: PyObject) -> PyResult<cloud::CloudOptions> {
    let untyped_options = Python::with_gil(|py| py_object.extract::<HashMap<String, String>>(py))
        .expect("Expected a dictionary for cloud_options");
    Ok(
        cloud::CloudOptions::from_untyped_config(url, untyped_options)
            .map_err(PyPolarsErr::from)?,
    )
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyLazyFrame {
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
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        // Used in pickle/pickling
        let mut writer: Vec<u8> = vec![];
        ciborium::ser::into_writer(&self.ldf.logical_plan, &mut writer)
            .map_err(|e| PyPolarsErr::Other(format!("{}", e)))?;

        Ok(PyBytes::new(py, &writer).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        // Used in pickle/pickling
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                let lp: LogicalPlan = ciborium::de::from_reader(s.as_bytes())
                    .map_err(|e| PyPolarsErr::Other(format!("{}", e)))?;
                self.ldf = LazyFrame::from(lp);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    #[cfg(all(feature = "json", feature = "serde_json"))]
    fn write_json(&self, py_f: PyObject) -> PyResult<()> {
        let file = BufWriter::new(get_file_like(py_f, true)?);
        serde_json::to_writer(file, &self.ldf.logical_plan)
            .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
        Ok(())
    }

    #[staticmethod]
    #[cfg(feature = "json")]
    fn read_json(py_f: PyObject) -> PyResult<Self> {
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
    fn new_from_ndjson(
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
    #[cfg(feature = "csv")]
    #[pyo3(signature = (path, separator, has_header, ignore_errors, skip_rows, n_rows, cache, overwrite_dtype,
        low_memory, comment_char, quote_char, null_values, missing_utf8_is_empty_string,
        infer_schema_length, with_schema_modify, rechunk, skip_rows_after_header,
        encoding, row_count, try_parse_dates, eol_char,
    )
    )]
    fn new_from_csv(
        path: String,
        separator: &str,
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
        try_parse_dates: bool,
        eol_char: &str,
    ) -> PyResult<Self> {
        let null_values = null_values.map(|w| w.0);
        let comment_char = comment_char.map(|s| s.as_bytes()[0]);
        let quote_char = quote_char.map(|s| s.as_bytes()[0]);
        let delimiter = separator.as_bytes()[0];
        let eol_char = eol_char.as_bytes()[0];
        let row_count = row_count.map(|(name, offset)| RowCount { name, offset });

        let overwrite_dtype = overwrite_dtype.map(|overwrite_dtype| {
            overwrite_dtype
                .into_iter()
                .map(|(name, dtype)| Field::new(name, dtype.0))
                .collect::<Schema>()
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
            .with_try_parse_dates(try_parse_dates)
            .with_null_values(null_values)
            .with_missing_is_null(!missing_utf8_is_empty_string);

        if let Some(lambda) = with_schema_modify {
            let f = |schema: Schema| {
                let iter = schema.iter_names().map(|s| s.as_str());
                Python::with_gil(|py| {
                    let names = PyList::new(py, iter);

                    let out = lambda.call1(py, (names,)).expect("python function failed");
                    let new_names = out
                        .extract::<Vec<String>>(py)
                        .expect("python function should return List[str]");
                    assert_eq!(new_names.len(), schema.len(), "The length of the new names list should be equal to the original column length");

                    Ok(schema
                        .iter_dtypes()
                        .zip(new_names)
                        .map(|(dtype, name)| Field::from_owned(name.into(), dtype.clone()))
                        .collect())
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
        low_memory, cloud_options, use_statistics)
    )]
    fn new_from_parquet(
        path: String,
        n_rows: Option<usize>,
        cache: bool,
        parallel: Wrap<ParallelStrategy>,
        rechunk: bool,
        row_count: Option<(String, IdxSize)>,
        low_memory: bool,
        cloud_options: Option<PyObject>,
        use_statistics: bool,
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
            use_statistics,
        };
        let lf = LazyFrame::scan_parquet(path, args).map_err(PyPolarsErr::from)?;
        Ok(lf.into())
    }

    #[cfg(feature = "ipc")]
    #[staticmethod]
    #[pyo3(signature = (path, n_rows, cache, rechunk, row_count, memory_map))]
    fn new_from_ipc(
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
    fn scan_from_python_function_arrow_schema(
        schema: &PyList,
        scan_fn: PyObject,
        pyarrow: bool,
    ) -> PyResult<Self> {
        let schema = pyarrow_schema_to_rust(schema)?;
        Ok(LazyFrame::scan_from_python_function(schema, scan_fn, pyarrow).into())
    }

    #[staticmethod]
    fn scan_from_python_function_pl_schema(
        schema: Vec<(&str, Wrap<DataType>)>,
        scan_fn: PyObject,
        pyarrow: bool,
    ) -> PyResult<Self> {
        let schema = Schema::from_iter(schema.into_iter().map(|(name, dt)| Field::new(name, dt.0)));
        Ok(LazyFrame::scan_from_python_function(schema, scan_fn, pyarrow).into())
    }

    fn describe_plan(&self) -> String {
        self.ldf.describe_plan()
    }

    fn describe_optimized_plan(&self) -> PyResult<String> {
        let result = self
            .ldf
            .describe_optimized_plan()
            .map_err(PyPolarsErr::from)?;
        Ok(result)
    }
    fn to_dot(&self, optimized: bool) -> PyResult<String> {
        let result = self.ldf.to_dot(optimized).map_err(PyPolarsErr::from)?;
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn optimization_toggle(
        &self,
        type_coercion: bool,
        predicate_pushdown: bool,
        projection_pushdown: bool,
        simplify_expr: bool,
        slice_pushdown: bool,
        cse: bool,
        streaming: bool,
    ) -> Self {
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

    fn sort(&self, by_column: &str, descending: bool, nulls_last: bool) -> Self {
        let ldf = self.ldf.clone();
        ldf.sort(
            by_column,
            SortOptions {
                descending,
                nulls_last,
                multithreaded: true,
            },
        )
        .into()
    }

    fn sort_by_exprs(&self, by: Vec<PyExpr>, descending: Vec<bool>, nulls_last: bool) -> Self {
        let ldf = self.ldf.clone();
        let exprs = by.to_exprs();
        ldf.sort_by_exprs(exprs, descending, nulls_last).into()
    }

    fn top_k(&self, k: IdxSize, by: Vec<PyExpr>, descending: Vec<bool>, nulls_last: bool) -> Self {
        let ldf = self.ldf.clone();
        let exprs = by.to_exprs();
        ldf.top_k(k, exprs, descending, nulls_last).into()
    }

    fn bottom_k(
        &self,
        k: IdxSize,
        by: Vec<PyExpr>,
        descending: Vec<bool>,
        nulls_last: bool,
    ) -> Self {
        let ldf = self.ldf.clone();
        let exprs = by.to_exprs();
        ldf.bottom_k(k, exprs, descending, nulls_last).into()
    }

    fn cache(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.cache().into()
    }

    fn profile(&self, py: Python) -> PyResult<(PyDataFrame, PyDataFrame)> {
        // if we don't allow threads and we have udfs trying to acquire the gil from different
        // threads we deadlock.
        let (df, time_df) = py.allow_threads(|| {
            let ldf = self.ldf.clone();
            ldf.profile().map_err(PyPolarsErr::from)
        })?;
        Ok((df.into(), time_df.into()))
    }

    fn collect(&self, py: Python) -> PyResult<PyDataFrame> {
        // if we don't allow threads and we have udfs trying to acquire the gil from different
        // threads we deadlock.
        let df = py.allow_threads(|| {
            let ldf = self.ldf.clone();
            ldf.collect().map_err(PyPolarsErr::from)
        })?;
        Ok(df.into())
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(all(feature = "streaming", feature = "parquet"))]
    #[pyo3(signature = (path, compression, compression_level, statistics, row_group_size, data_pagesize_limit, maintain_order))]
    fn sink_parquet(
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
    #[cfg(all(feature = "streaming", feature = "ipc"))]
    #[pyo3(signature = (path, compression, maintain_order))]
    fn sink_ipc(
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

    fn fetch(&self, py: Python, n_rows: usize) -> PyResult<PyDataFrame> {
        let ldf = self.ldf.clone();
        let df = py.allow_threads(|| ldf.fetch(n_rows).map_err(PyPolarsErr::from))?;
        Ok(df.into())
    }

    fn filter(&mut self, predicate: PyExpr) -> Self {
        let ldf = self.ldf.clone();
        ldf.filter(predicate.inner).into()
    }

    fn select(&mut self, exprs: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.clone();
        let exprs = exprs.to_exprs();
        ldf.select(exprs).into()
    }

    fn groupby(&mut self, by: Vec<PyExpr>, maintain_order: bool) -> PyLazyGroupBy {
        let ldf = self.ldf.clone();
        let by = by.to_exprs();
        let lazy_gb = if maintain_order {
            ldf.groupby_stable(by)
        } else {
            ldf.groupby(by)
        };

        PyLazyGroupBy { lgb: Some(lazy_gb) }
    }

    fn groupby_rolling(
        &mut self,
        index_column: PyExpr,
        period: &str,
        offset: &str,
        closed: Wrap<ClosedWindow>,
        by: Vec<PyExpr>,
        check_sorted: bool,
    ) -> PyLazyGroupBy {
        let closed_window = closed.0;
        let ldf = self.ldf.clone();
        let by = by
            .into_iter()
            .map(|pyexpr| pyexpr.inner)
            .collect::<Vec<_>>();
        let lazy_gb = ldf.groupby_rolling(
            index_column.inner,
            by,
            RollingGroupOptions {
                index_column: "".into(),
                period: Duration::parse(period),
                offset: Duration::parse(offset),
                closed_window,
                check_sorted,
            },
        );

        PyLazyGroupBy { lgb: Some(lazy_gb) }
    }

    #[allow(clippy::too_many_arguments)]
    fn groupby_dynamic(
        &mut self,
        index_column: PyExpr,
        every: &str,
        period: &str,
        offset: &str,
        truncate: bool,
        include_boundaries: bool,
        closed: Wrap<ClosedWindow>,
        by: Vec<PyExpr>,
        start_by: Wrap<StartBy>,
        check_sorted: bool,
    ) -> PyLazyGroupBy {
        let closed_window = closed.0;
        let by = by
            .into_iter()
            .map(|pyexpr| pyexpr.inner)
            .collect::<Vec<_>>();
        let ldf = self.ldf.clone();
        let lazy_gb = ldf.groupby_dynamic(
            index_column.inner,
            by,
            DynamicGroupOptions {
                every: Duration::parse(every),
                period: Duration::parse(period),
                offset: Duration::parse(offset),
                truncate,
                include_boundaries,
                closed_window,
                start_by: start_by.0,
                check_sorted,
                ..Default::default()
            },
        );

        PyLazyGroupBy { lgb: Some(lazy_gb) }
    }

    fn with_context(&self, contexts: Vec<Self>) -> Self {
        let contexts = contexts.into_iter().map(|ldf| ldf.ldf).collect::<Vec<_>>();
        self.ldf.clone().with_context(contexts).into()
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "asof_join")]
    #[pyo3(signature = (other, left_on, right_on, left_by, right_by, allow_parallel, force_parallel, suffix, strategy, tolerance, tolerance_str))]
    fn join_asof(
        &self,
        other: Self,
        left_on: PyExpr,
        right_on: PyExpr,
        left_by: Option<Vec<&str>>,
        right_by: Option<Vec<&str>>,
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
                left_by: left_by.map(strings_to_smartstrings),
                right_by: right_by.map(strings_to_smartstrings),
                tolerance: tolerance.map(|t| t.0.into_static().unwrap()),
                tolerance_str: tolerance_str.map(|s| s.into()),
            }))
            .suffix(suffix)
            .finish()
            .into())
    }

    #[allow(clippy::too_many_arguments)]
    fn join(
        &self,
        other: Self,
        left_on: Vec<PyExpr>,
        right_on: Vec<PyExpr>,
        allow_parallel: bool,
        force_parallel: bool,
        how: Wrap<JoinType>,
        suffix: String,
        validate: Wrap<JoinValidation>,
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
            .validate(validate.0)
            .suffix(suffix)
            .finish()
            .into())
    }

    fn with_column(&mut self, expr: PyExpr) -> Self {
        let ldf = self.ldf.clone();
        ldf.with_column(expr.inner).into()
    }

    fn with_columns(&mut self, exprs: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.clone();
        ldf.with_columns(exprs.to_exprs()).into()
    }

    fn rename(&mut self, existing: Vec<String>, new: Vec<String>) -> Self {
        let ldf = self.ldf.clone();
        ldf.rename(existing, new).into()
    }

    fn reverse(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.reverse().into()
    }

    fn shift(&self, periods: i64) -> Self {
        let ldf = self.ldf.clone();
        ldf.shift(periods).into()
    }

    fn shift_and_fill(&self, periods: i64, fill_value: PyExpr) -> Self {
        let ldf = self.ldf.clone();
        ldf.shift_and_fill(periods, fill_value.inner).into()
    }

    fn fill_nan(&self, fill_value: PyExpr) -> Self {
        let ldf = self.ldf.clone();
        ldf.fill_nan(fill_value.inner).into()
    }

    fn min(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.min().into()
    }

    fn max(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.max().into()
    }

    fn sum(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.sum().into()
    }

    fn mean(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.mean().into()
    }

    fn std(&self, ddof: u8) -> Self {
        let ldf = self.ldf.clone();
        ldf.std(ddof).into()
    }

    fn var(&self, ddof: u8) -> Self {
        let ldf = self.ldf.clone();
        ldf.var(ddof).into()
    }

    fn median(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.median().into()
    }

    fn quantile(&self, quantile: PyExpr, interpolation: Wrap<QuantileInterpolOptions>) -> Self {
        let ldf = self.ldf.clone();
        ldf.quantile(quantile.inner, interpolation.0).into()
    }

    fn explode(&self, column: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.clone();
        let column = column.to_exprs();
        ldf.explode(column).into()
    }

    fn null_count(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.null_count().into()
    }

    #[pyo3(signature = (maintain_order, subset, keep))]
    fn unique(
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

    fn drop_nulls(&self, subset: Option<Vec<String>>) -> Self {
        let ldf = self.ldf.clone();
        ldf.drop_nulls(subset.map(|v| v.into_iter().map(|s| col(&s)).collect()))
            .into()
    }

    fn slice(&self, offset: i64, len: Option<IdxSize>) -> Self {
        let ldf = self.ldf.clone();
        ldf.slice(offset, len.unwrap_or(IdxSize::MAX)).into()
    }

    fn tail(&self, n: IdxSize) -> Self {
        let ldf = self.ldf.clone();
        ldf.tail(n).into()
    }

    #[pyo3(signature = (id_vars, value_vars, value_name, variable_name, streamable))]
    fn melt(
        &self,
        id_vars: Vec<String>,
        value_vars: Vec<String>,
        value_name: Option<String>,
        variable_name: Option<String>,
        streamable: bool,
    ) -> Self {
        let args = MeltArgs {
            id_vars: strings_to_smartstrings(id_vars),
            value_vars: strings_to_smartstrings(value_vars),
            value_name: value_name.map(|s| s.into()),
            variable_name: variable_name.map(|s| s.into()),
            streamable,
        };

        let ldf = self.ldf.clone();
        ldf.melt(args).into()
    }

    fn with_row_count(&self, name: &str, offset: Option<IdxSize>) -> Self {
        let ldf = self.ldf.clone();
        ldf.with_row_count(name, offset).into()
    }

    #[pyo3(signature = (lambda, predicate_pushdown, projection_pushdown, slice_pushdown, streamable, schema, validate_output))]
    #[allow(clippy::too_many_arguments)]
    fn map(
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

    fn drop(&self, columns: Vec<String>) -> Self {
        let ldf = self.ldf.clone();
        ldf.drop_columns(columns).into()
    }

    fn clone(&self) -> Self {
        self.ldf.clone().into()
    }

    fn columns(&self, py: Python) -> PyResult<PyObject> {
        let schema = self.get_schema()?;
        let iter = schema.iter_names().map(|s| s.as_str());
        Ok(PyList::new(py, iter).to_object(py))
    }

    fn dtypes(&self, py: Python) -> PyResult<PyObject> {
        let schema = self.get_schema()?;
        let iter = schema
            .iter_dtypes()
            .map(|dt| Wrap(dt.clone()).to_object(py));
        Ok(PyList::new(py, iter).to_object(py))
    }

    fn schema(&self, py: Python) -> PyResult<PyObject> {
        let schema = self.get_schema()?;
        let schema_dict = PyDict::new(py);

        schema.iter_fields().for_each(|fld| {
            schema_dict
                .set_item(fld.name().as_str(), Wrap(fld.data_type().clone()))
                .unwrap()
        });
        Ok(schema_dict.to_object(py))
    }

    fn unnest(&self, columns: Vec<String>) -> Self {
        self.ldf.clone().unnest(columns).into()
    }

    fn width(&self) -> PyResult<usize> {
        Ok(self.get_schema()?.len())
    }

    #[cfg(feature = "merge_sorted")]
    fn merge_sorted(&self, other: Self, key: &str) -> PyResult<Self> {
        let out = self
            .ldf
            .clone()
            .merge_sorted(other.ldf, key)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }
}
