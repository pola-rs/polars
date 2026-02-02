use std::collections::HashMap;
use std::ffi::CString;
use std::num::NonZeroUsize;

use arrow::ffi::export_iterator;
use either::Either;
use parking_lot::Mutex;
use polars::io::RowIndex;
use polars::time::*;
use polars_core::prelude::*;
#[cfg(feature = "parquet")]
use polars_parquet::arrow::write::StatisticsOptions;
use polars_plan::dsl::ScanSources;
use polars_plan::plans::{AExpr, HintIR, IR, Sorted};
use polars_utils::arena::{Arena, Node};
use polars_utils::python_function::PythonObject;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyCapsule, PyDict, PyDictMethods, PyList};

use super::{PyLazyFrame, PyOptFlags};
use crate::error::PyPolarsErr;
use crate::expr::ToExprs;
use crate::expr::datatype::PyDataTypeExpr;
use crate::expr::selector::PySelector;
use crate::interop::arrow::to_rust::pyarrow_schema_to_rust;
#[cfg(feature = "json")]
use crate::io::cloud_options::OptPyCloudOptions;
use crate::io::scan_options::PyScanOptions;
use crate::io::sink_options::PySinkOptions;
use crate::io::sink_output::PyFileSinkDestination;
use crate::lazyframe::visit::NodeTraverser;
use crate::prelude::*;
use crate::utils::{EnterPolarsExt, to_py_err};
use crate::{PyDataFrame, PyExpr, PyLazyGroupBy};

fn pyobject_to_first_path_and_scan_sources(
    obj: Py<PyAny>,
) -> PyResult<(Option<PlRefPath>, ScanSources)> {
    use crate::file::{PythonScanSourceInput, get_python_scan_source_input};
    Ok(match get_python_scan_source_input(obj, false)? {
        PythonScanSourceInput::Path(path) => (
            Some(path.clone()),
            ScanSources::Paths(FromIterator::from_iter([path])),
        ),
        PythonScanSourceInput::File(file) => (None, ScanSources::Files([file.into()].into())),
        PythonScanSourceInput::Buffer(buff) => (None, ScanSources::Buffers([buff].into())),
    })
}

fn post_opt_callback(
    lambda: &Py<PyAny>,
    root: Node,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    duration_since_start: Option<std::time::Duration>,
) -> PolarsResult<()> {
    Python::attach(|py| {
        let nt = NodeTraverser::new(root, std::mem::take(lp_arena), std::mem::take(expr_arena));

        // Get a copy of the arenas.
        let arenas = nt.get_arenas();

        // Pass the node visitor which allows the python callback to replace parts of the query plan.
        // Remove "cuda" or specify better once we have multiple post-opt callbacks.
        lambda
            .call1(py, (nt, duration_since_start.map(|t| t.as_nanos() as u64)))
            .map_err(|e| polars_err!(ComputeError: "'cuda' conversion failed: {}", e))?;

        // Unpack the arenas.
        // At this point the `nt` is useless.

        std::mem::swap(lp_arena, &mut *arenas.0.lock().unwrap());
        std::mem::swap(expr_arena, &mut *arenas.1.lock().unwrap());

        Ok(())
    })
}

#[pymethods]
#[allow(clippy::should_implement_trait)]
impl PyLazyFrame {
    #[staticmethod]
    #[cfg(feature = "json")]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        source, sources, infer_schema_length, schema, schema_overrides, batch_size, n_rows, low_memory, rechunk,
        row_index, ignore_errors, include_file_paths, cloud_options, credential_provider
    ))]
    fn new_from_ndjson(
        source: Option<Py<PyAny>>,
        sources: Wrap<ScanSources>,
        infer_schema_length: Option<usize>,
        schema: Option<Wrap<Schema>>,
        schema_overrides: Option<Wrap<Schema>>,
        batch_size: Option<NonZeroUsize>,
        n_rows: Option<usize>,
        low_memory: bool,
        rechunk: bool,
        row_index: Option<(String, IdxSize)>,
        ignore_errors: bool,
        include_file_paths: Option<String>,
        cloud_options: OptPyCloudOptions,
        credential_provider: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: name.into(),
            offset,
        });

        let sources = sources.0;
        let (first_path, sources) = match source {
            None => (sources.first_path().cloned(), sources),
            Some(source) => pyobject_to_first_path_and_scan_sources(source)?,
        };

        let mut r = LazyJsonLineReader::new_with_sources(sources);

        if let Some(first_path) = first_path {
            let first_path_url = first_path.as_str();

            let cloud_options = cloud_options.extract_opt_cloud_options(
                CloudScheme::from_path(first_path_url),
                credential_provider,
            )?;

            r = r.with_cloud_options(cloud_options);
        };

        let lf = r
            .with_infer_schema_length(infer_schema_length.and_then(NonZeroUsize::new))
            .with_batch_size(batch_size)
            .with_n_rows(n_rows)
            .low_memory(low_memory)
            .with_rechunk(rechunk)
            .with_schema(schema.map(|schema| Arc::new(schema.0)))
            .with_schema_overwrite(schema_overrides.map(|x| Arc::new(x.0)))
            .with_row_index(row_index)
            .with_ignore_errors(ignore_errors)
            .with_include_file_paths(include_file_paths.map(|x| x.into()))
            .finish()
            .map_err(PyPolarsErr::from)?;

        Ok(lf.into())
    }

    #[staticmethod]
    #[cfg(feature = "csv")]
    #[pyo3(signature = (source, sources, separator, has_header, ignore_errors, skip_rows, skip_lines, n_rows, cache, overwrite_dtype,
        low_memory, comment_prefix, quote_char, null_values, missing_utf8_is_empty_string,
        infer_schema_length, with_schema_modify, rechunk, skip_rows_after_header,
        encoding, row_index, try_parse_dates, eol_char, raise_if_empty, truncate_ragged_lines, decimal_comma, glob, schema,
        cloud_options, credential_provider, include_file_paths
    )
    )]
    fn new_from_csv(
        source: Option<Py<PyAny>>,
        sources: Wrap<ScanSources>,
        separator: &str,
        has_header: bool,
        ignore_errors: bool,
        skip_rows: usize,
        skip_lines: usize,
        n_rows: Option<usize>,
        cache: bool,
        overwrite_dtype: Option<Vec<(PyBackedStr, Wrap<DataType>)>>,
        low_memory: bool,
        comment_prefix: Option<&str>,
        quote_char: Option<&str>,
        null_values: Option<Wrap<NullValues>>,
        missing_utf8_is_empty_string: bool,
        infer_schema_length: Option<usize>,
        with_schema_modify: Option<Py<PyAny>>,
        rechunk: bool,
        skip_rows_after_header: usize,
        encoding: Wrap<CsvEncoding>,
        row_index: Option<(String, IdxSize)>,
        try_parse_dates: bool,
        eol_char: &str,
        raise_if_empty: bool,
        truncate_ragged_lines: bool,
        decimal_comma: bool,
        glob: bool,
        schema: Option<Wrap<Schema>>,
        cloud_options: OptPyCloudOptions,
        credential_provider: Option<Py<PyAny>>,
        include_file_paths: Option<String>,
    ) -> PyResult<Self> {
        let null_values = null_values.map(|w| w.0);
        let quote_char = quote_char.and_then(|s| s.as_bytes().first()).copied();
        let separator = separator
            .as_bytes()
            .first()
            .ok_or_else(|| polars_err!(InvalidOperation: "`separator` cannot be empty"))
            .copied()
            .map_err(PyPolarsErr::from)?;
        let eol_char = eol_char
            .as_bytes()
            .first()
            .ok_or_else(|| polars_err!(InvalidOperation: "`eol_char` cannot be empty"))
            .copied()
            .map_err(PyPolarsErr::from)?;
        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: name.into(),
            offset,
        });

        let overwrite_dtype = overwrite_dtype.map(|overwrite_dtype| {
            overwrite_dtype
                .into_iter()
                .map(|(name, dtype)| Field::new((&*name).into(), dtype.0))
                .collect::<Schema>()
        });

        let sources = sources.0;
        let (first_path, sources) = match source {
            None => (sources.first_path().cloned(), sources),
            Some(source) => pyobject_to_first_path_and_scan_sources(source)?,
        };

        let mut r = LazyCsvReader::new_with_sources(sources);

        if let Some(first_path) = first_path {
            let first_path_url = first_path.as_str();
            let cloud_options = cloud_options.extract_opt_cloud_options(
                CloudScheme::from_path(first_path_url),
                credential_provider,
            )?;
            r = r.with_cloud_options(cloud_options);
        }

        let mut r = r
            .with_infer_schema_length(infer_schema_length)
            .with_separator(separator)
            .with_has_header(has_header)
            .with_ignore_errors(ignore_errors)
            .with_skip_rows(skip_rows)
            .with_skip_lines(skip_lines)
            .with_n_rows(n_rows)
            .with_cache(cache)
            .with_dtype_overwrite(overwrite_dtype.map(Arc::new))
            .with_schema(schema.map(|schema| Arc::new(schema.0)))
            .with_low_memory(low_memory)
            .with_comment_prefix(comment_prefix.map(|x| x.into()))
            .with_quote_char(quote_char)
            .with_eol_char(eol_char)
            .with_rechunk(rechunk)
            .with_skip_rows_after_header(skip_rows_after_header)
            .with_encoding(encoding.0)
            .with_row_index(row_index)
            .with_try_parse_dates(try_parse_dates)
            .with_null_values(null_values)
            .with_missing_is_null(!missing_utf8_is_empty_string)
            .with_truncate_ragged_lines(truncate_ragged_lines)
            .with_decimal_comma(decimal_comma)
            .with_glob(glob)
            .with_raise_if_empty(raise_if_empty)
            .with_include_file_paths(include_file_paths.map(|x| x.into()));

        if let Some(lambda) = with_schema_modify {
            let f = |schema: Schema| {
                let iter = schema.iter_names().map(|s| s.as_str());
                Python::attach(|py| {
                    let names = PyList::new(py, iter).unwrap();

                    let out = lambda.call1(py, (names,)).expect("python function failed");
                    let new_names = out
                        .extract::<Vec<String>>(py)
                        .expect("python function should return List[str]");
                    polars_ensure!(new_names.len() == schema.len(),
                        ShapeMismatch: "The length of the new names list should be equal to or less than the original column length",
                    );
                    Ok(schema
                        .iter_values()
                        .zip(new_names)
                        .map(|(dtype, name)| Field::new(name.into(), dtype.clone()))
                        .collect())
                })
            };
            r = r.with_schema_modify(f).map_err(PyPolarsErr::from)?
        }

        Ok(r.finish().map_err(PyPolarsErr::from)?.into())
    }

    #[cfg(feature = "parquet")]
    #[staticmethod]
    #[pyo3(signature = (
        sources, schema, scan_options, parallel, low_memory, use_statistics
    ))]
    fn new_from_parquet(
        sources: Wrap<ScanSources>,
        schema: Option<Wrap<Schema>>,
        scan_options: PyScanOptions,
        parallel: Wrap<ParallelStrategy>,
        low_memory: bool,
        use_statistics: bool,
    ) -> PyResult<Self> {
        use crate::utils::to_py_err;

        let parallel = parallel.0;

        let options = ParquetOptions {
            schema: schema.map(|x| Arc::new(x.0)),
            parallel,
            low_memory,
            use_statistics,
        };

        let sources = sources.0;
        let first_path = sources.first_path();

        let unified_scan_args =
            scan_options.extract_unified_scan_args(first_path.and_then(|x| x.scheme()))?;

        let lf: LazyFrame = DslBuilder::scan_parquet(sources, options, unified_scan_args)
            .map_err(to_py_err)?
            .build()
            .into();

        Ok(lf.into())
    }

    #[cfg(feature = "ipc")]
    #[staticmethod]
    #[pyo3(signature = (sources, record_batch_statistics, scan_options))]
    fn new_from_ipc(
        sources: Wrap<ScanSources>,
        record_batch_statistics: bool,
        scan_options: PyScanOptions,
    ) -> PyResult<Self> {
        let options = IpcScanOptions {
            record_batch_statistics,
            checked: Default::default(),
        };

        let sources = sources.0;
        let first_path = sources.first_path().cloned();

        let unified_scan_args =
            scan_options.extract_unified_scan_args(first_path.as_ref().and_then(|x| x.scheme()))?;

        let lf = LazyFrame::scan_ipc_sources(sources, options, unified_scan_args)
            .map_err(PyPolarsErr::from)?;
        Ok(lf.into())
    }

    #[cfg(feature = "scan_lines")]
    #[staticmethod]
    #[pyo3(signature = (sources, scan_options, name))]
    fn new_from_scan_lines(
        sources: Wrap<ScanSources>,
        scan_options: PyScanOptions,
        name: PyBackedStr,
    ) -> PyResult<Self> {
        let sources = sources.0;
        let first_path = sources.first_path();

        let unified_scan_args =
            scan_options.extract_unified_scan_args(first_path.and_then(|x| x.scheme()))?;

        let dsl: DslPlan = DslBuilder::scan_lines(sources, unified_scan_args, (&*name).into())
            .map_err(to_py_err)?
            .build();
        let lf: LazyFrame = dsl.into();

        Ok(lf.into())
    }

    #[staticmethod]
    #[pyo3(signature = (
        dataset_object
    ))]
    fn new_from_dataset_object(dataset_object: Py<PyAny>) -> PyResult<Self> {
        let lf =
            LazyFrame::from(DslBuilder::scan_python_dataset(PythonObject(dataset_object)).build())
                .into();

        Ok(lf)
    }

    #[staticmethod]
    fn scan_from_python_function_arrow_schema(
        schema: &Bound<'_, PyList>,
        scan_fn: Py<PyAny>,
        pyarrow: bool,
        validate_schema: bool,
        is_pure: bool,
    ) -> PyResult<Self> {
        let schema = Arc::new(pyarrow_schema_to_rust(schema)?);

        Ok(LazyFrame::scan_from_python_function(
            Either::Right(schema),
            scan_fn,
            pyarrow,
            validate_schema,
            is_pure,
        )
        .into())
    }

    #[staticmethod]
    fn scan_from_python_function_pl_schema(
        schema: Vec<(PyBackedStr, Wrap<DataType>)>,
        scan_fn: Py<PyAny>,
        pyarrow: bool,
        validate_schema: bool,
        is_pure: bool,
    ) -> PyResult<Self> {
        let schema = Arc::new(Schema::from_iter(
            schema
                .into_iter()
                .map(|(name, dt)| Field::new((&*name).into(), dt.0)),
        ));
        Ok(LazyFrame::scan_from_python_function(
            Either::Right(schema),
            scan_fn,
            pyarrow,
            validate_schema,
            is_pure,
        )
        .into())
    }

    #[staticmethod]
    fn scan_from_python_function_schema_function(
        schema_fn: Py<PyAny>,
        scan_fn: Py<PyAny>,
        validate_schema: bool,
        is_pure: bool,
    ) -> PyResult<Self> {
        Ok(LazyFrame::scan_from_python_function(
            Either::Left(schema_fn),
            scan_fn,
            false,
            validate_schema,
            is_pure,
        )
        .into())
    }

    fn describe_plan(&self, py: Python) -> PyResult<String> {
        py.enter_polars(|| self.ldf.read().describe_plan())
    }

    fn describe_optimized_plan(&self, py: Python) -> PyResult<String> {
        py.enter_polars(|| self.ldf.read().describe_optimized_plan())
    }

    fn describe_plan_tree(&self, py: Python) -> PyResult<String> {
        py.enter_polars(|| self.ldf.read().describe_plan_tree())
    }

    fn describe_optimized_plan_tree(&self, py: Python) -> PyResult<String> {
        py.enter_polars(|| self.ldf.read().describe_optimized_plan_tree())
    }

    fn to_dot(&self, py: Python<'_>, optimized: bool) -> PyResult<String> {
        py.enter_polars(|| self.ldf.read().to_dot(optimized))
    }

    #[cfg(feature = "new_streaming")]
    fn to_dot_streaming_phys(&self, py: Python, optimized: bool) -> PyResult<String> {
        py.enter_polars(|| self.ldf.read().to_dot_streaming_phys(optimized))
    }

    fn sort(
        &self,
        by_column: &str,
        descending: bool,
        nulls_last: bool,
        maintain_order: bool,
        multithreaded: bool,
    ) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.sort(
            [by_column],
            SortMultipleOptions {
                descending: vec![descending],
                nulls_last: vec![nulls_last],
                multithreaded,
                maintain_order,
                limit: None,
            },
        )
        .into()
    }

    fn sort_by_exprs(
        &self,
        by: Vec<PyExpr>,
        descending: Vec<bool>,
        nulls_last: Vec<bool>,
        maintain_order: bool,
        multithreaded: bool,
    ) -> Self {
        let ldf = self.ldf.read().clone();
        let exprs = by.to_exprs();
        ldf.sort_by_exprs(
            exprs,
            SortMultipleOptions {
                descending,
                nulls_last,
                maintain_order,
                multithreaded,
                limit: None,
            },
        )
        .into()
    }

    fn top_k(&self, k: IdxSize, by: Vec<PyExpr>, reverse: Vec<bool>) -> Self {
        let ldf = self.ldf.read().clone();
        let exprs = by.to_exprs();
        ldf.top_k(
            k,
            exprs,
            SortMultipleOptions::new().with_order_descending_multi(reverse),
        )
        .into()
    }

    fn bottom_k(&self, k: IdxSize, by: Vec<PyExpr>, reverse: Vec<bool>) -> Self {
        let ldf = self.ldf.read().clone();
        let exprs = by.to_exprs();
        ldf.bottom_k(
            k,
            exprs,
            SortMultipleOptions::new().with_order_descending_multi(reverse),
        )
        .into()
    }

    fn cache(&self) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.cache().into()
    }

    #[pyo3(signature = (optflags))]
    fn with_optimizations(&self, optflags: PyOptFlags) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.with_optimizations(optflags.inner.into_inner()).into()
    }

    #[pyo3(signature = (lambda_post_opt))]
    fn profile(
        &self,
        py: Python<'_>,
        lambda_post_opt: Option<Py<PyAny>>,
    ) -> PyResult<(PyDataFrame, PyDataFrame)> {
        let (df, time_df) = py.enter_polars(|| {
            let ldf = self.ldf.read().clone();
            if let Some(lambda) = lambda_post_opt {
                ldf._profile_post_opt(|root, lp_arena, expr_arena, duration_since_start| {
                    post_opt_callback(&lambda, root, lp_arena, expr_arena, duration_since_start)
                })
            } else {
                ldf.profile()
            }
        })?;
        Ok((df.into(), time_df.into()))
    }

    #[pyo3(signature = (engine, lambda_post_opt))]
    fn collect(
        &self,
        py: Python<'_>,
        engine: Wrap<Engine>,
        lambda_post_opt: Option<Py<PyAny>>,
    ) -> PyResult<PyDataFrame> {
        py.enter_polars_df(|| {
            let ldf = self.ldf.read().clone();
            if let Some(lambda) = lambda_post_opt {
                ldf._collect_post_opt(|root, lp_arena, expr_arena, _| {
                    post_opt_callback(&lambda, root, lp_arena, expr_arena, None)
                })
            } else {
                ldf.collect_with_engine(engine.0)
            }
        })
    }

    #[cfg(feature = "async")]
    #[pyo3(signature = (engine, lambda))]
    fn collect_with_callback(
        &self,
        py: Python<'_>,
        engine: Wrap<Engine>,
        lambda: Py<PyAny>,
    ) -> PyResult<()> {
        py.enter_polars_ok(|| {
            let ldf = self.ldf.read().clone();

            // We use a tokio spawn_blocking here as it has a high blocking
            // thread pool limit.
            polars_io::pl_async::get_runtime().spawn_blocking(move || {
                let result = ldf
                    .collect_with_engine(engine.0)
                    .map(PyDataFrame::new)
                    .map_err(PyPolarsErr::from);

                Python::attach(|py| match result {
                    Ok(df) => {
                        lambda.call1(py, (df,)).map_err(|err| err.restore(py)).ok();
                    },
                    Err(err) => {
                        lambda
                            .call1(py, (PyErr::from(err),))
                            .map_err(|err| err.restore(py))
                            .ok();
                    },
                });
            });
        })
    }

    #[cfg(feature = "async")]
    fn collect_batches(
        &self,
        py: Python<'_>,
        engine: Wrap<Engine>,
        maintain_order: bool,
        chunk_size: Option<NonZeroUsize>,
        lazy: bool,
    ) -> PyResult<PyCollectBatches> {
        py.enter_polars(|| {
            let ldf = self.ldf.read().clone();

            let collect_batches = ldf
                .clone()
                .collect_batches(engine.0, maintain_order, chunk_size, lazy)
                .map_err(PyPolarsErr::from)?;

            PyResult::Ok(PyCollectBatches {
                inner: Arc::new(Mutex::new(collect_batches)),
                ldf,
            })
        })
    }

    #[cfg(feature = "parquet")]
    #[pyo3(signature = (
        target, sink_options, compression, compression_level, statistics, row_group_size, data_page_size,
        metadata, arrow_schema
    ))]
    fn sink_parquet(
        &self,
        py: Python<'_>,
        target: PyFileSinkDestination,
        sink_options: PySinkOptions,
        compression: &str,
        compression_level: Option<i32>,
        statistics: Wrap<StatisticsOptions>,
        row_group_size: Option<usize>,
        data_page_size: Option<usize>,
        metadata: Wrap<Option<KeyValueMetadata>>,
        arrow_schema: Option<Wrap<ArrowSchema>>,
    ) -> PyResult<PyLazyFrame> {
        let compression = parse_parquet_compression(compression, compression_level)?;

        let options = ParquetWriteOptions {
            compression,
            statistics: statistics.0,
            row_group_size,
            data_page_size,
            key_value_metadata: metadata.0,
            arrow_schema: arrow_schema.map(|x| Arc::new(x.0)),
        };

        let target = target.extract_file_sink_destination()?;
        let unified_sink_args = sink_options.extract_unified_sink_args(target.cloud_scheme())?;

        py.enter_polars(|| {
            self.ldf
                .read()
                .clone()
                .sink(
                    target,
                    FileWriteFormat::Parquet(Arc::new(options)),
                    unified_sink_args,
                )
                .into()
        })
        .map(Into::into)
        .map_err(Into::into)
    }

    #[cfg(feature = "ipc")]
    #[pyo3(signature = (
        target, sink_options, compression, compat_level, record_batch_size, record_batch_statistics
    ))]
    fn sink_ipc(
        &self,
        py: Python<'_>,
        target: PyFileSinkDestination,
        sink_options: PySinkOptions,
        compression: Wrap<Option<IpcCompression>>,
        compat_level: PyCompatLevel,
        record_batch_size: Option<usize>,
        record_batch_statistics: bool,
    ) -> PyResult<PyLazyFrame> {
        let options = IpcWriterOptions {
            compression: compression.0,
            compat_level: compat_level.0,
            record_batch_size,
            record_batch_statistics,
        };

        let target = target.extract_file_sink_destination()?;
        let unified_sink_args = sink_options.extract_unified_sink_args(target.cloud_scheme())?;

        py.enter_polars(|| {
            self.ldf
                .read()
                .clone()
                .sink(target, FileWriteFormat::Ipc(options), unified_sink_args)
                .into()
        })
        .map(Into::into)
        .map_err(Into::into)
    }

    #[cfg(feature = "csv")]
    #[pyo3(signature = (
        target, sink_options, include_bom, compression, compression_level, check_extension,
        include_header, separator, line_terminator, quote_char, batch_size, datetime_format,
        date_format, time_format, float_scientific, float_precision, decimal_comma, null_value,
        quote_style
    ))]
    fn sink_csv(
        &self,
        py: Python<'_>,
        target: PyFileSinkDestination,
        sink_options: PySinkOptions,
        include_bom: bool,
        compression: &str,
        compression_level: Option<u32>,
        check_extension: bool,
        include_header: bool,
        separator: u8,
        line_terminator: Wrap<PlSmallStr>,
        quote_char: u8,
        batch_size: NonZeroUsize,
        datetime_format: Option<Wrap<PlSmallStr>>,
        date_format: Option<Wrap<PlSmallStr>>,
        time_format: Option<Wrap<PlSmallStr>>,
        float_scientific: Option<bool>,
        float_precision: Option<usize>,
        decimal_comma: bool,
        null_value: Option<Wrap<PlSmallStr>>,
        quote_style: Option<Wrap<QuoteStyle>>,
    ) -> PyResult<PyLazyFrame> {
        let quote_style = quote_style.map_or(QuoteStyle::default(), |wrap| wrap.0);
        let null_value = null_value
            .map(|x| x.0)
            .unwrap_or(SerializeOptions::default().null);

        let serialize_options = SerializeOptions {
            date_format: date_format.map(|x| x.0),
            time_format: time_format.map(|x| x.0),
            datetime_format: datetime_format.map(|x| x.0),
            float_scientific,
            float_precision,
            decimal_comma,
            separator,
            quote_char,
            null: null_value,
            line_terminator: line_terminator.0,
            quote_style,
        };

        let options = CsvWriterOptions {
            include_bom,
            compression: ExternalCompression::try_from(compression, compression_level)
                .map_err(PyPolarsErr::from)?,
            check_extension,
            include_header,
            batch_size,
            serialize_options: serialize_options.into(),
        };

        let target = target.extract_file_sink_destination()?;
        let unified_sink_args = sink_options.extract_unified_sink_args(target.cloud_scheme())?;

        py.enter_polars(|| {
            self.ldf
                .read()
                .clone()
                .sink(target, FileWriteFormat::Csv(options), unified_sink_args)
                .into()
        })
        .map(Into::into)
        .map_err(Into::into)
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "json")]
    #[pyo3(signature = (target, compression, compression_level, check_extension, sink_options))]
    fn sink_ndjson(
        &self,
        py: Python<'_>,
        target: PyFileSinkDestination,
        compression: &str,
        compression_level: Option<u32>,
        check_extension: bool,
        sink_options: PySinkOptions,
    ) -> PyResult<PyLazyFrame> {
        let options = NDJsonWriterOptions {
            compression: ExternalCompression::try_from(compression, compression_level)
                .map_err(PyPolarsErr::from)?,
            check_extension,
        };

        let target = target.extract_file_sink_destination()?;
        let unified_sink_args = sink_options.extract_unified_sink_args(target.cloud_scheme())?;

        py.enter_polars(|| {
            self.ldf
                .read()
                .clone()
                .sink(target, FileWriteFormat::NDJson(options), unified_sink_args)
                .into()
        })
        .map(Into::into)
        .map_err(Into::into)
    }

    #[pyo3(signature = (function, maintain_order, chunk_size))]
    pub fn sink_batches(
        &self,
        py: Python<'_>,
        function: Py<PyAny>,
        maintain_order: bool,
        chunk_size: Option<NonZeroUsize>,
    ) -> PyResult<PyLazyFrame> {
        let ldf = self.ldf.read().clone();
        py.enter_polars(|| {
            ldf.sink_batches(
                PlanCallback::new_python(PythonObject(function)),
                maintain_order,
                chunk_size,
            )
        })
        .map(Into::into)
        .map_err(Into::into)
    }

    fn filter(&self, predicate: PyExpr) -> Self {
        self.ldf.read().clone().filter(predicate.inner).into()
    }

    fn remove(&self, predicate: PyExpr) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.remove(predicate.inner).into()
    }

    fn select(&self, exprs: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.read().clone();
        let exprs = exprs.to_exprs();
        ldf.select(exprs).into()
    }

    fn select_seq(&self, exprs: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.read().clone();
        let exprs = exprs.to_exprs();
        ldf.select_seq(exprs).into()
    }

    fn group_by(&self, by: Vec<PyExpr>, maintain_order: bool) -> PyLazyGroupBy {
        let ldf = self.ldf.read().clone();
        let by = by.to_exprs();
        let lazy_gb = if maintain_order {
            ldf.group_by_stable(by)
        } else {
            ldf.group_by(by)
        };

        PyLazyGroupBy { lgb: Some(lazy_gb) }
    }

    fn rolling(
        &self,
        index_column: PyExpr,
        period: &str,
        offset: &str,
        closed: Wrap<ClosedWindow>,
        by: Vec<PyExpr>,
    ) -> PyResult<PyLazyGroupBy> {
        let closed_window = closed.0;
        let ldf = self.ldf.read().clone();
        let by = by
            .into_iter()
            .map(|pyexpr| pyexpr.inner)
            .collect::<Vec<_>>();
        let lazy_gb = ldf.rolling(
            index_column.inner,
            by,
            RollingGroupOptions {
                index_column: "".into(),
                period: Duration::try_parse(period).map_err(PyPolarsErr::from)?,
                offset: Duration::try_parse(offset).map_err(PyPolarsErr::from)?,
                closed_window,
            },
        );

        Ok(PyLazyGroupBy { lgb: Some(lazy_gb) })
    }

    fn group_by_dynamic(
        &self,
        index_column: PyExpr,
        every: &str,
        period: &str,
        offset: &str,
        label: Wrap<Label>,
        include_boundaries: bool,
        closed: Wrap<ClosedWindow>,
        group_by: Vec<PyExpr>,
        start_by: Wrap<StartBy>,
    ) -> PyResult<PyLazyGroupBy> {
        let closed_window = closed.0;
        let group_by = group_by
            .into_iter()
            .map(|pyexpr| pyexpr.inner)
            .collect::<Vec<_>>();
        let ldf = self.ldf.read().clone();
        let lazy_gb = ldf.group_by_dynamic(
            index_column.inner,
            group_by,
            DynamicGroupOptions {
                every: Duration::try_parse(every).map_err(PyPolarsErr::from)?,
                period: Duration::try_parse(period).map_err(PyPolarsErr::from)?,
                offset: Duration::try_parse(offset).map_err(PyPolarsErr::from)?,
                label: label.0,
                include_boundaries,
                closed_window,
                start_by: start_by.0,
                ..Default::default()
            },
        );

        Ok(PyLazyGroupBy { lgb: Some(lazy_gb) })
    }

    fn with_context(&self, contexts: Vec<Self>) -> Self {
        let contexts = contexts
            .into_iter()
            .map(|ldf| ldf.ldf.into_inner())
            .collect::<Vec<_>>();
        self.ldf.read().clone().with_context(contexts).into()
    }

    #[cfg(feature = "asof_join")]
    #[pyo3(signature = (other, left_on, right_on, left_by, right_by, allow_parallel, force_parallel, suffix, strategy, tolerance, tolerance_str, coalesce, allow_eq, check_sortedness))]
    fn join_asof(
        &self,
        other: Self,
        left_on: PyExpr,
        right_on: PyExpr,
        left_by: Option<Vec<PyBackedStr>>,
        right_by: Option<Vec<PyBackedStr>>,
        allow_parallel: bool,
        force_parallel: bool,
        suffix: String,
        strategy: Wrap<AsofStrategy>,
        tolerance: Option<Wrap<AnyValue<'_>>>,
        tolerance_str: Option<String>,
        coalesce: bool,
        allow_eq: bool,
        check_sortedness: bool,
    ) -> PyResult<Self> {
        let coalesce = if coalesce {
            JoinCoalesce::CoalesceColumns
        } else {
            JoinCoalesce::KeepColumns
        };
        let ldf = self.ldf.read().clone();
        let other = other.ldf.into_inner();
        let left_on = left_on.inner;
        let right_on = right_on.inner;
        Ok(ldf
            .join_builder()
            .with(other)
            .left_on([left_on])
            .right_on([right_on])
            .allow_parallel(allow_parallel)
            .force_parallel(force_parallel)
            .coalesce(coalesce)
            .how(JoinType::AsOf(Box::new(AsOfOptions {
                strategy: strategy.0,
                left_by: left_by.map(strings_to_pl_smallstr),
                right_by: right_by.map(strings_to_pl_smallstr),
                tolerance: tolerance.map(|t| {
                    let av = t.0.into_static();
                    let dtype = av.dtype();
                    Scalar::new(dtype, av)
                }),
                tolerance_str: tolerance_str.map(|s| s.into()),
                allow_eq,
                check_sortedness,
            })))
            .suffix(suffix)
            .finish()
            .into())
    }

    #[pyo3(signature = (other, left_on, right_on, allow_parallel, force_parallel, nulls_equal, how, suffix, validate, maintain_order, coalesce=None))]
    fn join(
        &self,
        other: Self,
        left_on: Vec<PyExpr>,
        right_on: Vec<PyExpr>,
        allow_parallel: bool,
        force_parallel: bool,
        nulls_equal: bool,
        how: Wrap<JoinType>,
        suffix: String,
        validate: Wrap<JoinValidation>,
        maintain_order: Wrap<MaintainOrderJoin>,
        coalesce: Option<bool>,
    ) -> PyResult<Self> {
        let coalesce = match coalesce {
            None => JoinCoalesce::JoinSpecific,
            Some(true) => JoinCoalesce::CoalesceColumns,
            Some(false) => JoinCoalesce::KeepColumns,
        };
        let ldf = self.ldf.read().clone();
        let other = other.ldf.into_inner();
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
            .join_nulls(nulls_equal)
            .how(how.0)
            .suffix(suffix)
            .validate(validate.0)
            .coalesce(coalesce)
            .maintain_order(maintain_order.0)
            .finish()
            .into())
    }

    fn join_where(&self, other: Self, predicates: Vec<PyExpr>, suffix: String) -> PyResult<Self> {
        let ldf = self.ldf.read().clone();
        let other = other.ldf.into_inner();

        let predicates = predicates.to_exprs();

        Ok(ldf
            .join_builder()
            .with(other)
            .suffix(suffix)
            .join_where(predicates)
            .into())
    }

    fn with_columns(&self, exprs: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.with_columns(exprs.to_exprs()).into()
    }

    fn with_columns_seq(&self, exprs: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.with_columns_seq(exprs.to_exprs()).into()
    }

    fn match_to_schema<'py>(
        &self,
        schema: Wrap<Schema>,
        missing_columns: &Bound<'py, PyAny>,
        missing_struct_fields: &Bound<'py, PyAny>,
        extra_columns: Wrap<ExtraColumnsPolicy>,
        extra_struct_fields: &Bound<'py, PyAny>,
        integer_cast: &Bound<'py, PyAny>,
        float_cast: &Bound<'py, PyAny>,
    ) -> PyResult<Self> {
        fn parse_missing_columns<'py>(
            schema: &Schema,
            missing_columns: &Bound<'py, PyAny>,
        ) -> PyResult<Vec<MissingColumnsPolicyOrExpr>> {
            let mut out = Vec::with_capacity(schema.len());
            if let Ok(policy) = missing_columns.extract::<Wrap<MissingColumnsPolicyOrExpr>>() {
                out.extend(std::iter::repeat_n(policy.0, schema.len()));
            } else if let Ok(dict) = missing_columns.cast::<PyDict>() {
                out.extend(std::iter::repeat_n(
                    MissingColumnsPolicyOrExpr::Raise,
                    schema.len(),
                ));
                for (key, value) in dict.iter() {
                    let key = key.extract::<String>()?;
                    let value = value.extract::<Wrap<MissingColumnsPolicyOrExpr>>()?;
                    out[schema.try_index_of(&key).map_err(to_py_err)?] = value.0;
                }
            } else {
                return Err(PyTypeError::new_err("Invalid value for `missing_columns`"));
            }
            Ok(out)
        }
        fn parse_missing_struct_fields<'py>(
            schema: &Schema,
            missing_struct_fields: &Bound<'py, PyAny>,
        ) -> PyResult<Vec<MissingColumnsPolicy>> {
            let mut out = Vec::with_capacity(schema.len());
            if let Ok(policy) = missing_struct_fields.extract::<Wrap<MissingColumnsPolicy>>() {
                out.extend(std::iter::repeat_n(policy.0, schema.len()));
            } else if let Ok(dict) = missing_struct_fields.cast::<PyDict>() {
                out.extend(std::iter::repeat_n(
                    MissingColumnsPolicy::Raise,
                    schema.len(),
                ));
                for (key, value) in dict.iter() {
                    let key = key.extract::<String>()?;
                    let value = value.extract::<Wrap<MissingColumnsPolicy>>()?;
                    out[schema.try_index_of(&key).map_err(to_py_err)?] = value.0;
                }
            } else {
                return Err(PyTypeError::new_err(
                    "Invalid value for `missing_struct_fields`",
                ));
            }
            Ok(out)
        }
        fn parse_extra_struct_fields<'py>(
            schema: &Schema,
            extra_struct_fields: &Bound<'py, PyAny>,
        ) -> PyResult<Vec<ExtraColumnsPolicy>> {
            let mut out = Vec::with_capacity(schema.len());
            if let Ok(policy) = extra_struct_fields.extract::<Wrap<ExtraColumnsPolicy>>() {
                out.extend(std::iter::repeat_n(policy.0, schema.len()));
            } else if let Ok(dict) = extra_struct_fields.cast::<PyDict>() {
                out.extend(std::iter::repeat_n(ExtraColumnsPolicy::Raise, schema.len()));
                for (key, value) in dict.iter() {
                    let key = key.extract::<String>()?;
                    let value = value.extract::<Wrap<ExtraColumnsPolicy>>()?;
                    out[schema.try_index_of(&key).map_err(to_py_err)?] = value.0;
                }
            } else {
                return Err(PyTypeError::new_err(
                    "Invalid value for `extra_struct_fields`",
                ));
            }
            Ok(out)
        }
        fn parse_cast<'py>(
            schema: &Schema,
            cast: &Bound<'py, PyAny>,
        ) -> PyResult<Vec<UpcastOrForbid>> {
            let mut out = Vec::with_capacity(schema.len());
            if let Ok(policy) = cast.extract::<Wrap<UpcastOrForbid>>() {
                out.extend(std::iter::repeat_n(policy.0, schema.len()));
            } else if let Ok(dict) = cast.cast::<PyDict>() {
                out.extend(std::iter::repeat_n(UpcastOrForbid::Forbid, schema.len()));
                for (key, value) in dict.iter() {
                    let key = key.extract::<String>()?;
                    let value = value.extract::<Wrap<UpcastOrForbid>>()?;
                    out[schema.try_index_of(&key).map_err(to_py_err)?] = value.0;
                }
            } else {
                return Err(PyTypeError::new_err(
                    "Invalid value for `integer_cast` / `float_cast`",
                ));
            }
            Ok(out)
        }

        let missing_columns = parse_missing_columns(&schema.0, missing_columns)?;
        let missing_struct_fields = parse_missing_struct_fields(&schema.0, missing_struct_fields)?;
        let extra_struct_fields = parse_extra_struct_fields(&schema.0, extra_struct_fields)?;
        let integer_cast = parse_cast(&schema.0, integer_cast)?;
        let float_cast = parse_cast(&schema.0, float_cast)?;

        let per_column = (0..schema.0.len())
            .map(|i| MatchToSchemaPerColumn {
                missing_columns: missing_columns[i].clone(),
                missing_struct_fields: missing_struct_fields[i],
                extra_struct_fields: extra_struct_fields[i],
                integer_cast: integer_cast[i],
                float_cast: float_cast[i],
            })
            .collect();

        let ldf = self.ldf.read().clone();
        Ok(ldf
            .match_to_schema(Arc::new(schema.0), per_column, extra_columns.0)
            .into())
    }

    fn pipe_with_schema(&self, callback: Py<PyAny>) -> Self {
        let ldf = self.ldf.read().clone();
        let function = PythonObject(callback);
        ldf.pipe_with_schema(PlanCallback::new_python(function))
            .into()
    }

    fn rename(&self, existing: Vec<String>, new: Vec<String>, strict: bool) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.rename(existing, new, strict).into()
    }

    fn reverse(&self) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.reverse().into()
    }

    #[pyo3(signature = (n, fill_value=None))]
    fn shift(&self, n: PyExpr, fill_value: Option<PyExpr>) -> Self {
        let lf = self.ldf.read().clone();
        let out = match fill_value {
            Some(v) => lf.shift_and_fill(n.inner, v.inner),
            None => lf.shift(n.inner),
        };
        out.into()
    }

    fn fill_nan(&self, fill_value: PyExpr) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.fill_nan(fill_value.inner).into()
    }

    fn min(&self) -> Self {
        let ldf = self.ldf.read().clone();
        let out = ldf.min();
        out.into()
    }

    fn max(&self) -> Self {
        let ldf = self.ldf.read().clone();
        let out = ldf.max();
        out.into()
    }

    fn sum(&self) -> Self {
        let ldf = self.ldf.read().clone();
        let out = ldf.sum();
        out.into()
    }

    fn mean(&self) -> Self {
        let ldf = self.ldf.read().clone();
        let out = ldf.mean();
        out.into()
    }

    fn std(&self, ddof: u8) -> Self {
        let ldf = self.ldf.read().clone();
        let out = ldf.std(ddof);
        out.into()
    }

    fn var(&self, ddof: u8) -> Self {
        let ldf = self.ldf.read().clone();
        let out = ldf.var(ddof);
        out.into()
    }

    fn median(&self) -> Self {
        let ldf = self.ldf.read().clone();
        let out = ldf.median();
        out.into()
    }

    fn quantile(&self, quantile: PyExpr, interpolation: Wrap<QuantileMethod>) -> Self {
        let ldf = self.ldf.read().clone();
        let out = ldf.quantile(quantile.inner, interpolation.0);
        out.into()
    }

    fn explode(&self, subset: PySelector, empty_as_null: bool, keep_nulls: bool) -> Self {
        self.ldf
            .read()
            .clone()
            .explode(
                subset.inner,
                ExplodeOptions {
                    empty_as_null,
                    keep_nulls,
                },
            )
            .into()
    }

    fn null_count(&self) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.null_count().into()
    }

    #[pyo3(signature = (maintain_order, subset, keep))]
    fn unique(
        &self,
        maintain_order: bool,
        subset: Option<Vec<PyExpr>>,
        keep: Wrap<UniqueKeepStrategy>,
    ) -> Self {
        let ldf = self.ldf.read().clone();
        let subset = subset.map(|exprs| exprs.into_iter().map(|e| e.inner).collect());
        match maintain_order {
            true => ldf.unique_stable_generic(subset, keep.0),
            false => ldf.unique_generic(subset, keep.0),
        }
        .into()
    }

    fn drop_nans(&self, subset: Option<PySelector>) -> Self {
        self.ldf
            .read()
            .clone()
            .drop_nans(subset.map(|e| e.inner))
            .into()
    }

    fn drop_nulls(&self, subset: Option<PySelector>) -> Self {
        self.ldf
            .read()
            .clone()
            .drop_nulls(subset.map(|e| e.inner))
            .into()
    }

    #[pyo3(signature = (offset, len=None))]
    fn slice(&self, offset: i64, len: Option<IdxSize>) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.slice(offset, len.unwrap_or(IdxSize::MAX)).into()
    }

    fn tail(&self, n: IdxSize) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.tail(n).into()
    }

    #[cfg(feature = "pivot")]
    #[pyo3(signature = (on, on_columns, index, values, agg, maintain_order, separator))]
    fn pivot(
        &self,
        on: PySelector,
        on_columns: PyDataFrame,
        index: PySelector,
        values: PySelector,
        agg: PyExpr,
        maintain_order: bool,
        separator: String,
    ) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.pivot(
            on.inner,
            Arc::new(on_columns.df.read().clone()),
            index.inner,
            values.inner,
            agg.inner,
            maintain_order,
            separator.into(),
        )
        .into()
    }

    #[cfg(feature = "pivot")]
    #[pyo3(signature = (on, index, value_name, variable_name))]
    fn unpivot(
        &self,
        on: Option<PySelector>,
        index: PySelector,
        value_name: Option<String>,
        variable_name: Option<String>,
    ) -> Self {
        let args = UnpivotArgsDSL {
            on: on.map(|on| on.inner),
            index: index.inner,
            value_name: value_name.map(|s| s.into()),
            variable_name: variable_name.map(|s| s.into()),
        };

        let ldf = self.ldf.read().clone();
        ldf.unpivot(args).into()
    }

    #[pyo3(signature = (name, offset=None))]
    fn with_row_index(&self, name: &str, offset: Option<IdxSize>) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.with_row_index(name, offset).into()
    }

    #[pyo3(signature = (function, predicate_pushdown, projection_pushdown, slice_pushdown, streamable, schema, validate_output))]
    fn map_batches(
        &self,
        function: Py<PyAny>,
        predicate_pushdown: bool,
        projection_pushdown: bool,
        slice_pushdown: bool,
        streamable: bool,
        schema: Option<Wrap<Schema>>,
        validate_output: bool,
    ) -> Self {
        let mut opt = OptFlags::default();
        opt.set(OptFlags::PREDICATE_PUSHDOWN, predicate_pushdown);
        opt.set(OptFlags::PROJECTION_PUSHDOWN, projection_pushdown);
        opt.set(OptFlags::SLICE_PUSHDOWN, slice_pushdown);
        opt.set(OptFlags::NEW_STREAMING, streamable);

        self.ldf
            .read()
            .clone()
            .map_python(
                function.into(),
                opt,
                schema.map(|s| Arc::new(s.0)),
                validate_output,
            )
            .into()
    }

    fn drop(&self, columns: PySelector) -> Self {
        self.ldf.read().clone().drop(columns.inner).into()
    }

    fn cast(&self, dtypes: HashMap<PyBackedStr, Wrap<DataType>>, strict: bool) -> Self {
        let mut cast_map = PlHashMap::with_capacity(dtypes.len());
        cast_map.extend(dtypes.iter().map(|(k, v)| (k.as_ref(), v.0.clone())));
        self.ldf.read().clone().cast(cast_map, strict).into()
    }

    fn cast_all(&self, dtype: PyDataTypeExpr, strict: bool) -> Self {
        self.ldf.read().clone().cast_all(dtype.inner, strict).into()
    }

    fn clone(&self) -> Self {
        self.ldf.read().clone().into()
    }

    fn collect_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let schema = py.enter_polars(|| self.ldf.write().collect_schema())?;

        let schema_dict = PyDict::new(py);
        schema.iter_fields().for_each(|fld| {
            schema_dict
                .set_item(fld.name().as_str(), &Wrap(fld.dtype().clone()))
                .unwrap()
        });
        Ok(schema_dict)
    }

    fn unnest(&self, columns: PySelector, separator: Option<&str>) -> Self {
        self.ldf
            .read()
            .clone()
            .unnest(columns.inner, separator.map(PlSmallStr::from_str))
            .into()
    }

    fn count(&self) -> Self {
        let ldf = self.ldf.read().clone();
        ldf.count().into()
    }

    #[cfg(feature = "merge_sorted")]
    fn merge_sorted(&self, other: Self, key: &str) -> PyResult<Self> {
        let out = self
            .ldf
            .read()
            .clone()
            .merge_sorted(other.ldf.into_inner(), key)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn _node_name(&self) -> &str {
        let plan = &self.ldf.read().logical_plan;
        plan.into()
    }

    fn hint_sorted(
        &self,
        columns: Vec<String>,
        descending: Vec<bool>,
        nulls_last: Vec<bool>,
    ) -> PyResult<Self> {
        if columns.len() != descending.len() && descending.len() != 1 {
            return Err(PyValueError::new_err(
                "`set_sorted` expects the same amount of `columns` as `descending` values.",
            ));
        }
        if columns.len() != nulls_last.len() && nulls_last.len() != 1 {
            return Err(PyValueError::new_err(
                "`set_sorted` expects the same amount of `columns` as `nulls_last` values.",
            ));
        }

        let mut sorted = columns
            .iter()
            .map(|c| Sorted {
                column: PlSmallStr::from_str(c.as_str()),
                descending: Some(false),
                nulls_last: Some(false),
            })
            .collect::<Vec<_>>();

        if !columns.is_empty() {
            if descending.len() != 1 {
                sorted
                    .iter_mut()
                    .zip(descending)
                    .for_each(|(s, d)| s.descending = Some(d));
            } else if descending[0] {
                sorted.iter_mut().for_each(|s| s.descending = Some(true));
            }

            if nulls_last.len() != 1 {
                sorted
                    .iter_mut()
                    .zip(nulls_last)
                    .for_each(|(s, d)| s.nulls_last = Some(d));
            } else if nulls_last[0] {
                sorted.iter_mut().for_each(|s| s.nulls_last = Some(true));
            }
        }

        let out = self
            .ldf
            .read()
            .clone()
            .hint(HintIR::Sorted(sorted.into()))
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }
}

#[pyclass(frozen)]
struct PyCollectBatches {
    inner: Arc<Mutex<CollectBatches>>,
    ldf: LazyFrame,
}

#[pymethods]
impl PyCollectBatches {
    fn start(&self) {
        self.inner.lock().start();
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRef<'_, Self>, py: Python) -> PyResult<Option<PyDataFrame>> {
        let inner = Arc::clone(&slf.inner);
        py.enter_polars(|| PolarsResult::Ok(inner.lock().next().transpose()?.map(PyDataFrame::new)))
    }

    #[allow(unused_variables)]
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let mut ldf = self.ldf.clone();
        let schema = ldf
            .collect_schema()
            .map_err(PyPolarsErr::from)?
            .to_arrow(CompatLevel::newest());

        let dtype = ArrowDataType::Struct(schema.into_iter_values().collect());

        let iter = Box::new(ArrowStreamIterator::new(self.inner.clone(), dtype.clone()));
        let field = ArrowField::new(PlSmallStr::EMPTY, dtype, false);
        let stream = export_iterator(iter, field);
        let stream_capsule_name = CString::new("arrow_array_stream").unwrap();
        PyCapsule::new(py, stream, Some(stream_capsule_name))
    }
}

pub struct ArrowStreamIterator {
    inner: Arc<Mutex<CollectBatches>>,
    dtype: ArrowDataType,
}

impl ArrowStreamIterator {
    fn new(inner: Arc<Mutex<CollectBatches>>, schema: ArrowDataType) -> Self {
        Self {
            inner,
            dtype: schema,
        }
    }
}

impl Iterator for ArrowStreamIterator {
    type Item = PolarsResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.inner.lock().next();
        match next {
            None => None,
            Some(Err(err)) => Some(Err(err)),
            Some(Ok(df)) => {
                let height = df.height();
                let arrays = df.rechunk_into_arrow(CompatLevel::newest());
                Some(Ok(Box::new(arrow::array::StructArray::new(
                    self.dtype.clone(),
                    height,
                    arrays,
                    None,
                ))))
            },
        }
    }
}
