use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use polars::io::{HiveOptions, RowIndex};
use polars::time::*;
use polars_core::prelude::*;
#[cfg(feature = "parquet")]
use polars_parquet::arrow::write::StatisticsOptions;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyDict, PyList};

use super::PyLazyFrame;
use crate::error::PyPolarsErr;
use crate::expr::ToExprs;
use crate::interop::arrow::to_rust::pyarrow_schema_to_rust;
use crate::lazyframe::visit::NodeTraverser;
use crate::prelude::*;
use crate::{PyDataFrame, PyExpr, PyLazyGroupBy};

#[pymethods]
#[allow(clippy::should_implement_trait)]
impl PyLazyFrame {
    #[staticmethod]
    #[cfg(feature = "json")]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        path, paths, infer_schema_length, schema, schema_overrides, batch_size, n_rows, low_memory, rechunk,
        row_index, ignore_errors, include_file_paths, cloud_options, retries, file_cache_ttl
    ))]
    fn new_from_ndjson(
        path: Option<PathBuf>,
        paths: Vec<PathBuf>,
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
        cloud_options: Option<Vec<(String, String)>>,
        retries: usize,
        file_cache_ttl: Option<u64>,
    ) -> PyResult<Self> {
        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: Arc::from(name.as_str()),
            offset,
        });

        #[cfg(feature = "cloud")]
        let cloud_options = {
            let first_path = if let Some(path) = &path {
                path
            } else {
                paths
                    .first()
                    .ok_or_else(|| PyValueError::new_err("expected a path argument"))?
            };

            let first_path_url = first_path.to_string_lossy();

            let mut cloud_options = if let Some(opts) = cloud_options {
                parse_cloud_options(&first_path_url, opts)?
            } else {
                parse_cloud_options(&first_path_url, vec![])?
            };

            cloud_options = cloud_options.with_max_retries(retries);

            if let Some(file_cache_ttl) = file_cache_ttl {
                cloud_options.file_cache_ttl = file_cache_ttl;
            }

            Some(cloud_options)
        };

        let r = if let Some(path) = &path {
            LazyJsonLineReader::new(path)
        } else {
            LazyJsonLineReader::new_paths(paths.into())
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
            .with_include_file_paths(include_file_paths.map(Arc::from))
            .with_cloud_options(cloud_options)
            .finish()
            .map_err(PyPolarsErr::from)?;

        Ok(lf.into())
    }

    #[staticmethod]
    #[cfg(feature = "csv")]
    #[pyo3(signature = (path, paths, separator, has_header, ignore_errors, skip_rows, n_rows, cache, overwrite_dtype,
        low_memory, comment_prefix, quote_char, null_values, missing_utf8_is_empty_string,
        infer_schema_length, with_schema_modify, rechunk, skip_rows_after_header,
        encoding, row_index, try_parse_dates, eol_char, raise_if_empty, truncate_ragged_lines, decimal_comma, glob, schema,
        cloud_options, retries, file_cache_ttl, include_file_paths
    )
    )]
    fn new_from_csv(
        path: Option<PathBuf>,
        paths: Vec<PathBuf>,
        separator: &str,
        has_header: bool,
        ignore_errors: bool,
        skip_rows: usize,
        n_rows: Option<usize>,
        cache: bool,
        overwrite_dtype: Option<Vec<(PyBackedStr, Wrap<DataType>)>>,
        low_memory: bool,
        comment_prefix: Option<&str>,
        quote_char: Option<&str>,
        null_values: Option<Wrap<NullValues>>,
        missing_utf8_is_empty_string: bool,
        infer_schema_length: Option<usize>,
        with_schema_modify: Option<PyObject>,
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
        cloud_options: Option<Vec<(String, String)>>,
        retries: usize,
        file_cache_ttl: Option<u64>,
        include_file_paths: Option<String>,
    ) -> PyResult<Self> {
        let null_values = null_values.map(|w| w.0);
        let quote_char = quote_char.map(|s| s.as_bytes()[0]);
        let separator = separator.as_bytes()[0];
        let eol_char = eol_char.as_bytes()[0];
        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: Arc::from(name.as_str()),
            offset,
        });

        let overwrite_dtype = overwrite_dtype.map(|overwrite_dtype| {
            overwrite_dtype
                .into_iter()
                .map(|(name, dtype)| Field::new(&name, dtype.0))
                .collect::<Schema>()
        });

        #[cfg(feature = "cloud")]
        let cloud_options = {
            let first_path = if let Some(path) = &path {
                path
            } else {
                paths
                    .first()
                    .ok_or_else(|| PyValueError::new_err("expected a path argument"))?
            };

            let first_path_url = first_path.to_string_lossy();

            let mut cloud_options = if let Some(opts) = cloud_options {
                parse_cloud_options(&first_path_url, opts)?
            } else {
                parse_cloud_options(&first_path_url, vec![])?
            };

            cloud_options = cloud_options.with_max_retries(retries);

            if let Some(file_cache_ttl) = file_cache_ttl {
                cloud_options.file_cache_ttl = file_cache_ttl;
            }

            Some(cloud_options)
        };

        let r = if let Some(path) = path.as_ref() {
            LazyCsvReader::new(path)
        } else {
            LazyCsvReader::new_paths(paths.into())
        };

        let mut r = r
            .with_infer_schema_length(infer_schema_length)
            .with_separator(separator)
            .with_has_header(has_header)
            .with_ignore_errors(ignore_errors)
            .with_skip_rows(skip_rows)
            .with_n_rows(n_rows)
            .with_cache(cache)
            .with_dtype_overwrite(overwrite_dtype.map(Arc::new))
            .with_schema(schema.map(|schema| Arc::new(schema.0)))
            .with_low_memory(low_memory)
            .with_comment_prefix(comment_prefix)
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
            .with_cloud_options(cloud_options)
            .with_include_file_paths(include_file_paths.map(Arc::from));

        if let Some(lambda) = with_schema_modify {
            let f = |schema: Schema| {
                let iter = schema.iter_names().map(|s| s.as_str());
                Python::with_gil(|py| {
                    let names = PyList::new_bound(py, iter);

                    let out = lambda.call1(py, (names,)).expect("python function failed");
                    let new_names = out
                        .extract::<Vec<String>>(py)
                        .expect("python function should return List[str]");
                    polars_ensure!(new_names.len() == schema.len(),
                        ShapeMismatch: "The length of the new names list should be equal to or less than the original column length",
                    );
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
    #[pyo3(signature = (path, paths, n_rows, cache, parallel, rechunk, row_index,
        low_memory, cloud_options, use_statistics, hive_partitioning, hive_schema, try_parse_hive_dates, retries, glob, include_file_paths)
    )]
    fn new_from_parquet(
        path: Option<PathBuf>,
        paths: Vec<PathBuf>,
        n_rows: Option<usize>,
        cache: bool,
        parallel: Wrap<ParallelStrategy>,
        rechunk: bool,
        row_index: Option<(String, IdxSize)>,
        low_memory: bool,
        cloud_options: Option<Vec<(String, String)>>,
        use_statistics: bool,
        hive_partitioning: Option<bool>,
        hive_schema: Option<Wrap<Schema>>,
        try_parse_hive_dates: bool,
        retries: usize,
        glob: bool,
        include_file_paths: Option<String>,
    ) -> PyResult<Self> {
        let parallel = parallel.0;
        let hive_schema = hive_schema.map(|s| Arc::new(s.0));

        let first_path = if let Some(path) = &path {
            path
        } else {
            paths
                .first()
                .ok_or_else(|| PyValueError::new_err("expected a path argument"))?
        };

        #[cfg(feature = "cloud")]
        let cloud_options = {
            let first_path_url = first_path.to_string_lossy();

            let mut cloud_options = if let Some(opts) = cloud_options {
                parse_cloud_options(&first_path_url, opts)?
            } else {
                parse_cloud_options(&first_path_url, vec![])?
            };

            cloud_options = cloud_options.with_max_retries(retries);

            Some(cloud_options)
        };

        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: Arc::from(name.as_str()),
            offset,
        });
        let hive_options = HiveOptions {
            enabled: hive_partitioning,
            hive_start_idx: 0,
            schema: hive_schema,
            try_parse_dates: try_parse_hive_dates,
        };

        let args = ScanArgsParquet {
            n_rows,
            cache,
            parallel,
            rechunk,
            row_index,
            low_memory,
            cloud_options,
            use_statistics,
            hive_options,
            glob,
            include_file_paths: include_file_paths.map(Arc::from),
        };

        let lf = if path.is_some() {
            LazyFrame::scan_parquet(first_path, args)
        } else {
            LazyFrame::scan_parquet_files(Arc::from(paths), args)
        }
        .map_err(PyPolarsErr::from)?;
        Ok(lf.into())
    }

    #[cfg(feature = "ipc")]
    #[staticmethod]
    #[pyo3(signature = (path, paths, n_rows, cache, rechunk, row_index, memory_map, cloud_options, hive_partitioning, hive_schema, try_parse_hive_dates, retries, file_cache_ttl, include_file_paths))]
    fn new_from_ipc(
        path: Option<PathBuf>,
        paths: Vec<PathBuf>,
        n_rows: Option<usize>,
        cache: bool,
        rechunk: bool,
        row_index: Option<(String, IdxSize)>,
        memory_map: bool,
        cloud_options: Option<Vec<(String, String)>>,
        hive_partitioning: Option<bool>,
        hive_schema: Option<Wrap<Schema>>,
        try_parse_hive_dates: bool,
        retries: usize,
        file_cache_ttl: Option<u64>,
        include_file_paths: Option<String>,
    ) -> PyResult<Self> {
        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: Arc::from(name.as_str()),
            offset,
        });

        #[cfg(feature = "cloud")]
        let cloud_options = {
            let first_path = if let Some(path) = &path {
                path
            } else {
                paths
                    .first()
                    .ok_or_else(|| PyValueError::new_err("expected a path argument"))?
            };

            let first_path_url = first_path.to_string_lossy();

            let mut cloud_options = if let Some(opts) = cloud_options {
                parse_cloud_options(&first_path_url, opts)?
            } else {
                parse_cloud_options(&first_path_url, vec![])?
            };

            cloud_options = cloud_options.with_max_retries(retries);

            if let Some(file_cache_ttl) = file_cache_ttl {
                cloud_options.file_cache_ttl = file_cache_ttl;
            }

            Some(cloud_options)
        };

        let hive_options = HiveOptions {
            enabled: hive_partitioning,
            hive_start_idx: 0,
            schema: hive_schema.map(|x| Arc::new(x.0)),
            try_parse_dates: try_parse_hive_dates,
        };

        let args = ScanArgsIpc {
            n_rows,
            cache,
            rechunk,
            row_index,
            memory_map,
            #[cfg(feature = "cloud")]
            cloud_options,
            hive_options,
            include_file_paths: include_file_paths.map(Arc::from),
        };

        let lf = if let Some(path) = &path {
            LazyFrame::scan_ipc(path, args)
        } else {
            LazyFrame::scan_ipc_files(paths.into(), args)
        }
        .map_err(PyPolarsErr::from)?;
        Ok(lf.into())
    }

    #[staticmethod]
    fn scan_from_python_function_arrow_schema(
        schema: &Bound<'_, PyList>,
        scan_fn: PyObject,
        pyarrow: bool,
    ) -> PyResult<Self> {
        let schema = pyarrow_schema_to_rust(schema)?;
        Ok(LazyFrame::scan_from_python_function(schema, scan_fn, pyarrow).into())
    }

    #[staticmethod]
    fn scan_from_python_function_pl_schema(
        schema: Vec<(PyBackedStr, Wrap<DataType>)>,
        scan_fn: PyObject,
        pyarrow: bool,
    ) -> PyResult<Self> {
        let schema =
            Schema::from_iter(schema.into_iter().map(|(name, dt)| Field::new(&name, dt.0)));
        Ok(LazyFrame::scan_from_python_function(schema, scan_fn, pyarrow).into())
    }

    fn describe_plan(&self) -> PyResult<String> {
        self.ldf
            .describe_plan()
            .map_err(PyPolarsErr::from)
            .map_err(Into::into)
    }

    fn describe_optimized_plan(&self) -> PyResult<String> {
        self.ldf
            .describe_optimized_plan()
            .map_err(PyPolarsErr::from)
            .map_err(Into::into)
    }

    fn describe_plan_tree(&self) -> PyResult<String> {
        self.ldf
            .describe_plan_tree()
            .map_err(PyPolarsErr::from)
            .map_err(Into::into)
    }

    fn describe_optimized_plan_tree(&self) -> PyResult<String> {
        self.ldf
            .describe_optimized_plan_tree()
            .map_err(PyPolarsErr::from)
            .map_err(Into::into)
    }

    fn to_dot(&self, optimized: bool) -> PyResult<String> {
        let result = self.ldf.to_dot(optimized).map_err(PyPolarsErr::from)?;
        Ok(result)
    }

    fn optimization_toggle(
        &self,
        type_coercion: bool,
        predicate_pushdown: bool,
        projection_pushdown: bool,
        simplify_expression: bool,
        slice_pushdown: bool,
        comm_subplan_elim: bool,
        comm_subexpr_elim: bool,
        cluster_with_columns: bool,
        streaming: bool,
        _eager: bool,
        #[allow(unused_variables)] new_streaming: bool,
    ) -> Self {
        let ldf = self.ldf.clone();
        let mut ldf = ldf
            .with_type_coercion(type_coercion)
            .with_predicate_pushdown(predicate_pushdown)
            .with_simplify_expr(simplify_expression)
            .with_slice_pushdown(slice_pushdown)
            .with_cluster_with_columns(cluster_with_columns)
            .with_streaming(streaming)
            ._with_eager(_eager)
            .with_projection_pushdown(projection_pushdown);

        #[cfg(feature = "new_streaming")]
        {
            ldf = ldf.with_new_streaming(new_streaming);
        }

        #[cfg(feature = "cse")]
        {
            ldf = ldf.with_comm_subplan_elim(comm_subplan_elim);
            ldf = ldf.with_comm_subexpr_elim(comm_subexpr_elim);
        }

        ldf.into()
    }

    fn sort(
        &self,
        by_column: &str,
        descending: bool,
        nulls_last: bool,
        maintain_order: bool,
        multithreaded: bool,
    ) -> Self {
        let ldf = self.ldf.clone();
        ldf.sort(
            [by_column],
            SortMultipleOptions {
                descending: vec![descending],
                nulls_last: vec![nulls_last],
                multithreaded,
                maintain_order,
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
        let ldf = self.ldf.clone();
        let exprs = by.to_exprs();
        ldf.sort_by_exprs(
            exprs,
            SortMultipleOptions {
                descending,
                nulls_last,
                maintain_order,
                multithreaded,
            },
        )
        .into()
    }

    fn top_k(&self, k: IdxSize, by: Vec<PyExpr>, reverse: Vec<bool>) -> Self {
        let ldf = self.ldf.clone();
        let exprs = by.to_exprs();
        ldf.top_k(
            k,
            exprs,
            SortMultipleOptions::new().with_order_descending_multi(reverse),
        )
        .into()
    }

    fn bottom_k(&self, k: IdxSize, by: Vec<PyExpr>, reverse: Vec<bool>) -> Self {
        let ldf = self.ldf.clone();
        let exprs = by.to_exprs();
        ldf.bottom_k(
            k,
            exprs,
            SortMultipleOptions::new().with_order_descending_multi(reverse),
        )
        .into()
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

    fn collect(&self, py: Python, lambda_post_opt: Option<PyObject>) -> PyResult<PyDataFrame> {
        // if we don't allow threads and we have udfs trying to acquire the gil from different
        // threads we deadlock.
        let df = py.allow_threads(|| {
            let ldf = self.ldf.clone();
            if let Some(lambda) = lambda_post_opt {
                ldf._collect_post_opt(|root, lp_arena, expr_arena| {
                    Python::with_gil(|py| {
                        let nt = NodeTraverser::new(
                            root,
                            std::mem::take(lp_arena),
                            std::mem::take(expr_arena),
                        );

                        // Get a copy of the arena's.
                        let arenas = nt.get_arenas();

                        // Pass the node visitor which allows the python callback to replace parts of the query plan.
                        // Remove "cuda" or specify better once we have multiple post-opt callbacks.
                        lambda.call1(py, (nt,)).map_err(
                            |e| polars_err!(ComputeError: "'cuda' conversion failed: {}", e),
                        )?;

                        // Unpack the arena's.
                        // At this point the `nt` is useless.

                        std::mem::swap(lp_arena, &mut *arenas.0.lock().unwrap());
                        std::mem::swap(expr_arena, &mut *arenas.1.lock().unwrap());

                        Ok(())
                    })
                })
            } else {
                ldf.collect()
            }
            .map_err(PyPolarsErr::from)
        })?;
        Ok(df.into())
    }

    #[pyo3(signature = (lambda,))]
    fn collect_with_callback(&self, lambda: PyObject) {
        let ldf = self.ldf.clone();

        polars_core::POOL.spawn(move || {
            let result = ldf
                .collect()
                .map(PyDataFrame::new)
                .map_err(PyPolarsErr::from);

            Python::with_gil(|py| match result {
                Ok(df) => {
                    lambda.call1(py, (df,)).map_err(|err| err.restore(py)).ok();
                },
                Err(err) => {
                    lambda
                        .call1(py, (PyErr::from(err).to_object(py),))
                        .map_err(|err| err.restore(py))
                        .ok();
                },
            });
        });
    }

    #[cfg(all(feature = "streaming", feature = "parquet"))]
    #[pyo3(signature = (path, compression, compression_level, statistics, row_group_size, data_page_size, maintain_order))]
    fn sink_parquet(
        &self,
        py: Python,
        path: PathBuf,
        compression: &str,
        compression_level: Option<i32>,
        statistics: Wrap<StatisticsOptions>,
        row_group_size: Option<usize>,
        data_page_size: Option<usize>,
        maintain_order: bool,
    ) -> PyResult<()> {
        let compression = parse_parquet_compression(compression, compression_level)?;

        let options = ParquetWriteOptions {
            compression,
            statistics: statistics.0,
            row_group_size,
            data_page_size,
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

    #[cfg(all(feature = "streaming", feature = "csv"))]
    #[pyo3(signature = (path, include_bom, include_header, separator, line_terminator, quote_char, batch_size, datetime_format, date_format, time_format, float_scientific, float_precision, null_value, quote_style, maintain_order))]
    fn sink_csv(
        &self,
        py: Python,
        path: PathBuf,
        include_bom: bool,
        include_header: bool,
        separator: u8,
        line_terminator: String,
        quote_char: u8,
        batch_size: NonZeroUsize,
        datetime_format: Option<String>,
        date_format: Option<String>,
        time_format: Option<String>,
        float_scientific: Option<bool>,
        float_precision: Option<usize>,
        null_value: Option<String>,
        quote_style: Option<Wrap<QuoteStyle>>,
        maintain_order: bool,
    ) -> PyResult<()> {
        let quote_style = quote_style.map_or(QuoteStyle::default(), |wrap| wrap.0);
        let null_value = null_value.unwrap_or(SerializeOptions::default().null);

        let serialize_options = SerializeOptions {
            date_format,
            time_format,
            datetime_format,
            float_scientific,
            float_precision,
            separator,
            quote_char,
            null: null_value,
            line_terminator,
            quote_style,
        };

        let options = CsvWriterOptions {
            include_bom,
            include_header,
            maintain_order,
            batch_size,
            serialize_options,
        };

        // if we don't allow threads and we have udfs trying to acquire the gil from different
        // threads we deadlock.
        py.allow_threads(|| {
            let ldf = self.ldf.clone();
            ldf.sink_csv(path, options).map_err(PyPolarsErr::from)
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(all(feature = "streaming", feature = "json"))]
    #[pyo3(signature = (path, maintain_order))]
    fn sink_json(&self, py: Python, path: PathBuf, maintain_order: bool) -> PyResult<()> {
        let options = JsonWriterOptions { maintain_order };

        // if we don't allow threads and we have udfs trying to acquire the gil from different
        // threads we deadlock.
        py.allow_threads(|| {
            let ldf = self.ldf.clone();
            ldf.sink_json(path, options).map_err(PyPolarsErr::from)
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

    fn select_seq(&mut self, exprs: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.clone();
        let exprs = exprs.to_exprs();
        ldf.select_seq(exprs).into()
    }

    fn group_by(&mut self, by: Vec<PyExpr>, maintain_order: bool) -> PyLazyGroupBy {
        let ldf = self.ldf.clone();
        let by = by.to_exprs();
        let lazy_gb = if maintain_order {
            ldf.group_by_stable(by)
        } else {
            ldf.group_by(by)
        };

        PyLazyGroupBy { lgb: Some(lazy_gb) }
    }

    fn rolling(
        &mut self,
        index_column: PyExpr,
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
        let lazy_gb = ldf.rolling(
            index_column.inner,
            by,
            RollingGroupOptions {
                index_column: "".into(),
                period: Duration::parse(period),
                offset: Duration::parse(offset),
                closed_window,
            },
        );

        PyLazyGroupBy { lgb: Some(lazy_gb) }
    }

    fn group_by_dynamic(
        &mut self,
        index_column: PyExpr,
        every: &str,
        period: &str,
        offset: &str,
        label: Wrap<Label>,
        include_boundaries: bool,
        closed: Wrap<ClosedWindow>,
        group_by: Vec<PyExpr>,
        start_by: Wrap<StartBy>,
    ) -> PyLazyGroupBy {
        let closed_window = closed.0;
        let group_by = group_by
            .into_iter()
            .map(|pyexpr| pyexpr.inner)
            .collect::<Vec<_>>();
        let ldf = self.ldf.clone();
        let lazy_gb = ldf.group_by_dynamic(
            index_column.inner,
            group_by,
            DynamicGroupOptions {
                every: Duration::parse(every),
                period: Duration::parse(period),
                offset: Duration::parse(offset),
                label: label.0,
                include_boundaries,
                closed_window,
                start_by: start_by.0,
                ..Default::default()
            },
        );

        PyLazyGroupBy { lgb: Some(lazy_gb) }
    }

    fn with_context(&self, contexts: Vec<Self>) -> Self {
        let contexts = contexts.into_iter().map(|ldf| ldf.ldf).collect::<Vec<_>>();
        self.ldf.clone().with_context(contexts).into()
    }

    #[cfg(feature = "asof_join")]
    #[pyo3(signature = (other, left_on, right_on, left_by, right_by, allow_parallel, force_parallel, suffix, strategy, tolerance, tolerance_str, coalesce))]
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
    ) -> PyResult<Self> {
        let coalesce = if coalesce {
            JoinCoalesce::CoalesceColumns
        } else {
            JoinCoalesce::KeepColumns
        };
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
            .coalesce(coalesce)
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

    fn join(
        &self,
        other: Self,
        left_on: Vec<PyExpr>,
        right_on: Vec<PyExpr>,
        allow_parallel: bool,
        force_parallel: bool,
        join_nulls: bool,
        how: Wrap<JoinType>,
        suffix: String,
        validate: Wrap<JoinValidation>,
        coalesce: Option<bool>,
    ) -> PyResult<Self> {
        let coalesce = match coalesce {
            None => JoinCoalesce::JoinSpecific,
            Some(true) => JoinCoalesce::CoalesceColumns,
            Some(false) => JoinCoalesce::KeepColumns,
        };
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
            .join_nulls(join_nulls)
            .how(how.0)
            .coalesce(coalesce)
            .validate(validate.0)
            .suffix(suffix)
            .finish()
            .into())
    }

    fn with_columns(&mut self, exprs: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.clone();
        ldf.with_columns(exprs.to_exprs()).into()
    }

    fn with_columns_seq(&mut self, exprs: Vec<PyExpr>) -> Self {
        let ldf = self.ldf.clone();
        ldf.with_columns_seq(exprs.to_exprs()).into()
    }

    fn rename(&mut self, existing: Vec<String>, new: Vec<String>) -> Self {
        let ldf = self.ldf.clone();
        ldf.rename(existing, new).into()
    }

    fn reverse(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.reverse().into()
    }

    fn shift(&self, n: PyExpr, fill_value: Option<PyExpr>) -> Self {
        let lf = self.ldf.clone();
        let out = match fill_value {
            Some(v) => lf.shift_and_fill(n.inner, v.inner),
            None => lf.shift(n.inner),
        };
        out.into()
    }

    fn fill_nan(&self, fill_value: PyExpr) -> Self {
        let ldf = self.ldf.clone();
        ldf.fill_nan(fill_value.inner).into()
    }

    fn min(&self) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.min();
        out.into()
    }

    fn max(&self) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.max();
        out.into()
    }

    fn sum(&self) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.sum();
        out.into()
    }

    fn mean(&self) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.mean();
        out.into()
    }

    fn std(&self, ddof: u8) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.std(ddof);
        out.into()
    }

    fn var(&self, ddof: u8) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.var(ddof);
        out.into()
    }

    fn median(&self) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.median();
        out.into()
    }

    fn quantile(&self, quantile: PyExpr, interpolation: Wrap<QuantileInterpolOptions>) -> Self {
        let ldf = self.ldf.clone();
        let out = ldf.quantile(quantile.inner, interpolation.0);
        out.into()
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
        subset: Option<Vec<PyExpr>>,
        keep: Wrap<UniqueKeepStrategy>,
    ) -> Self {
        let ldf = self.ldf.clone();
        let subset = subset.map(|e| e.to_exprs());
        match maintain_order {
            true => ldf.unique_stable_generic(subset, keep.0),
            false => ldf.unique_generic(subset, keep.0),
        }
        .into()
    }

    fn drop_nulls(&self, subset: Option<Vec<PyExpr>>) -> Self {
        let ldf = self.ldf.clone();
        let subset = subset.map(|e| e.to_exprs());
        ldf.drop_nulls(subset).into()
    }

    fn slice(&self, offset: i64, len: Option<IdxSize>) -> Self {
        let ldf = self.ldf.clone();
        ldf.slice(offset, len.unwrap_or(IdxSize::MAX)).into()
    }

    fn tail(&self, n: IdxSize) -> Self {
        let ldf = self.ldf.clone();
        ldf.tail(n).into()
    }

    #[pyo3(signature = (on, index, value_name, variable_name))]
    fn unpivot(
        &self,
        on: Vec<PyExpr>,
        index: Vec<PyExpr>,
        value_name: Option<String>,
        variable_name: Option<String>,
    ) -> Self {
        let args = UnpivotArgsDSL {
            on: on.into_iter().map(|e| e.inner.into()).collect(),
            index: index.into_iter().map(|e| e.inner.into()).collect(),
            value_name: value_name.map(|s| s.into()),
            variable_name: variable_name.map(|s| s.into()),
        };

        let ldf = self.ldf.clone();
        ldf.unpivot(args).into()
    }

    fn with_row_index(&self, name: &str, offset: Option<IdxSize>) -> Self {
        let ldf = self.ldf.clone();
        ldf.with_row_index(name, offset).into()
    }

    #[pyo3(signature = (lambda, predicate_pushdown, projection_pushdown, slice_pushdown, streamable, schema, validate_output))]
    fn map_batches(
        &self,
        lambda: PyObject,
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
        opt.set(OptFlags::STREAMING, streamable);

        self.ldf
            .clone()
            .map_python(
                lambda.into(),
                opt,
                schema.map(|s| Arc::new(s.0)),
                validate_output,
            )
            .into()
    }

    fn drop(&self, columns: Vec<PyExpr>, strict: bool) -> Self {
        let ldf = self.ldf.clone();
        let columns = columns.to_exprs();
        if strict {
            ldf.drop(columns)
        } else {
            ldf.drop_no_validate(columns)
        }
        .into()
    }

    fn cast(&self, dtypes: HashMap<PyBackedStr, Wrap<DataType>>, strict: bool) -> Self {
        let mut cast_map = PlHashMap::with_capacity(dtypes.len());
        cast_map.extend(dtypes.iter().map(|(k, v)| (k.as_ref(), v.0.clone())));
        self.ldf.clone().cast(cast_map, strict).into()
    }

    fn cast_all(&self, dtype: Wrap<DataType>, strict: bool) -> Self {
        self.ldf.clone().cast_all(dtype.0, strict).into()
    }

    fn clone(&self) -> Self {
        self.ldf.clone().into()
    }

    fn collect_schema(&mut self, py: Python) -> PyResult<PyObject> {
        let schema = py
            .allow_threads(|| self.ldf.collect_schema())
            .map_err(PyPolarsErr::from)?;

        let schema_dict = PyDict::new_bound(py);
        schema.iter_fields().for_each(|fld| {
            schema_dict
                .set_item(fld.name().as_str(), Wrap(fld.data_type().clone()))
                .unwrap()
        });
        Ok(schema_dict.to_object(py))
    }

    fn unnest(&self, columns: Vec<PyExpr>) -> Self {
        let columns = columns.to_exprs();
        self.ldf.clone().unnest(columns).into()
    }

    fn count(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.count().into()
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
