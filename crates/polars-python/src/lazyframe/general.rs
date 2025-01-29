use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use polars::io::{HiveOptions, RowIndex};
use polars::time::*;
use polars_core::prelude::*;
#[cfg(feature = "parquet")]
use polars_parquet::arrow::write::StatisticsOptions;
use polars_plan::plans::ScanSources;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyDict, PyList};

use super::PyLazyFrame;
use crate::error::PyPolarsErr;
use crate::expr::ToExprs;
use crate::interop::arrow::to_rust::pyarrow_schema_to_rust;
use crate::lazyframe::visit::NodeTraverser;
use crate::prelude::*;
use crate::utils::EnterPolarsExt;
use crate::{PyDataFrame, PyExpr, PyLazyGroupBy};

fn pyobject_to_first_path_and_scan_sources(
    obj: PyObject,
) -> PyResult<(Option<PathBuf>, ScanSources)> {
    use crate::file::{get_python_scan_source_input, PythonScanSourceInput};
    Ok(match get_python_scan_source_input(obj, false)? {
        PythonScanSourceInput::Path(path) => {
            (Some(path.clone()), ScanSources::Paths([path].into()))
        },
        PythonScanSourceInput::File(file) => (None, ScanSources::Files([file].into())),
        PythonScanSourceInput::Buffer(buff) => (None, ScanSources::Buffers([buff].into())),
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
        row_index, ignore_errors, include_file_paths, cloud_options, credential_provider, retries, file_cache_ttl
    ))]
    fn new_from_ndjson(
        source: Option<PyObject>,
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
        cloud_options: Option<Vec<(String, String)>>,
        credential_provider: Option<PyObject>,
        retries: usize,
        file_cache_ttl: Option<u64>,
    ) -> PyResult<Self> {
        use cloud::credential_provider::PlCredentialProvider;
        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: name.into(),
            offset,
        });

        let sources = sources.0;
        let (first_path, sources) = match source {
            None => (sources.first_path().map(|p| p.to_path_buf()), sources),
            Some(source) => pyobject_to_first_path_and_scan_sources(source)?,
        };

        let mut r = LazyJsonLineReader::new_with_sources(sources);

        #[cfg(feature = "cloud")]
        if let Some(first_path) = first_path {
            let first_path_url = first_path.to_string_lossy();

            let mut cloud_options =
                parse_cloud_options(&first_path_url, cloud_options.unwrap_or_default())?;
            cloud_options = cloud_options
                .with_max_retries(retries)
                .with_credential_provider(
                    credential_provider.map(PlCredentialProvider::from_python_func_object),
                );

            if let Some(file_cache_ttl) = file_cache_ttl {
                cloud_options.file_cache_ttl = file_cache_ttl;
            }

            r = r.with_cloud_options(Some(cloud_options));
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
        cloud_options, credential_provider, retries, file_cache_ttl, include_file_paths
    )
    )]
    fn new_from_csv(
        source: Option<PyObject>,
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
        credential_provider: Option<PyObject>,
        retries: usize,
        file_cache_ttl: Option<u64>,
        include_file_paths: Option<String>,
    ) -> PyResult<Self> {
        #[cfg(feature = "cloud")]
        use cloud::credential_provider::PlCredentialProvider;

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
            None => (sources.first_path().map(|p| p.to_path_buf()), sources),
            Some(source) => pyobject_to_first_path_and_scan_sources(source)?,
        };

        let mut r = LazyCsvReader::new_with_sources(sources);

        #[cfg(feature = "cloud")]
        if let Some(first_path) = first_path {
            let first_path_url = first_path.to_string_lossy();

            let mut cloud_options =
                parse_cloud_options(&first_path_url, cloud_options.unwrap_or_default())?;
            if let Some(file_cache_ttl) = file_cache_ttl {
                cloud_options.file_cache_ttl = file_cache_ttl;
            }
            cloud_options = cloud_options
                .with_max_retries(retries)
                .with_credential_provider(
                    credential_provider.map(PlCredentialProvider::from_python_func_object),
                );
            r = r.with_cloud_options(Some(cloud_options));
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
                Python::with_gil(|py| {
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
        source, sources, n_rows, cache, parallel, rechunk, row_index, low_memory, cloud_options,
        credential_provider, use_statistics, hive_partitioning, schema, hive_schema,
        try_parse_hive_dates, retries, glob, include_file_paths, allow_missing_columns,
    ))]
    fn new_from_parquet(
        source: Option<PyObject>,
        sources: Wrap<ScanSources>,
        n_rows: Option<usize>,
        cache: bool,
        parallel: Wrap<ParallelStrategy>,
        rechunk: bool,
        row_index: Option<(String, IdxSize)>,
        low_memory: bool,
        cloud_options: Option<Vec<(String, String)>>,
        credential_provider: Option<PyObject>,
        use_statistics: bool,
        hive_partitioning: Option<bool>,
        schema: Option<Wrap<Schema>>,
        hive_schema: Option<Wrap<Schema>>,
        try_parse_hive_dates: bool,
        retries: usize,
        glob: bool,
        include_file_paths: Option<String>,
        allow_missing_columns: bool,
    ) -> PyResult<Self> {
        use cloud::credential_provider::PlCredentialProvider;

        let parallel = parallel.0;
        let hive_schema = hive_schema.map(|s| Arc::new(s.0));

        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: name.into(),
            offset,
        });

        let hive_options = HiveOptions {
            enabled: hive_partitioning,
            hive_start_idx: 0,
            schema: hive_schema,
            try_parse_dates: try_parse_hive_dates,
        };

        let mut args = ScanArgsParquet {
            n_rows,
            cache,
            parallel,
            rechunk,
            row_index,
            low_memory,
            cloud_options: None,
            use_statistics,
            schema: schema.map(|x| Arc::new(x.0)),
            hive_options,
            glob,
            include_file_paths: include_file_paths.map(|x| x.into()),
            allow_missing_columns,
        };

        let sources = sources.0;
        let (first_path, sources) = match source {
            None => (sources.first_path().map(|p| p.to_path_buf()), sources),
            Some(source) => pyobject_to_first_path_and_scan_sources(source)?,
        };

        #[cfg(feature = "cloud")]
        if let Some(first_path) = first_path {
            let first_path_url = first_path.to_string_lossy();
            let cloud_options =
                parse_cloud_options(&first_path_url, cloud_options.unwrap_or_default())?;
            args.cloud_options = Some(
                cloud_options
                    .with_max_retries(retries)
                    .with_credential_provider(
                        credential_provider.map(PlCredentialProvider::from_python_func_object),
                    ),
            );
        }

        let lf = LazyFrame::scan_parquet_sources(sources, args).map_err(PyPolarsErr::from)?;

        Ok(lf.into())
    }

    #[cfg(feature = "ipc")]
    #[staticmethod]
    #[pyo3(signature = (
        source, sources, n_rows, cache, rechunk, row_index, cloud_options,credential_provider,
        hive_partitioning, hive_schema, try_parse_hive_dates, retries, file_cache_ttl,
        include_file_paths
    ))]
    fn new_from_ipc(
        source: Option<PyObject>,
        sources: Wrap<ScanSources>,
        n_rows: Option<usize>,
        cache: bool,
        rechunk: bool,
        row_index: Option<(String, IdxSize)>,
        cloud_options: Option<Vec<(String, String)>>,
        credential_provider: Option<PyObject>,
        hive_partitioning: Option<bool>,
        hive_schema: Option<Wrap<Schema>>,
        try_parse_hive_dates: bool,
        retries: usize,
        file_cache_ttl: Option<u64>,
        include_file_paths: Option<String>,
    ) -> PyResult<Self> {
        #[cfg(feature = "cloud")]
        use cloud::credential_provider::PlCredentialProvider;
        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: name.into(),
            offset,
        });

        let hive_options = HiveOptions {
            enabled: hive_partitioning,
            hive_start_idx: 0,
            schema: hive_schema.map(|x| Arc::new(x.0)),
            try_parse_dates: try_parse_hive_dates,
        };

        let mut args = ScanArgsIpc {
            n_rows,
            cache,
            rechunk,
            row_index,
            cloud_options: None,
            hive_options,
            include_file_paths: include_file_paths.map(|x| x.into()),
        };

        let sources = sources.0;
        let (first_path, sources) = match source {
            None => (sources.first_path().map(|p| p.to_path_buf()), sources),
            Some(source) => pyobject_to_first_path_and_scan_sources(source)?,
        };

        #[cfg(feature = "cloud")]
        if let Some(first_path) = first_path {
            let first_path_url = first_path.to_string_lossy();

            let mut cloud_options =
                parse_cloud_options(&first_path_url, cloud_options.unwrap_or_default())?;
            if let Some(file_cache_ttl) = file_cache_ttl {
                cloud_options.file_cache_ttl = file_cache_ttl;
            }
            args.cloud_options = Some(
                cloud_options
                    .with_max_retries(retries)
                    .with_credential_provider(
                        credential_provider.map(PlCredentialProvider::from_python_func_object),
                    ),
            );
        }

        let lf = LazyFrame::scan_ipc_sources(sources, args).map_err(PyPolarsErr::from)?;
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
        let schema = Schema::from_iter(
            schema
                .into_iter()
                .map(|(name, dt)| Field::new((&*name).into(), dt.0)),
        );
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
        type_check: bool,
        predicate_pushdown: bool,
        projection_pushdown: bool,
        simplify_expression: bool,
        slice_pushdown: bool,
        comm_subplan_elim: bool,
        comm_subexpr_elim: bool,
        cluster_with_columns: bool,
        collapse_joins: bool,
        streaming: bool,
        _eager: bool,
        _check_order: bool,
        #[allow(unused_variables)] new_streaming: bool,
    ) -> Self {
        let ldf = self.ldf.clone();
        let mut ldf = ldf
            .with_type_coercion(type_coercion)
            .with_type_check(type_check)
            .with_predicate_pushdown(predicate_pushdown)
            .with_simplify_expr(simplify_expression)
            .with_slice_pushdown(slice_pushdown)
            .with_cluster_with_columns(cluster_with_columns)
            .with_collapse_joins(collapse_joins)
            .with_check_order(_check_order)
            ._with_eager(_eager)
            .with_projection_pushdown(projection_pushdown);

        #[cfg(feature = "streaming")]
        {
            ldf = ldf.with_streaming(streaming);
        }

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
        let ldf = self.ldf.clone();
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
        let (df, time_df) = py.enter_polars(|| self.ldf.clone().profile())?;
        Ok((df.into(), time_df.into()))
    }

    #[pyo3(signature = (lambda_post_opt=None))]
    fn collect(&self, py: Python, lambda_post_opt: Option<PyObject>) -> PyResult<PyDataFrame> {
        py.enter_polars_df(|| {
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
        })
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
                        .call1(py, (PyErr::from(err),))
                        .map_err(|err| err.restore(py))
                        .ok();
                },
            });
        });
    }

    #[cfg(all(feature = "streaming", feature = "parquet"))]
    #[pyo3(signature = (
        path, compression, compression_level, statistics, row_group_size, data_page_size,
        maintain_order, cloud_options, credential_provider, retries
    ))]
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
        cloud_options: Option<Vec<(String, String)>>,
        credential_provider: Option<PyObject>,
        retries: usize,
    ) -> PyResult<()> {
        let compression = parse_parquet_compression(compression, compression_level)?;

        let options = ParquetWriteOptions {
            compression,
            statistics: statistics.0,
            row_group_size,
            data_page_size,
            maintain_order,
        };

        let cloud_options = {
            let cloud_options =
                parse_cloud_options(path.to_str().unwrap(), cloud_options.unwrap_or_default())?;
            Some(
                cloud_options
                    .with_max_retries(retries)
                    .with_credential_provider(
                        credential_provider.map(polars::prelude::cloud::credential_provider::PlCredentialProvider::from_python_func_object),
                    ),
            )
        };

        py.enter_polars(|| self.ldf.clone().sink_parquet(&path, options, cloud_options))
    }

    #[cfg(all(feature = "streaming", feature = "ipc"))]
    #[pyo3(signature = (path, compression, maintain_order, cloud_options, credential_provider, retries))]
    fn sink_ipc(
        &self,
        py: Python,
        path: PathBuf,
        compression: Option<Wrap<IpcCompression>>,
        maintain_order: bool,
        cloud_options: Option<Vec<(String, String)>>,
        credential_provider: Option<PyObject>,
        retries: usize,
    ) -> PyResult<()> {
        let options = IpcWriterOptions {
            compression: compression.map(|c| c.0),
            maintain_order,
        };

        #[cfg(feature = "cloud")]
        let cloud_options = {
            let cloud_options =
                parse_cloud_options(path.to_str().unwrap(), cloud_options.unwrap_or_default())?;
            Some(
                cloud_options
                    .with_max_retries(retries)
                    .with_credential_provider(
                        credential_provider.map(polars::prelude::cloud::credential_provider::PlCredentialProvider::from_python_func_object),
                    ),
            )
        };

        #[cfg(not(feature = "cloud"))]
        let cloud_options = None;

        py.enter_polars(|| self.ldf.clone().sink_ipc(path, options, cloud_options))
    }

    #[cfg(all(feature = "streaming", feature = "csv"))]
    #[pyo3(signature = (
        path, include_bom, include_header, separator, line_terminator, quote_char, batch_size,
        datetime_format, date_format, time_format, float_scientific, float_precision, null_value,
        quote_style, maintain_order, cloud_options, credential_provider, retries
    ))]
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
        cloud_options: Option<Vec<(String, String)>>,
        credential_provider: Option<PyObject>,
        retries: usize,
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

        #[cfg(feature = "cloud")]
        let cloud_options = {
            let cloud_options =
                parse_cloud_options(path.to_str().unwrap(), cloud_options.unwrap_or_default())?;
            Some(
                cloud_options
                    .with_max_retries(retries)
                    .with_credential_provider(
                        credential_provider.map(polars::prelude::cloud::credential_provider::PlCredentialProvider::from_python_func_object),
                    ),
            )
        };

        #[cfg(not(feature = "cloud"))]
        let cloud_options = None;

        py.enter_polars(|| {
            let ldf = self.ldf.clone();
            ldf.sink_csv(path, options, cloud_options)
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(all(feature = "streaming", feature = "json"))]
    #[pyo3(signature = (path, maintain_order, cloud_options, credential_provider, retries))]
    fn sink_json(
        &self,
        py: Python,
        path: PathBuf,
        maintain_order: bool,
        cloud_options: Option<Vec<(String, String)>>,
        credential_provider: Option<PyObject>,
        retries: usize,
    ) -> PyResult<()> {
        let options = JsonWriterOptions { maintain_order };

        let cloud_options = {
            let cloud_options =
                parse_cloud_options(path.to_str().unwrap(), cloud_options.unwrap_or_default())?;
            Some(
                cloud_options
                    .with_max_retries(retries)
                    .with_credential_provider(
                        credential_provider.map(polars::prelude::cloud::credential_provider::PlCredentialProvider::from_python_func_object),
                    ),
            )
        };

        py.enter_polars(|| {
            let ldf = self.ldf.clone();
            ldf.sink_json(path, options, cloud_options)
        })
    }

    fn fetch(&self, py: Python, n_rows: usize) -> PyResult<PyDataFrame> {
        let ldf = self.ldf.clone();
        py.enter_polars_df(|| ldf.fetch(n_rows))
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
    ) -> PyResult<PyLazyGroupBy> {
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
                period: Duration::try_parse(period).map_err(PyPolarsErr::from)?,
                offset: Duration::try_parse(offset).map_err(PyPolarsErr::from)?,
                closed_window,
            },
        );

        Ok(PyLazyGroupBy { lgb: Some(lazy_gb) })
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
    ) -> PyResult<PyLazyGroupBy> {
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
        let contexts = contexts.into_iter().map(|ldf| ldf.ldf).collect::<Vec<_>>();
        self.ldf.clone().with_context(contexts).into()
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
                left_by: left_by.map(strings_to_pl_smallstr),
                right_by: right_by.map(strings_to_pl_smallstr),
                tolerance: tolerance.map(|t| t.0.into_static()),
                tolerance_str: tolerance_str.map(|s| s.into()),
                allow_eq,
                check_sortedness,
            }))
            .suffix(suffix)
            .finish()
            .into())
    }

    #[pyo3(signature = (other, left_on, right_on, allow_parallel, force_parallel, join_nulls, how, suffix, validate, maintain_order, coalesce=None))]
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
        maintain_order: Wrap<MaintainOrderJoin>,
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
            .suffix(suffix)
            .validate(validate.0)
            .coalesce(coalesce)
            .maintain_order(maintain_order.0)
            .finish()
            .into())
    }

    fn join_where(&self, other: Self, predicates: Vec<PyExpr>, suffix: String) -> PyResult<Self> {
        let ldf = self.ldf.clone();
        let other = other.ldf;

        let predicates = predicates.to_exprs();

        Ok(ldf
            .join_builder()
            .with(other)
            .suffix(suffix)
            .join_where(predicates)
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

    fn rename(&mut self, existing: Vec<String>, new: Vec<String>, strict: bool) -> Self {
        let ldf = self.ldf.clone();
        ldf.rename(existing, new, strict).into()
    }

    fn reverse(&self) -> Self {
        let ldf = self.ldf.clone();
        ldf.reverse().into()
    }

    #[pyo3(signature = (n, fill_value=None))]
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

    fn quantile(&self, quantile: PyExpr, interpolation: Wrap<QuantileMethod>) -> Self {
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

    #[pyo3(signature = (subset=None))]
    fn drop_nans(&self, subset: Option<Vec<PyExpr>>) -> Self {
        let ldf = self.ldf.clone();
        let subset = subset.map(|e| e.to_exprs());
        ldf.drop_nans(subset).into()
    }

    #[pyo3(signature = (subset=None))]
    fn drop_nulls(&self, subset: Option<Vec<PyExpr>>) -> Self {
        let ldf = self.ldf.clone();
        let subset = subset.map(|e| e.to_exprs());
        ldf.drop_nulls(subset).into()
    }

    #[pyo3(signature = (offset, len=None))]
    fn slice(&self, offset: i64, len: Option<IdxSize>) -> Self {
        let ldf = self.ldf.clone();
        ldf.slice(offset, len.unwrap_or(IdxSize::MAX)).into()
    }

    fn tail(&self, n: IdxSize) -> Self {
        let ldf = self.ldf.clone();
        ldf.tail(n).into()
    }

    #[cfg(feature = "pivot")]
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

    #[pyo3(signature = (name, offset=None))]
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

    fn collect_schema<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let schema = py.enter_polars(|| self.ldf.collect_schema())?;

        let schema_dict = PyDict::new(py);
        schema.iter_fields().for_each(|fld| {
            schema_dict
                .set_item(fld.name().as_str(), &Wrap(fld.dtype().clone()))
                .unwrap()
        });
        Ok(schema_dict)
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
