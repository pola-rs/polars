#[cfg(feature = "csv")]
use std::io::{Read, Seek};

use polars_core::prelude::*;
#[cfg(feature = "parquet")]
use polars_io::cloud::CloudOptions;
#[cfg(all(feature = "parquet", feature = "async"))]
use polars_io::parquet::ParquetAsyncReader;
#[cfg(feature = "parquet")]
use polars_io::parquet::ParquetReader;
#[cfg(all(feature = "cloud", feature = "parquet"))]
use polars_io::pl_async::get_runtime;
use polars_io::HiveOptions;
#[cfg(any(
    feature = "parquet",
    feature = "parquet_async",
    feature = "csv",
    feature = "ipc"
))]
use polars_io::RowIndex;
#[cfg(feature = "csv")]
use polars_io::{
    csv::utils::{infer_file_schema, is_compressed},
    csv::CommentPrefix,
    csv::CsvEncoding,
    csv::NullValues,
    utils::get_reader_bytes,
};

use super::builder_functions::*;
use crate::constants::UNLIMITED_CACHE;
use crate::dsl::functions::horizontal::all_horizontal;
use crate::logical_plan::expr_expansion::{rewrite_projections};
#[cfg(feature = "python")]
use crate::prelude::python_udf::PythonFunction;
use crate::prelude::*;

pub(crate) fn prepare_projection(
    exprs: Vec<Expr>,
    schema: &Schema,
) -> PolarsResult<(Vec<Expr>, Schema)> {
    let exprs = rewrite_projections(exprs, schema, &[])?;
    let schema = expressions_to_schema(&exprs, schema, Context::Default)?;
    Ok((exprs, schema))
}

pub struct DslBuilder(pub DslPlan);

impl From<DslPlan> for DslBuilder {
    fn from(lp: DslPlan) -> Self {
        DslBuilder(lp)
    }
}

fn format_err(msg: &str, input: &DslPlan) -> String {
    format!("{msg}\n\nError originated just after this operation:\n{input:?}")
}

/// Returns every error or msg: &str as `ComputeError`. It also shows the logical plan node where the error originated.
/// If `input` is already a `DslPlan::Error`, then return it as is; errors already keep track of their previous
/// inputs, so we don't have to do it again here.
macro_rules! raise_err {
    ($err:expr, $input:expr, $convert:ident) => {{
        let input: DslPlan = $input.clone();
        match &input {
            DslPlan::Error { .. } => input,
            _ => {
                let format_err_outer = |msg: &str| format_err(msg, &input);
                let err = $err.wrap_msg(&format_err_outer);

                DslPlan::Error {
                    input: Arc::new(input),
                    err: err.into(), // PolarsError -> ErrorState
                }
            },
        }
        .$convert()
    }};
}

macro_rules! try_delayed {
    ($fallible:expr, $input:expr, $convert:ident) => {
        match $fallible {
            Ok(success) => success,
            Err(err) => return raise_err!(err, $input, $convert),
        }
    };
}

#[cfg(any(feature = "parquet", feature = "parquet_async",))]
fn prepare_schema(mut schema: Schema, row_index: Option<&RowIndex>) -> SchemaRef {
    if let Some(rc) = row_index {
        let _ = schema.insert_at_index(0, rc.name.as_str().into(), IDX_DTYPE);
    }
    Arc::new(schema)
}

impl DslBuilder {
    pub fn anonymous_scan(
        function: Arc<dyn AnonymousScan>,
        schema: Option<SchemaRef>,
        infer_schema_length: Option<usize>,
        skip_rows: Option<usize>,
        n_rows: Option<usize>,
        name: &'static str,
    ) -> PolarsResult<Self> {
        let schema = match schema {
            Some(s) => s,
            None => function.schema(infer_schema_length)?,
        };

        let file_info = FileInfo::new(schema.clone(), None, (n_rows, n_rows.unwrap_or(usize::MAX)));
        let file_options = FileScanOptions {
            n_rows,
            with_columns: None,
            cache: false,
            row_index: None,
            rechunk: false,
            file_counter: Default::default(),
            // TODO: Support Hive partitioning.
            hive_options: HiveOptions {
                enabled: false,
                ..Default::default()
            },
        };

        Ok(DslPlan::Scan {
            paths: Arc::new([]),
            file_info,
            predicate: None,
            file_options,
            scan_type: FileScan::Anonymous {
                function,
                options: Arc::new(AnonymousScanOptions {
                    fmt_str: name,
                    skip_rows,
                }),
            },
        }
        .into())
    }

    #[cfg(any(feature = "parquet", feature = "parquet_async"))]
    #[allow(clippy::too_many_arguments)]
    pub fn scan_parquet<P: Into<Arc<[std::path::PathBuf]>>>(
        paths: P,
        n_rows: Option<usize>,
        cache: bool,
        parallel: polars_io::parquet::ParallelStrategy,
        row_index: Option<RowIndex>,
        rechunk: bool,
        low_memory: bool,
        cloud_options: Option<CloudOptions>,
        use_statistics: bool,
        hive_options: HiveOptions,
    ) -> PolarsResult<Self> {
        use polars_io::{is_cloud_url, SerReader as _};

        let paths = paths.into();
        polars_ensure!(paths.len() >= 1, ComputeError: "expected at least 1 path");

        // Use first path to get schema.
        let path = &paths[0];

        let (schema, reader_schema, num_rows, metadata) = if is_cloud_url(path) {
            #[cfg(not(feature = "cloud"))]
            panic!(
                "One or more of the cloud storage features ('aws', 'gcp', ...) must be enabled."
            );

            #[cfg(feature = "cloud")]
            {
                let uri = path.to_string_lossy();
                get_runtime().block_on(async {
                    let mut reader =
                        ParquetAsyncReader::from_uri(&uri, cloud_options.as_ref(), None, None)
                            .await?;
                    let reader_schema = reader.schema().await?;
                    let num_rows = reader.num_rows().await?;
                    let metadata = reader.get_metadata().await?.clone();

                    let schema = prepare_schema((&reader_schema).into(), row_index.as_ref());
                    PolarsResult::Ok((schema, reader_schema, Some(num_rows), Some(metadata)))
                })?
            }
        } else {
            let file = polars_utils::open_file(path)?;
            let mut reader = ParquetReader::new(file);
            let reader_schema = reader.schema()?;
            let schema = prepare_schema((&reader_schema).into(), row_index.as_ref());
            (
                schema,
                reader_schema,
                Some(reader.num_rows()?),
                Some(reader.get_metadata()?.clone()),
            )
        };

        let mut file_info = FileInfo::new(
            schema,
            Some(reader_schema),
            (num_rows, num_rows.unwrap_or(0)),
        );

        if hive_options.enabled {
            file_info.init_hive_partitions(path.as_path(), hive_options.schema.clone())?
        }

        let options = FileScanOptions {
            with_columns: None,
            cache,
            n_rows,
            rechunk,
            row_index,
            file_counter: Default::default(),
            hive_options,
        };
        Ok(DslPlan::Scan {
            paths,
            file_info,
            file_options: options,
            predicate: None,
            scan_type: FileScan::Parquet {
                options: ParquetOptions {
                    parallel,
                    low_memory,
                    use_statistics,
                },
                cloud_options,
                metadata,
            },
        }
        .into())
    }

    #[cfg(feature = "ipc")]
    pub fn scan_ipc<P: Into<Arc<[std::path::PathBuf]>>>(
        paths: P,
        options: IpcScanOptions,
        n_rows: Option<usize>,
        cache: bool,
        row_index: Option<RowIndex>,
        rechunk: bool,
        cloud_options: Option<CloudOptions>,
    ) -> PolarsResult<Self> {
        use polars_io::is_cloud_url;

        let paths = paths.into();

        // Use first path to get schema.
        let path = paths
            .first()
            .ok_or_else(|| polars_err!(ComputeError: "expected at least 1 path"))?;

        let metadata = if is_cloud_url(path) {
            #[cfg(not(feature = "cloud"))]
            panic!(
                "One or more of the cloud storage features ('aws', 'gcp', ...) must be enabled."
            );

            #[cfg(feature = "cloud")]
            {
                let uri = path.to_string_lossy();
                get_runtime().block_on(async {
                    polars_io::ipc::IpcReaderAsync::from_uri(&uri, cloud_options.as_ref())
                        .await?
                        .metadata()
                        .await
                })?
            }
        } else {
            arrow::io::ipc::read::read_file_metadata(&mut std::io::BufReader::new(
                polars_utils::open_file(path)?,
            ))?
        };

        Ok(DslPlan::Scan {
            paths,
            file_info: FileInfo::new(
                prepare_schema(metadata.schema.as_ref().into(), row_index.as_ref()),
                Some(Arc::clone(&metadata.schema)),
                (None, 0),
            ),
            file_options: FileScanOptions {
                with_columns: None,
                cache,
                n_rows,
                rechunk,
                row_index,
                file_counter: Default::default(),
                // TODO: Support Hive partitioning.
                hive_options: HiveOptions {
                    enabled: false,
                    ..Default::default()
                },
            },
            predicate: None,
            scan_type: FileScan::Ipc {
                options,
                cloud_options,
                metadata: Some(metadata),
            },
        }
        .into())
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "csv")]
    pub fn scan_csv<P: Into<std::path::PathBuf>>(
        path: P,
        separator: u8,
        has_header: bool,
        ignore_errors: bool,
        mut skip_rows: usize,
        n_rows: Option<usize>,
        cache: bool,
        mut schema: Option<Arc<Schema>>,
        schema_overwrite: Option<&Schema>,
        low_memory: bool,
        comment_prefix: Option<CommentPrefix>,
        quote_char: Option<u8>,
        eol_char: u8,
        null_values: Option<NullValues>,
        infer_schema_length: Option<usize>,
        rechunk: bool,
        skip_rows_after_header: usize,
        encoding: CsvEncoding,
        row_index: Option<RowIndex>,
        try_parse_dates: bool,
        raise_if_empty: bool,
        truncate_ragged_lines: bool,
        mut n_threads: Option<usize>,
    ) -> PolarsResult<Self> {
        let path = path.into();
        let mut file = polars_utils::open_file(&path)?;

        let paths = Arc::new([path]);

        let mut magic_nr = [0u8; 4];
        let res_len = file.read(&mut magic_nr)?;
        if res_len < 2 {
            if raise_if_empty {
                polars_bail!(NoData: "empty CSV")
            }
        } else {
            polars_ensure!(
            !is_compressed(&magic_nr),
            ComputeError: "cannot scan compressed csv; use `read_csv` for compressed data",
            );
        }

        file.rewind()?;
        let reader_bytes = get_reader_bytes(&mut file).expect("could not mmap file");

        // TODO! delay inferring schema until absolutely necessary
        // this needs a way to estimated bytes/rows.
        let (mut inferred_schema, rows_read, bytes_read) = infer_file_schema(
            &reader_bytes,
            separator,
            infer_schema_length,
            has_header,
            schema_overwrite,
            &mut skip_rows,
            skip_rows_after_header,
            comment_prefix.as_ref(),
            quote_char,
            eol_char,
            null_values.as_ref(),
            try_parse_dates,
            raise_if_empty,
            &mut n_threads,
        )?;

        if let Some(rc) = &row_index {
            match schema {
                None => {
                    let _ = inferred_schema.insert_at_index(0, rc.name.as_str().into(), IDX_DTYPE);
                },
                Some(inner) => {
                    schema = Some(Arc::new(
                        inner
                            .new_inserting_at_index(0, rc.name.as_str().into(), IDX_DTYPE)
                            .unwrap(),
                    ));
                },
            }
        }

        let schema = schema.unwrap_or_else(|| Arc::new(inferred_schema));
        let n_bytes = reader_bytes.len();
        let estimated_n_rows = (rows_read as f64 / bytes_read as f64 * n_bytes as f64) as usize;

        skip_rows += skip_rows_after_header;
        let file_info = FileInfo::new(schema, None, (None, estimated_n_rows));

        let options = FileScanOptions {
            with_columns: None,
            cache,
            n_rows,
            rechunk,
            row_index,
            file_counter: Default::default(),
            // TODO: Support Hive partitioning.
            hive_options: HiveOptions {
                enabled: false,
                ..Default::default()
            },
        };
        Ok(DslPlan::Scan {
            paths,
            file_info,
            file_options: options,
            predicate: None,
            scan_type: FileScan::Csv {
                options: CsvParserOptions {
                    has_header,
                    separator,
                    ignore_errors,
                    skip_rows,
                    low_memory,
                    comment_prefix,
                    quote_char,
                    eol_char,
                    null_values,
                    encoding,
                    try_parse_dates,
                    raise_if_empty,
                    truncate_ragged_lines,
                    n_threads,
                },
            },
        }
        .into())
    }

    pub fn cache(self) -> Self {
        let input = Arc::new(self.0);
        let id = input.as_ref() as *const DslPlan as usize;
        DslPlan::Cache {
            input,
            id,
            cache_hits: UNLIMITED_CACHE,
        }
        .into()
    }

    pub fn drop(self, to_drop: PlHashSet<String>) -> Self {
        let schema = try_delayed!(self.0.schema(), &self.0, into);

        let mut output_schema = Schema::with_capacity(schema.len().saturating_sub(to_drop.len()));
        let columns = schema
            .iter()
            .filter_map(|(col_name, dtype)| {
                if to_drop.contains(col_name.as_str()) {
                    None
                } else {
                    let out = Some(col(col_name));
                    output_schema.with_column(col_name.clone(), dtype.clone());
                    out
                }
            })
            .collect::<Vec<_>>();

        if columns.is_empty() {
            self.map(
                |_| Ok(DataFrame::empty()),
                AllowedOptimizations::default(),
                Some(Arc::new(|_: &Schema| Ok(Arc::new(Schema::default())))),
                "EMPTY PROJECTION",
            )
        } else {
            DslPlan::Select {
                expr: columns,
                input: Arc::new(self.0),
                schema: Arc::new(output_schema),
                options: ProjectionOptions {
                    run_parallel: false,
                    duplicate_check: false,
                },
            }
            .into()
        }
    }

    pub fn project(self, exprs: Vec<Expr>, options: ProjectionOptions) -> Self {
        let schema = try_delayed!(self.0.schema(), &self.0, into);
        let (exprs, schema) = try_delayed!(prepare_projection(exprs, &schema), &self.0, into);

        if exprs.is_empty() {
            self.map(
                |_| Ok(DataFrame::empty()),
                AllowedOptimizations::default(),
                Some(Arc::new(|_: &Schema| Ok(Arc::new(Schema::default())))),
                "EMPTY PROJECTION",
            )
        } else {
            DslPlan::Select {
                expr: exprs,
                input: Arc::new(self.0),
                schema: Arc::new(schema),
                options,
            }
            .into()
        }
    }

    pub fn fill_null(self, fill_value: Expr) -> Self {
        let schema = try_delayed!(self.0.schema(), &self.0, into);
        let exprs = schema
            .iter_names()
            .map(|name| col(name).fill_null(fill_value.clone()))
            .collect();
        self.project(exprs, Default::default())
    }

    pub fn drop_nulls(self, subset: Option<Vec<Expr>>) -> Self {
        match subset {
            None => {
                let predicate =
                    try_delayed!(all_horizontal([col("*").is_not_null()]), &self.0, into);
                self.filter(predicate)
            },
            Some(subset) => {
                let predicate = try_delayed!(
                    all_horizontal(
                        subset
                            .into_iter()
                            .map(|e| e.is_not_null())
                            .collect::<Vec<_>>(),
                    ),
                    &self.0,
                    into
                );
                self.filter(predicate)
            },
        }
    }

    pub fn fill_nan(self, fill_value: Expr) -> Self {
        let schema = try_delayed!(self.0.schema(), &self.0, into);

        let exprs = schema
            .iter()
            .filter_map(|(name, dtype)| match dtype {
                DataType::Float32 | DataType::Float64 => {
                    Some(col(name).fill_nan(fill_value.clone()).alias(name))
                },
                _ => None,
            })
            .collect();
        self.with_columns(
            exprs,
            ProjectionOptions {
                run_parallel: false,
                duplicate_check: false,
            },
        )
    }

    pub fn with_columns(self, exprs: Vec<Expr>, options: ProjectionOptions) -> Self {
        if exprs.is_empty() {
            return self;
        }

        DslPlan::HStack {
            input: Arc::new(self.0),
            exprs,
            options,
        }
        .into()
    }

    pub fn add_err(self, err: PolarsError) -> Self {
        DslPlan::Error {
            input: Arc::new(self.0),
            err: err.into(),
        }
        .into()
    }

    pub fn with_context(self, contexts: Vec<DslPlan>) -> Self {
        DslPlan::ExtContext {
            input: Arc::new(self.0),
            contexts,
        }
        .into()
    }

    /// Apply a filter
    pub fn filter(self, predicate: Expr) -> Self {
        DslPlan::Filter {
            predicate,
            input: Arc::new(self.0),
        }
        .into()
    }

    pub fn group_by<E: AsRef<[Expr]>>(
        self,
        keys: Vec<Expr>,
        aggs: E,
        apply: Option<(Arc<dyn DataFrameUdf>, SchemaRef)>,
        maintain_order: bool,
        #[cfg(feature = "dynamic_group_by")] dynamic_options: Option<DynamicGroupOptions>,
        #[cfg(feature = "dynamic_group_by")] rolling_options: Option<RollingGroupOptions>,
    ) -> Self {
        let aggs = aggs.as_ref().to_vec();
        let options = GroupbyOptions {
            #[cfg(feature = "dynamic_group_by")]
            dynamic: dynamic_options,
            #[cfg(feature = "dynamic_group_by")]
            rolling: rolling_options,
            slice: None,
        };

        DslPlan::GroupBy {
            input: Arc::new(self.0),
            keys,
            aggs,
            apply,
            maintain_order,
            options: Arc::new(options),
        }
        .into()
    }

    pub fn build(self) -> DslPlan {
        self.0
    }

    pub fn from_existing_df(df: DataFrame) -> Self {
        let schema = Arc::new(df.schema());
        DslPlan::DataFrameScan {
            df: Arc::new(df),
            schema,
            output_schema: None,
            projection: None,
            selection: None,
        }
        .into()
    }

    pub fn sort(self, by_column: Vec<Expr>, sort_options: SortMultipleOptions) -> Self {
        DslPlan::Sort {
            input: Arc::new(self.0),
            by_column,
            slice: None,
            sort_options,
        }
        .into()
    }

    pub fn explode(self, columns: Vec<Expr>) -> Self {
        let schema = try_delayed!(self.0.schema(), &self.0, into);
        let columns = try_delayed!(rewrite_projections(columns, &schema, &[]), &self.0, into);

        // columns to string
        let columns = columns
            .iter()
            .map(|e| {
                if let Expr::Column(name) = e {
                    name.clone()
                } else {
                    panic!("expected column expression")
                }
            })
            .collect::<Arc<[Arc<str>]>>();

        let mut schema = (**schema).clone();
        try_delayed!(explode_schema(&mut schema, &columns), &self.0, into);

        DslPlan::MapFunction {
            input: Arc::new(self.0),
            function: FunctionNode::Explode {
                columns,
                schema: Arc::new(schema),
            },
        }
        .into()
    }

    pub fn melt(self, args: Arc<MeltArgs>) -> Self {
        let schema = try_delayed!(self.0.schema(), &self.0, into);
        let schema = det_melt_schema(&args, &schema);
        DslPlan::MapFunction {
            input: Arc::new(self.0),
            function: FunctionNode::Melt { args, schema },
        }
        .into()
    }

    pub fn row_index(self, name: &str, offset: Option<IdxSize>) -> Self {
        DslPlan::MapFunction {
            input: Arc::new(self.0),
            function: FunctionNode::RowIndex {
                name: ColumnName::from(name),
                offset,
                schema: Default::default(),
            },
        }
        .into()
    }

    pub fn distinct(self, options: DistinctOptions) -> Self {
        DslPlan::Distinct {
            input: Arc::new(self.0),
            options,
        }
        .into()
    }

    pub fn slice(self, offset: i64, len: IdxSize) -> Self {
        DslPlan::Slice {
            input: Arc::new(self.0),
            offset,
            len,
        }
        .into()
    }

    pub fn join(
        self,
        other: DslPlan,
        left_on: Vec<Expr>,
        right_on: Vec<Expr>,
        options: Arc<JoinOptions>,
    ) -> Self {
        for e in left_on.iter().chain(right_on.iter()) {
            if has_expr(e, |e| matches!(e, Expr::Alias(_, _))) {
                return DslPlan::Error {
                    input: Arc::new(self.0),
                    err: polars_err!(
                        ComputeError:
                        "'alias' is not allowed in a join key, use 'with_columns' first",
                    )
                    .into(),
                }
                .into();
            }
        }

        let schema_left = try_delayed!(self.0.schema(), &self.0, into);
        let schema_right = try_delayed!(other.schema(), &self.0, into);

        let schema = try_delayed!(
            det_join_schema(&schema_left, &schema_right, &left_on, &right_on, &options),
            self.0,
            into
        );

        DslPlan::Join {
            input_left: Arc::new(self.0),
            input_right: Arc::new(other),
            schema,
            left_on,
            right_on,
            options,
        }
        .into()
    }
    pub fn map_private(self, function: FunctionNode) -> Self {
        DslPlan::MapFunction {
            input: Arc::new(self.0),
            function,
        }
        .into()
    }

    #[cfg(feature = "python")]
    pub fn map_python(
        self,
        function: PythonFunction,
        optimizations: AllowedOptimizations,
        schema: Option<SchemaRef>,
        validate_output: bool,
    ) -> Self {
        DslPlan::MapFunction {
            input: Arc::new(self.0),
            function: FunctionNode::OpaquePython {
                function,
                schema,
                predicate_pd: optimizations.predicate_pushdown,
                projection_pd: optimizations.projection_pushdown,
                streamable: optimizations.streaming,
                validate_output,
            },
        }
        .into()
    }

    pub fn map<F>(
        self,
        function: F,
        optimizations: AllowedOptimizations,
        schema: Option<Arc<dyn UdfSchema>>,
        name: &'static str,
    ) -> Self
    where
        F: DataFrameUdf + 'static,
    {
        let function = Arc::new(function);

        DslPlan::MapFunction {
            input: Arc::new(self.0),
            function: FunctionNode::Opaque {
                function,
                schema,
                predicate_pd: optimizations.predicate_pushdown,
                projection_pd: optimizations.projection_pushdown,
                streamable: optimizations.streaming,
                fmt_str: name,
            },
        }
        .into()
    }
}
