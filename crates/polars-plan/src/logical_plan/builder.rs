#[cfg(feature = "csv")]
use std::io::{Read, Seek};

use polars_core::frame::explode::MeltArgs;
use polars_core::prelude::*;
#[cfg(feature = "parquet")]
use polars_io::cloud::CloudOptions;
#[cfg(feature = "ipc")]
use polars_io::ipc::IpcReader;
#[cfg(all(feature = "parquet", feature = "async"))]
use polars_io::parquet::ParquetAsyncReader;
#[cfg(feature = "parquet")]
use polars_io::parquet::ParquetReader;
#[cfg(all(feature = "cloud", feature = "parquet"))]
use polars_io::pl_async::get_runtime;
#[cfg(any(
    feature = "parquet",
    feature = "parquet_async",
    feature = "csv",
    feature = "ipc"
))]
use polars_io::RowCount;
#[cfg(feature = "csv")]
use polars_io::{
    csv::utils::{infer_file_schema, is_compressed},
    csv::CommentPrefix,
    csv::CsvEncoding,
    csv::NullValues,
    mmap::ReaderFactory,
    utils::get_reader_bytes,
};

use super::builder_functions::*;
use crate::dsl::functions::horizontal::all_horizontal;
use crate::logical_plan::functions::FunctionNode;
use crate::logical_plan::projection::{is_regex_projection, rewrite_projections};
use crate::logical_plan::schema::{det_join_schema, FileInfo};
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

pub struct LogicalPlanBuilder(pub LogicalPlan);

impl From<LogicalPlan> for LogicalPlanBuilder {
    fn from(lp: LogicalPlan) -> Self {
        LogicalPlanBuilder(lp)
    }
}

fn format_err(msg: &str, input: &LogicalPlan) -> String {
    format!("{msg}\n\nError originated just after this operation:\n{input:?}")
}

/// Returns every error or msg: &str as `ComputeError`.
/// It also shows the logical plan node where the error
/// originated.
macro_rules! raise_err {
    ($err:expr, $input:expr, $convert:ident) => {{
        let format_err_outer = |msg: &str| format_err(msg, &$input);

        let err = $err.wrap_msg(&format_err_outer);

        LogicalPlan::Error {
            input: Box::new($input.clone()),
            err: err.into(),
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
fn prepare_schema(mut schema: Schema, row_count: Option<&RowCount>) -> SchemaRef {
    if let Some(rc) = row_count {
        let _ = schema.insert_at_index(0, rc.name.as_str().into(), IDX_DTYPE);
    }
    Arc::new(schema)
}

impl LogicalPlanBuilder {
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
            row_count: None,
            rechunk: false,
            file_counter: Default::default(),
            hive_partitioning: false,
        };

        Ok(LogicalPlan::Scan {
            reader_factories: Arc::new([]),
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
    pub fn scan_parquet(
        reader_factories: Arc<[ReaderFactory]>,
        n_rows: Option<usize>,
        cache: bool,
        parallel: polars_io::parquet::ParallelStrategy,
        row_count: Option<RowCount>,
        rechunk: bool,
        low_memory: bool,
        cloud_options: Option<CloudOptions>,
        use_statistics: bool,
        hive_partitioning: bool,
    ) -> PolarsResult<Self> {
        use std::path::PathBuf;

        use polars_io::SerReader as _;

        polars_ensure!(reader_factories.len() >= 1, ComputeError: "expected at least 1 path");

        // Use first reader to get schema.
        let reader_factory = reader_factories[0];

        let (schema, reader_schema, num_rows, metadata, uri) = match reader_factory {
            ReaderFactory::RemoteFile { uri } => {
                #[cfg(not(feature = "cloud"))]
                panic!(
                    "One or more of the cloud storage features ('aws', 'gcp', ...) must be enabled."
                );

                #[cfg(feature = "cloud")]
                {
                    get_runtime().block_on(async {
                        let mut reader =
                            ParquetAsyncReader::from_uri(&uri, cloud_options.as_ref(), None, None)
                                .await?;
                        let reader_schema = reader.schema().await?;
                        let num_rows = reader.num_rows().await?;
                        let metadata = reader.get_metadata().await?.clone();

                        let schema = prepare_schema((&reader_schema).into(), row_count.as_ref());
                        PolarsResult::Ok((
                            schema,
                            reader_schema,
                            Some(num_rows),
                            Some(metadata),
                            Some(uri),
                        ))
                    })?
                }
            },
            ReaderFactory::LocalFile { path } => {
                let file = polars_utils::open_file(path)?;
                let mut reader = ParquetReader::new(file);
                let reader_schema = reader.schema()?;
                let schema = prepare_schema((&reader_schema).into(), row_count.as_ref());
                (
                    schema,
                    reader_schema,
                    Some(reader.num_rows()?),
                    Some(reader.get_metadata()?.clone()),
                    None,
                )
            },
        };

        let mut file_info = FileInfo::new(
            schema,
            Some(reader_schema),
            (num_rows, num_rows.unwrap_or(0)),
        );

        // We set the hive partitions of the first path to determine the schema.
        // On iteration the partition values will be re-set per file.
        if hive_partitioning {
            if let Some(uri) = uri {
                let path = PathBuf::from(uri);
                file_info.init_hive_partitions(path.as_path())?;
            }
        }

        let options = FileScanOptions {
            with_columns: None,
            cache,
            n_rows,
            rechunk,
            row_count,
            file_counter: Default::default(),
            hive_partitioning,
        };
        Ok(LogicalPlan::Scan {
            reader_factories,
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
    pub fn scan_ipc(
        reader_factory: ReaderFactory,
        options: IpcScanOptions,
        n_rows: Option<usize>,
        cache: bool,
        row_count: Option<RowCount>,
        rechunk: bool,
    ) -> PolarsResult<Self> {
        use polars_io::SerReader as _;

        let reader = reader_factory.mmapbytesreader()?;
        let mut reader = IpcReader::new(reader);

        let reader_schema = reader.schema()?;
        let mut schema: Schema = (&reader_schema).into();
        if let Some(rc) = &row_count {
            let _ = schema.insert_at_index(0, rc.name.as_str().into(), IDX_DTYPE);
        }

        let num_rows = reader._num_rows()?;
        let file_info = FileInfo::new(Arc::new(schema), Some(reader_schema), (None, num_rows));

        let file_options = FileScanOptions {
            with_columns: None,
            cache,
            n_rows,
            rechunk,
            row_count,
            file_counter: Default::default(),
            // TODO! add
            hive_partitioning: false,
        };
        Ok(LogicalPlan::Scan {
            reader_factories: Arc::new([reader_factory]),
            file_info,
            file_options,
            predicate: None,
            scan_type: FileScan::Ipc { options },
        }
        .into())
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "csv")]
    pub fn scan_csv(
        reader_factory: ReaderFactory,
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
        row_count: Option<RowCount>,
        try_parse_dates: bool,
        raise_if_empty: bool,
        truncate_ragged_lines: bool,
    ) -> PolarsResult<Self> {
        let mut reader = reader_factory.mmapbytesreader()?;
        let reader_factories = Arc::new([reader_factory]);
        let mut magic_nr = [0u8; 2];
        let res = reader.read_exact(&mut magic_nr);
        if raise_if_empty {
            res.map_err(|_| polars_err!(NoData: "empty CSV"))?;
        };
        polars_ensure!(
            !is_compressed(&magic_nr),
            ComputeError: "cannot scan compressed csv; use `read_csv` for compressed data",
        );
        reader.rewind()?;
        let reader_bytes = get_reader_bytes(&mut reader).expect("could not mmap file");

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
        )?;

        if let Some(rc) = &row_count {
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
            row_count,
            file_counter: Default::default(),
            // TODO! add
            hive_partitioning: false,
        };
        Ok(LogicalPlan::Scan {
            reader_factories,
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
                },
            },
        }
        .into())
    }

    pub fn cache(self) -> Self {
        let input = Box::new(self.0);
        let id = input.as_ref() as *const LogicalPlan as usize;
        LogicalPlan::Cache {
            input,
            id,
            count: usize::MAX,
        }
        .into()
    }

    pub fn drop_columns(self, to_drop: PlHashSet<String>) -> Self {
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
                |_| Ok(DataFrame::new_no_checks(vec![])),
                AllowedOptimizations::default(),
                Some(Arc::new(|_: &Schema| Ok(Arc::new(Schema::default())))),
                "EMPTY PROJECTION",
            )
        } else {
            LogicalPlan::Projection {
                expr: columns,
                input: Box::new(self.0),
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
                |_| Ok(DataFrame::new_no_checks(vec![])),
                AllowedOptimizations::default(),
                Some(Arc::new(|_: &Schema| Ok(Arc::new(Schema::default())))),
                "EMPTY PROJECTION",
            )
        } else {
            LogicalPlan::Projection {
                expr: exprs,
                input: Box::new(self.0),
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
        // current schema
        let schema = try_delayed!(self.0.schema(), &self.0, into);
        let mut new_schema = (**schema).clone();
        let (exprs, _) = try_delayed!(prepare_projection(exprs, &schema), &self.0, into);

        let mut output_names = PlHashSet::with_capacity(exprs.len());

        let mut arena = Arena::with_capacity(8);
        for e in &exprs {
            let field = e
                .to_field_amortized(&schema, Context::Default, &mut arena)
                .unwrap();
            if !output_names.insert(field.name().clone()) {
                let msg = format!(
                    "The name: '{}' passed to `LazyFrame.with_columns` is duplicate",
                    field.name()
                );
                return raise_err!(polars_err!(ComputeError: msg), &self.0, into);
            }
            new_schema.with_column(field.name().clone(), field.data_type().clone());
            arena.clear();
        }

        LogicalPlan::HStack {
            input: Box::new(self.0),
            exprs,
            schema: Arc::new(new_schema),
            options,
        }
        .into()
    }

    pub fn add_err(self, err: PolarsError) -> Self {
        LogicalPlan::Error {
            input: Box::new(self.0),
            err: err.into(),
        }
        .into()
    }

    pub fn with_context(self, contexts: Vec<LogicalPlan>) -> Self {
        let mut schema = try_delayed!(self.0.schema(), &self.0, into)
            .as_ref()
            .as_ref()
            .clone();

        for lp in &contexts {
            let other_schema = try_delayed!(lp.schema(), lp, into);

            for fld in other_schema.iter_fields() {
                if schema.get(fld.name()).is_none() {
                    schema.with_column(fld.name, fld.dtype);
                }
            }
        }
        LogicalPlan::ExtContext {
            input: Box::new(self.0),
            contexts,
            schema: Arc::new(schema),
        }
        .into()
    }

    /// Apply a filter
    pub fn filter(self, predicate: Expr) -> Self {
        let schema = try_delayed!(self.0.schema(), &self.0, into);

        let predicate = if has_expr(&predicate, |e| match e {
            Expr::Column(name) => is_regex_projection(name),
            Expr::Wildcard
            | Expr::RenameAlias { .. }
            | Expr::Columns(_)
            | Expr::DtypeColumn(_)
            | Expr::Nth(_) => true,
            _ => false,
        }) {
            let mut rewritten = try_delayed!(
                rewrite_projections(vec![predicate], &schema, &[]),
                &self.0,
                into
            );
            match rewritten.len() {
                1 => {
                    // all good
                    rewritten.pop().unwrap()
                },
                0 => {
                    let msg = "The predicate expanded to zero expressions. \
                        This may for example be caused by a regex not matching column names or \
                        a column dtype match not hitting any dtypes in the DataFrame";
                    return raise_err!(polars_err!(ComputeError: msg), &self.0, into);
                },
                _ => {
                    let mut expanded = String::new();
                    for e in rewritten.iter().take(5) {
                        expanded.push_str(&format!("\t{e},\n"))
                    }
                    // pop latest comma
                    expanded.pop();
                    if rewritten.len() > 5 {
                        expanded.push_str("\t...\n")
                    }

                    let msg = if cfg!(feature = "python") {
                        format!("The predicate passed to 'LazyFrame.filter' expanded to multiple expressions: \n\n{expanded}\n\
                            This is ambiguous. Try to combine the predicates with the 'all' or `any' expression.")
                    } else {
                        format!("The predicate passed to 'LazyFrame.filter' expanded to multiple expressions: \n\n{expanded}\n\
                            This is ambiguous. Try to combine the predicates with the 'all_horizontal' or `any_horizontal' expression.")
                    };
                    return raise_err!(polars_err!(ComputeError: msg), &self.0, into);
                },
            }
        } else {
            predicate
        };

        // Check predicates refer to valid column names here, as this is not
        // checked by predicate pushdown and may otherwise lead to incorrect
        // optimizations. For example:
        //
        // (unoptimized)
        // FILTER [(col("x")) == (1)] FROM
        //   SELECT [col("x").alias("y")] FROM
        //     DF ["x"]; PROJECT 1/1 COLUMNS; SELECTION: "None"
        //
        // (optimized)
        // SELECT [col("x").alias("y")] FROM
        //   DF ["x"]; PROJECT 1/1 COLUMNS; SELECTION: "[(col(\"x\")) == (1)]"
        //                                             ^^^
        // "x" is incorrectly pushed down even though it didn't exist after SELECT
        try_delayed!(
            expr_to_leaf_column_names_iter(&predicate)
                .try_for_each(|c| schema.try_index_of(&c).and(Ok(()))),
            &self.0,
            into
        );

        LogicalPlan::Selection {
            predicate,
            input: Box::new(self.0),
        }
        .into()
    }

    pub fn group_by<E: AsRef<[Expr]>>(
        self,
        keys: Vec<Expr>,
        aggs: E,
        apply: Option<Arc<dyn DataFrameUdf>>,
        maintain_order: bool,
        #[cfg(feature = "dynamic_group_by")] dynamic_options: Option<DynamicGroupOptions>,
        #[cfg(feature = "dynamic_group_by")] rolling_options: Option<RollingGroupOptions>,
    ) -> Self {
        let current_schema = try_delayed!(self.0.schema(), &self.0, into);
        let current_schema = current_schema.as_ref();
        let keys = try_delayed!(
            rewrite_projections(keys, current_schema, &[]),
            &self.0,
            into
        );
        let aggs = try_delayed!(
            rewrite_projections(aggs.as_ref().to_vec(), current_schema, keys.as_ref()),
            &self.0,
            into
        );

        // Initialize schema from keys
        let mut schema = try_delayed!(
            expressions_to_schema(&keys, current_schema, Context::Default),
            &self.0,
            into
        );

        // Add dynamic groupby index column(s)
        #[cfg(feature = "dynamic_group_by")]
        {
            if let Some(options) = rolling_options.as_ref() {
                let name = &options.index_column;
                let dtype = try_delayed!(current_schema.try_get(name), self.0, into);
                schema.with_column(name.clone(), dtype.clone());
            } else if let Some(options) = dynamic_options.as_ref() {
                let name = &options.index_column;
                let dtype = try_delayed!(current_schema.try_get(name), self.0, into);
                if options.include_boundaries {
                    schema.with_column("_lower_boundary".into(), dtype.clone());
                    schema.with_column("_upper_boundary".into(), dtype.clone());
                }
                schema.with_column(name.clone(), dtype.clone());
            }
        }
        let keys_index_len = schema.len();

        // Add aggregation column(s)
        let aggs_schema = try_delayed!(
            expressions_to_schema(&aggs, current_schema, Context::Aggregation),
            &self.0,
            into
        );
        schema.merge(aggs_schema);

        // Make sure aggregation columns do not contain keys or index columns
        if schema.len() < (keys_index_len + aggs.len()) {
            let check_names = || {
                let mut names = PlHashSet::with_capacity(schema.len());
                for expr in aggs.iter().chain(keys.iter()) {
                    let name = expr_output_name(expr)?;
                    if !names.insert(name.clone()) {
                        polars_bail!(duplicate = name);
                    }
                }
                Ok(())
            };
            try_delayed!(check_names(), &self.0, into)
        }

        let options = GroupbyOptions {
            #[cfg(feature = "dynamic_group_by")]
            dynamic: dynamic_options,
            #[cfg(feature = "dynamic_group_by")]
            rolling: rolling_options,
            slice: None,
        };

        LogicalPlan::Aggregate {
            input: Box::new(self.0),
            keys: Arc::new(keys),
            aggs,
            schema: Arc::new(schema),
            apply,
            maintain_order,
            options: Arc::new(options),
        }
        .into()
    }

    pub fn build(self) -> LogicalPlan {
        self.0
    }

    pub fn from_existing_df(df: DataFrame) -> Self {
        let schema = Arc::new(df.schema());
        LogicalPlan::DataFrameScan {
            df: Arc::new(df),
            schema,
            output_schema: None,
            projection: None,
            selection: None,
        }
        .into()
    }

    pub fn sort(
        self,
        by_column: Vec<Expr>,
        descending: Vec<bool>,
        null_last: bool,
        maintain_order: bool,
    ) -> Self {
        let schema = try_delayed!(self.0.schema(), &self.0, into);
        let by_column = try_delayed!(rewrite_projections(by_column, &schema, &[]), &self.0, into);
        LogicalPlan::Sort {
            input: Box::new(self.0),
            by_column,
            args: SortArguments {
                descending,
                nulls_last: null_last,
                slice: None,
                maintain_order,
            },
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

        LogicalPlan::MapFunction {
            input: Box::new(self.0),
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
        LogicalPlan::MapFunction {
            input: Box::new(self.0),
            function: FunctionNode::Melt { args, schema },
        }
        .into()
    }

    pub fn row_count(self, name: &str, offset: Option<IdxSize>) -> Self {
        let mut schema = try_delayed!(self.0.schema(), &self.0, into).into_owned();
        let schema_mut = Arc::make_mut(&mut schema);
        row_count_schema(schema_mut, name);

        LogicalPlan::MapFunction {
            input: Box::new(self.0),
            function: FunctionNode::RowCount {
                name: Arc::from(name),
                offset,
                schema,
            },
        }
        .into()
    }

    pub fn distinct(self, options: DistinctOptions) -> Self {
        LogicalPlan::Distinct {
            input: Box::new(self.0),
            options,
        }
        .into()
    }

    pub fn slice(self, offset: i64, len: IdxSize) -> Self {
        LogicalPlan::Slice {
            input: Box::new(self.0),
            offset,
            len,
        }
        .into()
    }

    pub fn join(
        self,
        other: LogicalPlan,
        left_on: Vec<Expr>,
        right_on: Vec<Expr>,
        options: Arc<JoinOptions>,
    ) -> Self {
        for e in left_on.iter().chain(right_on.iter()) {
            if has_expr(e, |e| matches!(e, Expr::Alias(_, _))) {
                return LogicalPlan::Error {
                    input: Box::new(self.0),
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

        LogicalPlan::Join {
            input_left: Box::new(self.0),
            input_right: Box::new(other),
            schema,
            left_on,
            right_on,
            options,
        }
        .into()
    }
    pub fn map_private(self, function: FunctionNode) -> Self {
        LogicalPlan::MapFunction {
            input: Box::new(self.0),
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
        LogicalPlan::MapFunction {
            input: Box::new(self.0),
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

        LogicalPlan::MapFunction {
            input: Box::new(self.0),
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
