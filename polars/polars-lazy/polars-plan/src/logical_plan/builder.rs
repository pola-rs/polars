use std::io::{Read, Seek};
use std::ops::Deref;
use std::path::PathBuf;

#[cfg(feature = "parquet")]
use polars_core::cloud::CloudOptions;
use polars_core::frame::_duplicate_err;
use polars_core::frame::explode::MeltArgs;
use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
#[cfg(feature = "ipc")]
use polars_io::ipc::IpcReader;
#[cfg(all(feature = "parquet", feature = "async"))]
use polars_io::parquet::ParquetAsyncReader;
#[cfg(feature = "parquet")]
use polars_io::parquet::ParquetReader;
use polars_io::RowCount;
#[cfg(feature = "csv-file")]
use polars_io::{
    csv::utils::{get_reader_bytes, infer_file_schema, is_compressed},
    csv::CsvEncoding,
    csv::NullValues,
};

use crate::logical_plan::functions::FunctionNode;
use crate::logical_plan::projection::{is_regex_projection, rewrite_projections};
use crate::logical_plan::schema::{det_join_schema, FileInfo};
use crate::prelude::*;
use crate::utils;
use crate::utils::{combine_predicates_expr, has_expr};

pub(crate) fn prepare_projection(
    exprs: Vec<Expr>,
    schema: &Schema,
) -> PolarsResult<(Vec<Expr>, Schema)> {
    let exprs = rewrite_projections(exprs, schema, &[])?;
    let schema = utils::expressions_to_schema(&exprs, schema, Context::Default)?;
    Ok((exprs, schema))
}

pub struct LogicalPlanBuilder(LogicalPlan);

impl From<LogicalPlan> for LogicalPlanBuilder {
    fn from(lp: LogicalPlan) -> Self {
        LogicalPlanBuilder(lp)
    }
}

fn format_err(msg: &str, input: &LogicalPlan) -> String {
    format!(
        "{msg}\n\n> Error originated just after operation: '{input:?}'\n\
    This operation could not be added to the plan.",
    )
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

impl LogicalPlanBuilder {
    pub fn anonymous_scan(
        function: Arc<dyn AnonymousScan>,
        schema: Option<Schema>,
        infer_schema_length: Option<usize>,
        skip_rows: Option<usize>,
        n_rows: Option<usize>,
        name: &'static str,
    ) -> PolarsResult<Self> {
        let schema = Arc::new(match schema {
            Some(s) => s,
            None => function.schema(infer_schema_length)?,
        });

        let file_info = FileInfo {
            schema: schema.clone(),
            row_estimation: (n_rows, n_rows.unwrap_or(usize::MAX)),
        };
        Ok(LogicalPlan::AnonymousScan {
            function,
            file_info,
            predicate: None,
            options: AnonymousScanOptions {
                fmt_str: name,
                schema,
                skip_rows,
                n_rows,
                output_schema: None,
                with_columns: None,
                predicate: None,
            },
        }
        .into())
    }

    #[cfg(any(feature = "parquet", feature = "parquet_async"))]
    #[allow(clippy::too_many_arguments)]
    pub fn scan_parquet<P: Into<PathBuf>>(
        path: P,
        n_rows: Option<usize>,
        cache: bool,
        parallel: polars_io::parquet::ParallelStrategy,
        row_count: Option<RowCount>,
        rechunk: bool,
        low_memory: bool,
        cloud_options: Option<CloudOptions>,
    ) -> PolarsResult<Self> {
        use polars_io::{is_cloud_url, SerReader as _};

        let path = path.into();
        let file_info: PolarsResult<FileInfo> = if is_cloud_url(&path) {
            #[cfg(not(feature = "async"))]
            panic!(
                "One or more of the cloud storage features ('aws', 'gcp', ...) must be enabled."
            );

            #[cfg(feature = "async")]
            {
                let uri = path.to_string_lossy();
                let (schema, num_rows) =
                    ParquetAsyncReader::file_info(&uri, cloud_options.as_ref())?;
                Ok(FileInfo {
                    schema: Arc::new(schema),
                    row_estimation: (Some(num_rows), num_rows),
                })
            }
        } else {
            let file = std::fs::File::open(&path)?;
            let mut reader = ParquetReader::new(file);
            let schema = Arc::new(reader.schema()?);
            let num_rows = reader.num_rows()?;
            Ok(FileInfo {
                schema,
                row_estimation: (Some(num_rows), num_rows),
            })
        };
        let file_info = file_info?;

        Ok(LogicalPlan::ParquetScan {
            path,
            file_info,
            predicate: None,
            options: ParquetOptions {
                n_rows,
                with_columns: None,
                cache,
                parallel,
                row_count,
                rechunk,
                file_counter: Default::default(),
                low_memory,
            },
            cloud_options,
        }
        .into())
    }

    #[cfg(feature = "ipc")]
    pub fn scan_ipc<P: Into<PathBuf>>(path: P, options: IpcScanOptions) -> PolarsResult<Self> {
        use polars_io::SerReader as _;

        let path = path.into();
        let file = std::fs::File::open(&path)?;
        let mut reader = IpcReader::new(file);
        let schema = Arc::new(reader.schema()?);

        let num_rows = reader._num_rows()?;
        let file_info = FileInfo {
            schema,
            row_estimation: (None, num_rows),
        };
        Ok(LogicalPlan::IpcScan {
            path,
            file_info,
            predicate: None,
            options: options.into(),
        }
        .into())
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "csv-file")]
    pub fn scan_csv<P: Into<PathBuf>>(
        path: P,
        delimiter: u8,
        has_header: bool,
        ignore_errors: bool,
        mut skip_rows: usize,
        n_rows: Option<usize>,
        cache: bool,
        schema: Option<Arc<Schema>>,
        schema_overwrite: Option<&Schema>,
        low_memory: bool,
        comment_char: Option<u8>,
        quote_char: Option<u8>,
        eol_char: u8,
        null_values: Option<NullValues>,
        infer_schema_length: Option<usize>,
        rechunk: bool,
        skip_rows_after_header: usize,
        encoding: CsvEncoding,
        row_count: Option<RowCount>,
        parse_dates: bool,
    ) -> PolarsResult<Self> {
        let path = path.into();
        let mut file = std::fs::File::open(&path)?;
        let mut magic_nr = [0u8; 2];
        file.read_exact(&mut magic_nr)
            .map_err(|_| PolarsError::NoData("empty csv".into()))?;

        if is_compressed(&magic_nr) {
            return Err(PolarsError::ComputeError(
                "cannot scan compressed csv; use read_csv for compressed data".into(),
            ));
        }
        file.rewind()?;
        let reader_bytes = get_reader_bytes(&mut file).expect("could not mmap file");

        // TODO! delay inferring schema until absolutely necessary
        // this needs a way to estimated bytes/rows.
        let (inferred_schema, rows_read, bytes_read) = infer_file_schema(
            &reader_bytes,
            delimiter,
            infer_schema_length,
            has_header,
            schema_overwrite,
            &mut skip_rows,
            skip_rows_after_header,
            comment_char,
            quote_char,
            eol_char,
            null_values.as_ref(),
            parse_dates,
        )?;

        let schema = schema.unwrap_or_else(|| Arc::new(inferred_schema));
        let n_bytes = reader_bytes.deref().len();
        let estimated_n_rows = (rows_read as f64 / bytes_read as f64 * n_bytes as f64) as usize;

        skip_rows += skip_rows_after_header;
        let file_info = FileInfo {
            schema,
            row_estimation: (None, estimated_n_rows),
        };
        Ok(LogicalPlan::CsvScan {
            path,
            file_info,
            options: CsvParserOptions {
                has_header,
                delimiter,
                ignore_errors,
                skip_rows,
                n_rows,
                with_columns: None,
                low_memory,
                cache,
                comment_char,
                quote_char,
                eol_char,
                null_values,
                rechunk,
                encoding,
                row_count,
                parse_dates,
                file_counter: Default::default(),
            },
            predicate: None,
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

    pub fn project(self, exprs: Vec<Expr>) -> Self {
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
            }
            .into()
        }
    }

    pub fn project_local(self, exprs: Vec<Expr>) -> Self {
        let schema = try_delayed!(self.0.schema(), &self.0, into);
        let (exprs, schema) = try_delayed!(prepare_projection(exprs, &schema), &self.0, into);
        LogicalPlan::LocalProjection {
            expr: exprs,
            input: Box::new(self.0),
            schema: Arc::new(schema),
        }
        .into()
    }

    pub fn fill_null(self, fill_value: Expr) -> Self {
        let schema = try_delayed!(self.0.schema(), &self.0, into);
        let exprs = schema
            .iter_names()
            .map(|name| {
                when(col(name).is_null())
                    .then(fill_value.clone())
                    .otherwise(col(name))
                    .alias(name)
            })
            .collect();
        self.project_local(exprs)
    }

    pub fn fill_nan(self, fill_value: Expr) -> Self {
        let schema = try_delayed!(self.0.schema(), &self.0, into);

        let exprs = schema
            .iter()
            .filter_map(|(name, dtype)| match dtype {
                DataType::Float32 | DataType::Float64 => {
                    Some(col(name).fill_nan(fill_value.clone()).alias(name))
                }
                _ => None,
            })
            .collect();
        self.with_columns(exprs)
    }

    pub fn with_columns(self, exprs: Vec<Expr>) -> Self {
        // current schema
        let schema = try_delayed!(self.0.schema(), &self.0, into);
        let mut new_schema = (**schema).clone();
        let (exprs, _) = try_delayed!(prepare_projection(exprs, &schema), &self.0, into);

        let mut arena = Arena::with_capacity(8);
        for e in &exprs {
            let field = e
                .to_field_amortized(&schema, Context::Default, &mut arena)
                .unwrap();
            new_schema.with_column(field.name().to_string(), field.data_type().clone());
            arena.clear();
        }

        LogicalPlan::HStack {
            input: Box::new(self.0),
            exprs,
            schema: Arc::new(new_schema),
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
        let predicate = if has_expr(&predicate, |e| match e {
            Expr::Column(name) => is_regex_projection(name),
            Expr::Wildcard | Expr::RenameAlias { .. } | Expr::Columns(_) | Expr::DtypeColumn(_) => {
                true
            }
            _ => false,
        }) {
            let schema = try_delayed!(self.0.schema(), &self.0, into);
            let rewritten = try_delayed!(
                rewrite_projections(vec![predicate], &schema, &[]),
                &self.0,
                into
            );
            if rewritten.is_empty() {
                let msg = "The predicate expanded to zero expressions. \
                This may for example be caused by a regex not matching column names or \
                a column dtype match not hitting any dtypes in the DataFrame";
                return raise_err!(PolarsError::ComputeError(msg.into()), &self.0, into);
            }
            combine_predicates_expr(rewritten.into_iter())
        } else {
            predicate
        };
        LogicalPlan::Selection {
            predicate,
            input: Box::new(self.0),
        }
        .into()
    }

    pub fn groupby<E: AsRef<[Expr]>>(
        self,
        keys: Vec<Expr>,
        aggs: E,
        apply: Option<Arc<dyn DataFrameUdf>>,
        maintain_order: bool,
        #[cfg(feature = "dynamic_groupby")] dynamic_options: Option<DynamicGroupOptions>,
        #[cfg(feature = "dynamic_groupby")] rolling_options: Option<RollingGroupOptions>,
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

        let mut schema = try_delayed!(
            utils::expressions_to_schema(&keys, current_schema, Context::Default),
            &self.0,
            into
        );
        let other = try_delayed!(
            utils::expressions_to_schema(&aggs, current_schema, Context::Aggregation),
            &self.0,
            into
        );
        schema.merge(other);

        if schema.len() < keys.len() + aggs.len() {
            let check_names = || {
                let mut names = PlHashSet::with_capacity(schema.len());
                for expr in aggs.iter().chain(keys.iter()) {
                    let name = expr_output_name(expr)?;
                    if !names.insert(name.clone()) {
                        return _duplicate_err(name.as_ref());
                    }
                }
                Ok(())
            };
            try_delayed!(check_names(), &self.0, into)
        }

        #[cfg(feature = "dynamic_groupby")]
        {
            let index_columns = &[
                rolling_options
                    .as_ref()
                    .map(|options| &options.index_column),
                dynamic_options
                    .as_ref()
                    .map(|options| &options.index_column),
            ];
            for &name in index_columns.iter().flatten() {
                let dtype = try_delayed!(
                    current_schema
                        .get(name)
                        .ok_or_else(|| PolarsError::ColumnNotFound(name.to_string().into())),
                    self.0,
                    into
                );
                schema.with_column(name.clone(), dtype.clone());
            }
        }

        #[cfg(feature = "dynamic_groupby")]
        let options = GroupbyOptions {
            dynamic: dynamic_options,
            rolling: rolling_options,
            slice: None,
        };

        #[cfg(not(feature = "dynamic_groupby"))]
        let options = GroupbyOptions { slice: None };

        LogicalPlan::Aggregate {
            input: Box::new(self.0),
            keys: Arc::new(keys),
            aggs,
            schema: Arc::new(schema),
            apply,
            maintain_order,
            options,
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

    pub fn sort(self, by_column: Vec<Expr>, reverse: Vec<bool>, null_last: bool) -> Self {
        let schema = try_delayed!(self.0.schema(), &self.0, into);
        let by_column = try_delayed!(rewrite_projections(by_column, &schema, &[]), &self.0, into);
        LogicalPlan::Sort {
            input: Box::new(self.0),
            by_column,
            args: SortArguments {
                reverse,
                nulls_last: null_last,
                slice: None,
            },
        }
        .into()
    }

    pub fn explode(self, columns: Vec<Expr>) -> Self {
        let schema = try_delayed!(self.0.schema(), &self.0, into);
        let columns = try_delayed!(rewrite_projections(columns, &schema, &[]), &self.0, into);

        let mut schema = (**schema).clone();

        // columns to string
        let columns = columns
            .iter()
            .map(|e| {
                if let Expr::Column(name) = e {
                    if let Some(DataType::List(inner)) = schema.get(name) {
                        let inner = *inner.clone();
                        schema.with_column(name.to_string(), inner);
                    }

                    (**name).to_owned()
                } else {
                    panic!("expected column expression")
                }
            })
            .collect();
        LogicalPlan::Explode {
            input: Box::new(self.0),
            columns,
            schema: Arc::new(schema),
        }
        .into()
    }

    pub fn melt(self, args: Arc<MeltArgs>) -> Self {
        let schema = try_delayed!(self.0.schema(), &self.0, into);
        let schema = det_melt_schema(&args, &schema);
        LogicalPlan::Melt {
            input: Box::new(self.0),
            args,
            schema,
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
        options: JoinOptions,
    ) -> Self {
        for e in left_on.iter().chain(right_on.iter()) {
            if has_expr(e, |e| matches!(e, Expr::Alias(_, _))) {
                return LogicalPlan::Error {
                    input: Box::new(self.0),
                    err: PolarsError::ComputeError(
                        "'alias' is not allowed in a join key. Use 'with_columns' first.".into(),
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

pub(crate) fn det_melt_schema(args: &MeltArgs, input_schema: &Schema) -> SchemaRef {
    let mut new_schema = Schema::from(
        args.id_vars
            .iter()
            .map(|id| Field::new(id, input_schema.get(id).unwrap().clone())),
    );
    let variable_name = args
        .variable_name
        .as_ref()
        .cloned()
        .unwrap_or_else(|| "variable".to_string());
    let value_name = args
        .value_name
        .as_ref()
        .cloned()
        .unwrap_or_else(|| "value".to_string());

    new_schema.with_column(variable_name, DataType::Utf8);

    // We need to determine the supertype of all value columns.
    let mut st = None;

    // take all columns that are not in `id_vars` as `value_var`
    if args.value_vars.is_empty() {
        let id_vars = PlHashSet::from_iter(&args.id_vars);
        for (name, dtype) in input_schema.iter() {
            if !id_vars.contains(name) {
                match &st {
                    None => st = Some(dtype.clone()),
                    Some(st_) => st = Some(try_get_supertype(st_, dtype).unwrap()),
                }
            }
        }
    } else {
        for name in &args.value_vars {
            let dtype = input_schema.get(name).unwrap();
            match &st {
                None => st = Some(dtype.clone()),
                Some(st_) => st = Some(try_get_supertype(st_, dtype).unwrap()),
            }
        }
    }
    new_schema.with_column(value_name, st.unwrap());
    Arc::new(new_schema)
}
