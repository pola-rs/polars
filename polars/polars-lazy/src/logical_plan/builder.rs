use crate::logical_plan::projection::rewrite_projections;
use crate::prelude::*;
use crate::utils;
use crate::utils::{combine_predicates_expr, has_expr};
use parking_lot::Mutex;
use polars_core::prelude::*;
use polars_core::utils::get_supertype;
use polars_io::csv::CsvEncoding;
#[cfg(feature = "csv-file")]
use polars_io::csv_core::utils::infer_file_schema;
#[cfg(feature = "ipc")]
use polars_io::ipc::IpcReader;
#[cfg(feature = "parquet")]
use polars_io::parquet::ParquetReader;
use polars_io::RowCount;
#[cfg(feature = "csv-file")]
use polars_io::{
    csv::NullValues,
    csv_core::utils::{get_reader_bytes, is_compressed},
};
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

pub(crate) fn prepare_projection(exprs: Vec<Expr>, schema: &Schema) -> Result<(Vec<Expr>, Schema)> {
    let exprs = rewrite_projections(exprs, schema, &[]);
    let schema = utils::expressions_to_schema(&exprs, schema, Context::Default)?;
    Ok((exprs, schema))
}

pub struct LogicalPlanBuilder(LogicalPlan);

impl From<LogicalPlan> for LogicalPlanBuilder {
    fn from(lp: LogicalPlan) -> Self {
        LogicalPlanBuilder(lp)
    }
}

macro_rules! try_delayed {
    ($fallible:expr, $input:expr, $convert:ident) => {
        match $fallible {
            Ok(success) => success,
            Err(err) => {
                return LogicalPlan::Error {
                    input: Box::new($input.clone()),
                    err: Arc::new(Mutex::new(Some(err))),
                }
                .$convert()
            }
        }
    };
}

impl LogicalPlanBuilder {
    #[cfg(feature = "parquet")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parquet")))]
    pub fn scan_parquet<P: Into<PathBuf>>(
        path: P,
        n_rows: Option<usize>,
        cache: bool,
        parallel: bool,
        row_count: Option<RowCount>,
    ) -> Result<Self> {
        use polars_io::SerReader as _;

        let path = path.into();
        let file = std::fs::File::open(&path)?;
        let schema = Arc::new(ParquetReader::new(file).schema()?);

        Ok(LogicalPlan::ParquetScan {
            path,
            schema,
            predicate: None,
            aggregate: vec![],
            options: ParquetOptions {
                n_rows,
                with_columns: None,
                cache,
                parallel,
                row_count,
            },
        }
        .into())
    }

    #[cfg(feature = "ipc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ipc")))]
    pub fn scan_ipc<P: Into<PathBuf>>(path: P, options: IpcScanOptions) -> Result<Self> {
        use polars_io::SerReader as _;

        let path = path.into();
        let file = std::fs::File::open(&path)?;
        let schema = Arc::new(IpcReader::new(file).schema()?);

        Ok(LogicalPlan::IpcScan {
            path,
            schema,
            predicate: None,
            aggregate: vec![],
            options,
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
        null_values: Option<NullValues>,
        infer_schema_length: Option<usize>,
        rechunk: bool,
        skip_rows_after_header: usize,
        encoding: CsvEncoding,
        row_count: Option<RowCount>,
    ) -> Result<Self> {
        let path = path.into();
        let mut file = std::fs::File::open(&path)?;
        let mut magic_nr = [0u8; 2];
        file.read_exact(&mut magic_nr)?;
        if is_compressed(&magic_nr) {
            return Err(PolarsError::ComputeError(
                "cannot scan compressed csv; use read_csv for compressed data".into(),
            ));
        }
        file.seek(SeekFrom::Start(0))?;
        let reader_bytes = get_reader_bytes(&mut file).expect("could not mmap file");

        let schema = schema.unwrap_or_else(|| {
            let (schema, _) = infer_file_schema(
                &reader_bytes,
                delimiter,
                infer_schema_length,
                has_header,
                schema_overwrite,
                &mut skip_rows,
                comment_char,
                quote_char,
                null_values.as_ref(),
            )
            .expect("could not read schema");
            Arc::new(schema)
        });
        skip_rows += skip_rows_after_header;
        Ok(LogicalPlan::CsvScan {
            path,
            schema,
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
                null_values,
                rechunk,
                encoding,
                row_count,
            },
            predicate: None,
            aggregate: vec![],
        }
        .into())
    }

    pub fn cache(self) -> Self {
        LogicalPlan::Cache {
            input: Box::new(self.0),
        }
        .into()
    }

    pub fn project(self, exprs: Vec<Expr>) -> Self {
        let (exprs, schema) =
            try_delayed!(prepare_projection(exprs, self.0.schema()), &self.0, into);

        if exprs.is_empty() {
            self.map(
                |_| Ok(DataFrame::new_no_checks(vec![])),
                AllowedOptimizations::default(),
                Some(Arc::new(Schema::default())),
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
        let (exprs, schema) =
            try_delayed!(prepare_projection(exprs, self.0.schema()), &self.0, into);
        if !exprs.is_empty() {
            LogicalPlan::LocalProjection {
                expr: exprs,
                input: Box::new(self.0),
                schema: Arc::new(schema),
            }
            .into()
        } else {
            self
        }
    }

    pub fn fill_null(self, fill_value: Expr) -> Self {
        let schema = self.0.schema();
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
        let schema = self.0.schema();
        let exprs = schema
            .iter_names()
            .map(|name| {
                when(col(name).is_nan())
                    .then(fill_value.clone())
                    .otherwise(col(name))
                    .alias(name)
            })
            .collect();
        self.project_local(exprs)
    }

    pub fn with_columns(self, exprs: Vec<Expr>) -> Self {
        // current schema
        let schema = self.0.schema();
        let mut new_schema = (**schema).clone();
        let (exprs, _) = try_delayed!(prepare_projection(exprs, schema), &self.0, into);

        for e in &exprs {
            let field = e.to_field(schema, Context::Default).unwrap();
            new_schema.with_column(field.name().to_string(), field.data_type().clone());
        }

        LogicalPlan::HStack {
            input: Box::new(self.0),
            exprs,
            schema: Arc::new(new_schema),
        }
        .into()
    }

    /// Apply a filter
    pub fn filter(self, predicate: Expr) -> Self {
        let predicate = if has_expr(&predicate, |e| matches!(e, Expr::Wildcard)) {
            let rewritten = rewrite_projections(vec![predicate], self.0.schema(), &[]);
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
        keys: Arc<Vec<Expr>>,
        aggs: E,
        apply: Option<Arc<dyn DataFrameUdf>>,
        maintain_order: bool,
        dynamic_options: Option<DynamicGroupOptions>,
        rolling_options: Option<RollingGroupOptions>,
    ) -> Self {
        let current_schema = self.0.schema();
        let aggs = rewrite_projections(aggs.as_ref().to_vec(), current_schema, keys.as_ref());

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

        LogicalPlan::Aggregate {
            input: Box::new(self.0),
            keys,
            aggs,
            schema: Arc::new(schema),
            apply,
            maintain_order,
            options: GroupbyOptions {
                dynamic: dynamic_options,
                rolling: rolling_options,
            },
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
            projection: None,
            selection: None,
        }
        .into()
    }

    pub fn sort(self, by_column: Vec<Expr>, reverse: Vec<bool>, null_last: bool) -> Self {
        LogicalPlan::Sort {
            input: Box::new(self.0),
            by_column,
            args: SortArguments {
                reverse,
                nulls_last: null_last,
            },
        }
        .into()
    }

    pub fn explode(self, columns: Vec<Expr>) -> Self {
        let columns = rewrite_projections(columns, self.0.schema(), &[]);
        // columns to string
        let columns = columns
            .iter()
            .map(|e| {
                if let Expr::Column(name) = e {
                    (**name).to_owned()
                } else {
                    panic!("expected column expression")
                }
            })
            .collect();
        LogicalPlan::Explode {
            input: Box::new(self.0),
            columns,
        }
        .into()
    }

    pub fn melt(self, id_vars: Arc<Vec<String>>, value_vars: Arc<Vec<String>>) -> Self {
        let schema = det_melt_schema(&id_vars, &value_vars, self.0.schema());
        LogicalPlan::Melt {
            input: Box::new(self.0),
            id_vars,
            value_vars,
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

    pub fn slice(self, offset: i64, len: u32) -> Self {
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
        let schema_left = self.0.schema();
        let schema_right = other.schema();

        // column names of left table
        let mut names: PlHashSet<&str> = PlHashSet::default();
        let mut new_schema = Schema::with_capacity(schema_left.len() + schema_right.len());

        for (name, dtype) in schema_left.iter() {
            names.insert(name);
            new_schema.with_column(name.to_string(), dtype.clone())
        }

        let right_names: PlHashSet<_> = right_on
            .iter()
            .map(|e| utils::expr_output_name(e).expect("could not find name"))
            .collect();

        for (name, dtype) in schema_right.iter() {
            if !right_names.iter().any(|s| s.as_ref() == name) {
                if names.contains(&**name) {
                    let new_name = format!("{}{}", name, options.suffix.as_ref());
                    new_schema.with_column(new_name, dtype.clone())
                } else {
                    new_schema.with_column(name.to_string(), dtype.clone())
                }
            }
        }

        let schema = Arc::new(new_schema);

        drop(names);
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
    pub fn map<F>(
        self,
        function: F,
        optimizations: AllowedOptimizations,
        schema: Option<SchemaRef>,
        name: &'static str,
    ) -> Self
    where
        F: DataFrameUdf + 'static,
    {
        let options = LogicalPlanUdfOptions {
            predicate_pd: optimizations.predicate_pushdown,
            projection_pd: optimizations.projection_pushdown,
            fmt_str: name,
        };

        LogicalPlan::Udf {
            input: Box::new(self.0),
            function: Arc::new(function),
            options,
            schema,
        }
        .into()
    }
}

pub(crate) fn det_melt_schema(
    id_vars: &[String],
    value_vars: &[String],
    input_schema: &Schema,
) -> SchemaRef {
    let mut new_schema = Schema::from(
        id_vars
            .iter()
            .map(|id| Field::new(id, input_schema.get(id).unwrap().clone())),
    );
    new_schema.with_column("variable".to_string(), DataType::Utf8);

    // We need to determine the supertype of all value columns.
    let mut st = None;

    // take all columns that are not in `id_vars` as `value_var`
    if value_vars.is_empty() {
        let id_vars = PlHashSet::from_iter(id_vars);
        for (name, dtype) in input_schema.iter() {
            if !id_vars.contains(name) {
                match &st {
                    None => st = Some(dtype.clone()),
                    Some(st_) => st = Some(get_supertype(st_, dtype).unwrap()),
                }
            }
        }
    } else {
        for name in value_vars {
            let dtype = input_schema.get(name).unwrap();
            match &st {
                None => st = Some(dtype.clone()),
                Some(st_) => st = Some(get_supertype(st_, dtype).unwrap()),
            }
        }
    }
    new_schema.with_column("value".to_string(), st.unwrap());
    Arc::new(new_schema)
}
