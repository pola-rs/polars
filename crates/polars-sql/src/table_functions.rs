use std::str::FromStr;

#[cfg(feature = "csv")]
use polars_core::prelude::PlSmallStr;
use polars_core::prelude::{
    DataFrame, ExplodeOptions, PolarsError, PolarsResult, polars_bail, polars_ensure,
};
#[cfg(feature = "csv")]
use polars_lazy::prelude::LazyCsvReader;
use polars_lazy::prelude::{IntoLazy, LazyFrame};
use polars_plan::utils::expr_to_leaf_column_names_iter;
use polars_utils::pl_path::PlRefPath;
use sqlparser::ast::{
    Expr as SQLExpr, FunctionArg as SQLFunctionArg, FunctionArgExpr as SQLFunctionArgExpr,
    Value as SQLValue, ValueWithSpan as SQLValueWithSpan,
};

use crate::SQLContext;
use crate::context::SubqueryShape;
use crate::functions::series_expr;
use crate::sql_expr::parse_sql_expr;
#[cfg(any(feature = "csv", feature = "json"))]
use crate::table_function_args::TableFunctionOptions;
#[cfg(feature = "csv")]
use crate::table_function_args::{DtypeOverrides, StringValues};

/// Table functions that are supported by Polars
#[allow(clippy::enum_variant_names)]
pub(crate) enum PolarsTableFunctions {
    /// SQL 'read_csv' function, with optional named options as in Python's `scan_csv`.
    /// ```sql
    /// SELECT * FROM read_csv('path/to/file.csv')
    /// SELECT * FROM read_csv('path/to/file.csv', has_header => false, separator => ';')
    /// ```
    #[cfg(feature = "csv")]
    ReadCsv,
    /// SQL 'read_parquet' function.
    /// ```sql
    /// SELECT * FROM read_parquet('path/to/file.parquet')
    /// ```
    #[cfg(feature = "parquet")]
    ReadParquet,
    /// SQL 'read_ipc' function.
    /// ```sql
    /// SELECT * FROM read_ipc('path/to/file.ipc')
    /// ```
    #[cfg(feature = "ipc")]
    ReadIpc,
    /// SQL 'read_json' function: reads newline-delimited JSON from a path only; for named options,
    /// use `read_ndjson`. It stays path-only so that it can later read JSON documents, as Python's
    /// `read_json` does, without a change of meaning for its options.
    /// ```sql
    /// SELECT * FROM read_json('path/to/file.ndjson')
    /// ```
    #[cfg(feature = "json")]
    ReadJson,
    /// SQL 'read_ndjson' function, with optional named options as in Python's `scan_ndjson`.
    /// ```sql
    /// SELECT * FROM read_ndjson('path/to/file.ndjson')
    /// SELECT * FROM read_ndjson('path/to/file.ndjson', infer_schema_length => NULL)
    /// ```
    #[cfg(feature = "json")]
    ReadNdjson,
    /// SQL 'generate_series' function (stop included).
    /// ```sql
    /// SELECT * FROM generate_series(1, 10, 2)
    /// SELECT * FROM generate_series(DATE '2026-01-01', DATE '2026-01-31', INTERVAL '1 day')
    /// ```
    GenerateSeries,
    /// SQL 'range' function (stop not included).
    /// ```sql
    /// SELECT * FROM range(0, 10, 2)
    /// ```
    Range,
}

impl FromStr for PolarsTableFunctions {
    type Err = PolarsError;

    #[allow(unreachable_code)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            #[cfg(feature = "csv")]
            "read_csv" => PolarsTableFunctions::ReadCsv,
            #[cfg(feature = "parquet")]
            "read_parquet" => PolarsTableFunctions::ReadParquet,
            #[cfg(feature = "ipc")]
            "read_ipc" => PolarsTableFunctions::ReadIpc,
            #[cfg(feature = "json")]
            "read_json" => PolarsTableFunctions::ReadJson,
            #[cfg(feature = "json")]
            "read_ndjson" => PolarsTableFunctions::ReadNdjson,
            "generate_series" => PolarsTableFunctions::GenerateSeries,
            "range" => PolarsTableFunctions::Range,
            _ => polars_bail!(SQLInterface: "'{}' is not a supported table function", s),
        })
    }
}

impl PolarsTableFunctions {
    #[allow(unreachable_code)]
    pub(crate) fn execute(
        &self,
        args: &[SQLFunctionArg],
        ctx: &mut SQLContext,
    ) -> PolarsResult<(String, LazyFrame)> {
        let (path, lf): (PlRefPath, LazyFrame) = match self {
            #[cfg(feature = "csv")]
            PolarsTableFunctions::ReadCsv => self.read_csv(args)?,
            #[cfg(feature = "parquet")]
            PolarsTableFunctions::ReadParquet => self.read_parquet(args)?,
            #[cfg(feature = "ipc")]
            PolarsTableFunctions::ReadIpc => self.read_ipc(args)?,
            #[cfg(feature = "json")]
            PolarsTableFunctions::ReadJson => self.read_json(args)?,
            #[cfg(feature = "json")]
            PolarsTableFunctions::ReadNdjson => self.read_ndjson(args)?,
            PolarsTableFunctions::GenerateSeries => return self.series(args, ctx, true),
            PolarsTableFunctions::Range => return self.series(args, ctx, false),
        };
        Ok((path.to_string(), lf))
    }

    fn series(
        &self,
        args: &[SQLFunctionArg],
        ctx: &mut SQLContext,
        include_end: bool,
    ) -> PolarsResult<(String, LazyFrame)> {
        let fname = if include_end {
            "generate_series"
        } else {
            "range"
        };
        let args = args
            .iter()
            .map(|arg| match arg {
                SQLFunctionArg::Unnamed(SQLFunctionArgExpr::Expr(e)) => Ok(e),
                _ => polars_bail!(SQLSyntax: "`{}` expects positional arguments; found {}", fname, arg),
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        let mut expr = series_expr(fname, &args, include_end, |e| {
            let expr = parse_sql_expr(e, ctx, None)?;
            polars_ensure!(
                expr_to_leaf_column_names_iter(&expr).next().is_none(),
                SQLInterface: "`{}` in FROM takes only constant arguments; use UNNEST({}(...)) in the SELECT list for a series per row", fname, fname
            );
            Ok(expr)
        })?;
        let (lf, _) = ctx.process_subqueries(
            DataFrame::empty_with_height(1).lazy(),
            vec![&mut expr],
            SubqueryShape::Scalar,
        )?;
        let lf = lf.select([expr
            .explode(ExplodeOptions {
                empty_as_null: false,
                keep_nulls: false,
            })
            .alias(fname)]);
        Ok((fname.to_string(), lf))
    }

    #[cfg(feature = "csv")]
    fn read_csv(&self, args: &[SQLFunctionArg]) -> PolarsResult<(PlRefPath, LazyFrame)> {
        use std::sync::Arc;

        use polars_io::csv::read::{CsvEncoding, NullValues};
        use polars_lazy::frame::LazyFileListReader;
        use polars_plan::dsl::{ExtraColumnsPolicy, MissingColumnsPolicy};

        let (path_arg, options) = TableFunctionOptions::parse(
            "read_csv",
            args,
            READ_CSV_OPTIONS,
            READ_CSV_EXCLUDED_OPTIONS,
        )?;
        let path = self.get_file_path_from_arg(path_arg)?;

        let has_header = options.bool("has_header")?.unwrap_or(true);
        let schema = options.schema("schema")?;
        let mut reader = LazyCsvReader::new(path.clone())
            .with_has_header(has_header)
            .with_try_parse_dates(options.bool("try_parse_dates")?.unwrap_or(true))
            // As in Python: an empty file only errors when there is no schema to return.
            .with_raise_if_empty(schema.is_none() || has_header)
            .with_schema(schema.map(Arc::new));

        if let Some(separator) = options.byte("separator")? {
            reader = reader.with_separator(separator);
        }
        if let Some(quote_char) = options.nullable_byte("quote_char")? {
            reader = reader.with_quote_char(quote_char);
        }
        if let Some(prefix) = options.nullable_string("comment_prefix")? {
            reader = reader.with_comment_prefix(prefix.map(PlSmallStr::from));
        }
        if let Some(n) = options.usize("skip_rows")? {
            reader = reader.with_skip_rows(n);
        }
        if let Some(n) = options.usize("skip_rows_after_header")? {
            reader = reader.with_skip_rows_after_header(n);
        }
        if let Some(n) = options.usize("skip_lines")? {
            reader = reader.with_skip_lines(n);
        }
        if let Some(encoding) = options.one_of(
            "encoding",
            &[
                ("utf8", CsvEncoding::Utf8),
                ("utf8-lossy", CsvEncoding::LossyUtf8),
            ],
        )? {
            reader = reader.with_encoding(encoding);
        }
        if let Some(names) = options.strings("new_columns")? {
            let names: Vec<PlSmallStr> = names.into_iter().map(PlSmallStr::from).collect();
            reader = reader.with_column_names_overwrite(names.into());
        }
        match options.dtype_overrides("schema_overrides")? {
            Some(DtypeOverrides::ByName(schema)) => {
                reader = reader.with_dtype_overwrite(Some(Arc::new(schema)));
            },
            Some(DtypeOverrides::ByPosition(dtypes)) => {
                reader = reader.with_dtype_overwrite_by_position(Some(Arc::new(dtypes)));
            },
            None => {},
        }
        if let Some(n) = options.nullable_usize("infer_schema_length")? {
            reader = reader.with_infer_schema_length(n);
        }
        if let Some(n) = options.positive_usize("infer_schema_files")? {
            reader = reader.with_infer_schema_files(n);
        }
        // As in Python: `infer_schema => false` reads every column as a string.
        if options.bool("infer_schema")? == Some(false) {
            reader = reader.with_infer_schema_length(Some(0));
        }
        if let Some(decimal_comma) = options.bool("decimal_comma")? {
            reader = reader.with_decimal_comma(decimal_comma);
        }
        let null_values = options
            .string_values("null_values")?
            .map(|values| match values {
                StringValues::One(value) => NullValues::AllColumnsSingle(value.into()),
                StringValues::List(values) => {
                    NullValues::AllColumns(values.into_iter().map(Into::into).collect())
                },
                StringValues::Dict(pairs) => NullValues::Named(
                    pairs
                        .into_iter()
                        .map(|(k, v)| (k.into(), v.into()))
                        .collect(),
                ),
            });
        reader = reader.with_null_values(null_values);
        if let Some(ignore_errors) = options.bool("ignore_errors")? {
            reader = reader.with_ignore_errors(ignore_errors);
        }
        if let Some(policy) = options.one_of(
            "missing_columns",
            &[
                ("insert", MissingColumnsPolicy::Insert),
                ("raise", MissingColumnsPolicy::Raise),
            ],
        )? {
            reader = reader.with_missing_columns_policy(Some(policy));
        }
        let extra_columns = options.one_of(
            "extra_columns",
            &[
                ("ignore", ExtraColumnsPolicy::Ignore),
                ("raise", ExtraColumnsPolicy::Raise),
            ],
        )?;
        // As in Python: ignoring extra columns also truncates rows with extra fields, unless
        // `truncate_ragged_lines` is given. The reader doesn't apply this rule itself, so `scan_csv`
        // and `read_csv` each do, as for `raise_if_empty` above.
        let ignore_extra_columns = extra_columns == Some(ExtraColumnsPolicy::Ignore);
        let truncate = options.bool("truncate_ragged_lines")?;
        polars_ensure!(
            !(ignore_extra_columns && truncate == Some(false)),
            SQLSyntax: "`read_csv` cannot set truncate_ragged_lines => false with extra_columns => 'ignore'"
        );
        if let Some(truncate) = truncate.or(ignore_extra_columns.then_some(true)) {
            reader = reader.with_truncate_ragged_lines(truncate);
        }
        if let Some(policy) = extra_columns {
            reader = reader.with_extra_columns_policy(policy);
        }
        if let Some(column) = options.string("include_file_paths")? {
            reader = reader.with_include_file_paths(Some(column.into()));
        }

        let lf = reader.finish()?;
        Ok((path, lf))
    }

    #[cfg(feature = "parquet")]
    fn read_parquet(&self, args: &[SQLFunctionArg]) -> PolarsResult<(PlRefPath, LazyFrame)> {
        polars_ensure!(args.len() == 1, SQLSyntax: "`read_parquet` expects a single file path; found {:?} arguments", args.len());

        let path = self.get_file_path_from_arg(&args[0])?;
        let lf = LazyFrame::scan_parquet(path.clone(), Default::default())?;
        Ok((path, lf))
    }

    #[cfg(feature = "ipc")]
    fn read_ipc(&self, args: &[SQLFunctionArg]) -> PolarsResult<(PlRefPath, LazyFrame)> {
        polars_ensure!(args.len() == 1, SQLSyntax: "`read_ipc` expects a single file path; found {:?} arguments", args.len());

        let path = self.get_file_path_from_arg(&args[0])?;
        let lf = LazyFrame::scan_ipc(path.clone(), Default::default(), Default::default())?;
        Ok((path, lf))
    }
    #[cfg(feature = "json")]
    fn read_json(&self, args: &[SQLFunctionArg]) -> PolarsResult<(PlRefPath, LazyFrame)> {
        use polars_lazy::frame::LazyFileListReader;
        use polars_lazy::prelude::LazyJsonLineReader;

        polars_ensure!(
            args.len() == 1,
            SQLSyntax: "`read_json` expects a single file path; found {} arguments{}",
            args.len(),
            if args.len() > 1 { " (for options, use `read_ndjson`)" } else { "" }
        );
        let path = self.get_file_path_from_arg(&args[0])?;
        let lf = LazyJsonLineReader::new(path.clone()).finish()?;
        Ok((path, lf))
    }

    #[cfg(feature = "json")]
    fn read_ndjson(&self, args: &[SQLFunctionArg]) -> PolarsResult<(PlRefPath, LazyFrame)> {
        use std::sync::Arc;

        use polars_lazy::frame::LazyFileListReader;
        use polars_lazy::prelude::LazyJsonLineReader;

        let (path_arg, options) = TableFunctionOptions::parse(
            "read_ndjson",
            args,
            READ_NDJSON_OPTIONS,
            READ_NDJSON_EXCLUDED_OPTIONS,
        )?;
        let path = self.get_file_path_from_arg(path_arg)?;

        let mut reader = LazyJsonLineReader::new(path.clone())
            .with_schema(options.schema("schema")?.map(Arc::new))
            .with_schema_overwrite(options.schema("schema_overrides")?.map(Arc::new));
        if let Some(n) = options.nullable_positive_usize("infer_schema_length")? {
            reader = reader.with_infer_schema_length(n);
        }
        if let Some(ignore_errors) = options.bool("ignore_errors")? {
            reader = reader.with_ignore_errors(ignore_errors);
        }
        if let Some(column) = options.string("include_file_paths")? {
            reader = reader.with_include_file_paths(Some(column.into()));
        }

        let lf = reader.finish()?;
        Ok((path, lf))
    }

    #[allow(dead_code)]
    fn get_file_path_from_arg(&self, arg: &SQLFunctionArg) -> PolarsResult<PlRefPath> {
        match arg {
            SQLFunctionArg::Unnamed(SQLFunctionArgExpr::Expr(SQLExpr::Value(
                SQLValueWithSpan {
                    value: SQLValue::SingleQuotedString(s) | SQLValue::EscapedStringLiteral(s),
                    ..
                },
            ))) => Ok(PlRefPath::new(s)),
            _ => polars_bail!(
                SQLSyntax:
                "expected a valid file path as a single-quoted string; found: {}", arg,
            ),
        }
    }
}

/// Named options of `read_csv`: the options of Python's `scan_csv` that SQL can express.
#[cfg(feature = "csv")]
const READ_CSV_OPTIONS: &[&str] = &[
    "has_header",
    "separator",
    "quote_char",
    "comment_prefix",
    "skip_rows",
    "skip_rows_after_header",
    "skip_lines",
    "encoding",
    "new_columns",
    "schema",
    "schema_overrides",
    "infer_schema",
    "infer_schema_length",
    "infer_schema_files",
    "try_parse_dates",
    "decimal_comma",
    "null_values",
    "ignore_errors",
    "truncate_ragged_lines",
    "missing_columns",
    "extra_columns",
    "include_file_paths",
];

/// Options of Python's `scan_csv` and `read_csv` that `read_csv` does not accept, with what to do
/// instead.
#[cfg(feature = "csv")]
const READ_CSV_EXCLUDED_OPTIONS: &[(&str, &str)] = &[
    (
        "credential_provider",
        "cloud credentials are read from the environment",
    ),
    (
        "storage_options",
        "cloud credentials are read from the environment",
    ),
    ("with_column_names", "use new_columns"),
    ("n_rows", "use LIMIT, which is pushed down into the scan"),
    (
        "columns",
        "select the columns in the SELECT list; to set their types, use schema",
    ),
    ("low_memory", ""),
    ("eol_char", ""),
    ("raise_if_empty", ""),
    ("glob", ""),
    ("empty_string_is_null", ""),
    ("row_index_name", ""),
    ("row_index_offset", ""),
];

/// Named options of `read_ndjson`: the stable options of Python's `scan_ndjson` that SQL can express.
#[cfg(feature = "json")]
const READ_NDJSON_OPTIONS: &[&str] = &[
    "schema",
    "schema_overrides",
    "infer_schema_length",
    "ignore_errors",
    "include_file_paths",
];

/// Options of Python's `scan_ndjson` that `read_ndjson` does not accept, with what to do instead.
#[cfg(feature = "json")]
const READ_NDJSON_EXCLUDED_OPTIONS: &[(&str, &str)] = &[
    (
        "credential_provider",
        "cloud credentials are read from the environment",
    ),
    (
        "storage_options",
        "cloud credentials are read from the environment",
    ),
    ("n_rows", "use LIMIT, which is pushed down into the scan"),
    ("batch_size", ""),
    ("low_memory", ""),
    ("row_index_name", ""),
    ("row_index_offset", ""),
];

impl PolarsTableFunctions {
    // list sql names of all table functions
    pub(crate) fn keywords() -> &'static [&'static str] {
        &[
            #[cfg(feature = "csv")]
            "read_csv",
            #[cfg(feature = "parquet")]
            "read_parquet",
            #[cfg(feature = "ipc")]
            "read_ipc",
            #[cfg(feature = "json")]
            "read_json",
            #[cfg(feature = "json")]
            "read_ndjson",
            "generate_series",
            "range",
        ]
    }
}
