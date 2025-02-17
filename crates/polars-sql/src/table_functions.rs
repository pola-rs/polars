use std::str::FromStr;

#[cfg(any(
    feature = "csv",
    feature = "parquet",
    feature = "ipc",
    feature = "json"
))]
use polars_core::prelude::polars_ensure;
use polars_core::prelude::{polars_bail, PolarsError, PolarsResult};
#[cfg(feature = "csv")]
use polars_lazy::prelude::LazyCsvReader;
use polars_lazy::prelude::LazyFrame;
use sqlparser::ast::{FunctionArg, FunctionArgExpr};

/// Table functions that are supported by Polars
#[allow(clippy::enum_variant_names)]
pub(crate) enum PolarsTableFunctions {
    /// SQL 'read_csv' function
    /// ```sql
    /// SELECT * FROM read_csv('path/to/file.csv')
    /// ```
    #[cfg(feature = "csv")]
    ReadCsv,
    /// SQL 'read_parquet' function
    /// ```sql
    /// SELECT * FROM read_parquet('path/to/file.parquet')
    /// ```
    #[cfg(feature = "parquet")]
    ReadParquet,
    /// SQL 'read_ipc' function
    /// ```sql
    /// SELECT * FROM read_ipc('path/to/file.ipc')
    /// ```
    #[cfg(feature = "ipc")]
    ReadIpc,
    /// SQL 'read_json' function. *Only ndjson is currently supported.*
    /// ```sql
    /// SELECT * FROM read_json('path/to/file.json')
    /// ```
    #[cfg(feature = "json")]
    ReadJson,
}

impl FromStr for PolarsTableFunctions {
    type Err = PolarsError;

    #[allow(unreachable_code)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            #[cfg(feature = "csv")]
            "read_csv" => PolarsTableFunctions::ReadCsv,
            #[cfg(feature = "parquet")]
            "read_parquet" => PolarsTableFunctions::ReadParquet,
            #[cfg(feature = "ipc")]
            "read_ipc" => PolarsTableFunctions::ReadIpc,
            #[cfg(feature = "json")]
            "read_json" => PolarsTableFunctions::ReadJson,
            _ => polars_bail!(SQLInterface: "'{}' is not a supported table function", s),
        })
    }
}

impl PolarsTableFunctions {
    #[allow(unused_variables, unreachable_patterns)]
    pub(crate) fn execute(&self, args: &[FunctionArg]) -> PolarsResult<(String, LazyFrame)> {
        match self {
            #[cfg(feature = "csv")]
            PolarsTableFunctions::ReadCsv => self.read_csv(args),
            #[cfg(feature = "parquet")]
            PolarsTableFunctions::ReadParquet => self.read_parquet(args),
            #[cfg(feature = "ipc")]
            PolarsTableFunctions::ReadIpc => self.read_ipc(args),
            #[cfg(feature = "json")]
            PolarsTableFunctions::ReadJson => self.read_ndjson(args),
            _ => unreachable!(),
        }
    }

    #[cfg(feature = "csv")]
    fn read_csv(&self, args: &[FunctionArg]) -> PolarsResult<(String, LazyFrame)> {
        polars_ensure!(args.len() == 1, SQLSyntax: "`read_csv` expects a single file path; found {:?} arguments", args.len());

        use polars_lazy::frame::LazyFileListReader;
        let path = self.get_file_path_from_arg(&args[0])?;
        let lf = LazyCsvReader::new(&path)
            .with_try_parse_dates(true)
            .with_missing_is_null(true)
            .finish()?;
        Ok((path, lf))
    }

    #[cfg(feature = "parquet")]
    fn read_parquet(&self, args: &[FunctionArg]) -> PolarsResult<(String, LazyFrame)> {
        polars_ensure!(args.len() == 1, SQLSyntax: "`read_parquet` expects a single file path; found {:?} arguments", args.len());

        let path = self.get_file_path_from_arg(&args[0])?;
        let lf = LazyFrame::scan_parquet(&path, Default::default())?;
        Ok((path, lf))
    }

    #[cfg(feature = "ipc")]
    fn read_ipc(&self, args: &[FunctionArg]) -> PolarsResult<(String, LazyFrame)> {
        polars_ensure!(args.len() == 1, SQLSyntax: "`read_ipc` expects a single file path; found {:?} arguments", args.len());

        let path = self.get_file_path_from_arg(&args[0])?;
        let lf = LazyFrame::scan_ipc(&path, Default::default())?;
        Ok((path, lf))
    }
    #[cfg(feature = "json")]
    fn read_ndjson(&self, args: &[FunctionArg]) -> PolarsResult<(String, LazyFrame)> {
        polars_ensure!(args.len() == 1, SQLSyntax: "`read_ndjson` expects a single file path; found {:?} arguments", args.len());

        use polars_lazy::frame::LazyFileListReader;
        use polars_lazy::prelude::LazyJsonLineReader;

        let path = self.get_file_path_from_arg(&args[0])?;
        let lf = LazyJsonLineReader::new(path.clone()).finish()?;
        Ok((path, lf))
    }

    #[allow(dead_code)]
    fn get_file_path_from_arg(&self, arg: &FunctionArg) -> PolarsResult<String> {
        use sqlparser::ast::{Expr as SQLExpr, Value as SQLValue};
        match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(SQLExpr::Value(
                SQLValue::SingleQuotedString(s),
            ))) => Ok(s.to_string()),
            _ => polars_bail!(
                SQLSyntax:
                "expected a valid file path as a single-quoted string; found: {}", arg,
            ),
        }
    }
}

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
        ]
    }
}
