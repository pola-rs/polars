use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::csv::utils::{get_reader_bytes, infer_file_schema};
use polars_io::csv::{CsvEncoding, NullValues};
use polars_io::RowCount;

use crate::prelude::*;

#[derive(Clone)]
#[cfg(feature = "csv-file")]
pub struct LazyCsvReader<'a> {
    path: PathBuf,
    delimiter: u8,
    has_header: bool,
    ignore_errors: bool,
    skip_rows: usize,
    n_rows: Option<usize>,
    cache: bool,
    schema: Option<SchemaRef>,
    schema_overwrite: Option<&'a Schema>,
    low_memory: bool,
    comment_char: Option<u8>,
    quote_char: Option<u8>,
    eol_char: u8,
    null_values: Option<NullValues>,
    missing_is_null: bool,
    infer_schema_length: Option<usize>,
    rechunk: bool,
    skip_rows_after_header: usize,
    encoding: CsvEncoding,
    row_count: Option<RowCount>,
    parse_dates: bool,
}

#[cfg(feature = "csv-file")]
impl<'a> LazyCsvReader<'a> {
    pub fn new(path: impl AsRef<Path>) -> Self {
        LazyCsvReader {
            path: path.as_ref().to_owned(),
            delimiter: b',',
            has_header: true,
            ignore_errors: false,
            skip_rows: 0,
            n_rows: None,
            cache: true,
            schema: None,
            schema_overwrite: None,
            low_memory: false,
            comment_char: None,
            quote_char: Some(b'"'),
            eol_char: b'\n',
            null_values: None,
            missing_is_null: true,
            infer_schema_length: Some(100),
            rechunk: true,
            skip_rows_after_header: 0,
            encoding: CsvEncoding::Utf8,
            row_count: None,
            parse_dates: false,
        }
    }

    /// Skip this number of rows after the header location.
    #[must_use]
    pub fn with_skip_rows_after_header(mut self, offset: usize) -> Self {
        self.skip_rows_after_header = offset;
        self
    }

    /// Add a `row_count` column.
    #[must_use]
    pub fn with_row_count(mut self, row_count: Option<RowCount>) -> Self {
        self.row_count = row_count;
        self
    }

    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    #[must_use]
    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.n_rows = num_rows;
        self
    }

    /// Set the number of rows to use when inferring the csv schema.
    /// the default is 100 rows.
    /// Setting to `None` will do a full table scan, very slow.
    #[must_use]
    pub fn with_infer_schema_length(mut self, num_rows: Option<usize>) -> Self {
        self.infer_schema_length = num_rows;
        self
    }

    /// Continue with next batch when a ParserError is encountered.
    #[must_use]
    pub fn with_ignore_errors(mut self, ignore: bool) -> Self {
        self.ignore_errors = ignore;
        self
    }

    /// Set the CSV file's schema
    #[must_use]
    pub fn with_schema(mut self, schema: SchemaRef) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Skip the first `n` rows during parsing. The header will be parsed at row `n`.
    #[must_use]
    pub fn with_skip_rows(mut self, skip_rows: usize) -> Self {
        self.skip_rows = skip_rows;
        self
    }

    /// Overwrite the schema with the dtypes in this given Schema. The given schema may be a subset
    /// of the total schema.
    #[must_use]
    pub fn with_dtype_overwrite(mut self, schema: Option<&'a Schema>) -> Self {
        self.schema_overwrite = schema;
        self
    }

    /// Set whether the CSV file has headers
    #[must_use]
    pub fn has_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Set the CSV file's column delimiter as a byte character
    #[must_use]
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set the comment character. Lines starting with this character will be ignored.
    #[must_use]
    pub fn with_comment_char(mut self, comment_char: Option<u8>) -> Self {
        self.comment_char = comment_char;
        self
    }

    /// Set the `char` used as quote char. The default is `b'"'`. If set to `[None]` quoting is disabled.
    #[must_use]
    pub fn with_quote_char(mut self, quote: Option<u8>) -> Self {
        self.quote_char = quote;
        self
    }

    /// Set the `char` used as end of line. The default is `b'\n'`.
    #[must_use]
    pub fn with_end_of_line_char(mut self, eol_char: u8) -> Self {
        self.eol_char = eol_char;
        self
    }

    /// Set values that will be interpreted as missing/ null.
    #[must_use]
    pub fn with_null_values(mut self, null_values: Option<NullValues>) -> Self {
        self.null_values = null_values;
        self
    }

    /// Treat missing fields as null.
    pub fn with_missing_is_null(mut self, missing_is_null: bool) -> Self {
        self.missing_is_null = missing_is_null;
        self
    }

    /// Cache the DataFrame after reading.
    #[must_use]
    pub fn with_cache(mut self, cache: bool) -> Self {
        self.cache = cache;
        self
    }

    /// Reduce memory usage in expensive of performance
    #[must_use]
    pub fn low_memory(mut self, toggle: bool) -> Self {
        self.low_memory = toggle;
        self
    }

    /// Rechunk the memory to contiguous chunks when parsing is done.
    #[must_use]
    pub fn with_rechunk(mut self, toggle: bool) -> Self {
        self.rechunk = toggle;
        self
    }

    /// Set  [`CsvEncoding`]
    #[must_use]
    pub fn with_encoding(mut self, enc: CsvEncoding) -> Self {
        self.encoding = enc;
        self
    }

    /// Automatically try to parse dates/ datetimes and time. If parsing fails, columns remain of dtype `[DataType::Utf8]`.
    #[cfg(feature = "temporal")]
    pub fn with_parse_dates(mut self, toggle: bool) -> Self {
        self.parse_dates = toggle;
        self
    }

    /// Modify a schema before we run the lazy scanning.
    ///
    /// Important! Run this function latest in the builder!
    pub fn with_schema_modify<F>(self, f: F) -> PolarsResult<Self>
    where
        F: Fn(Schema) -> PolarsResult<Schema>,
    {
        let path;
        let path_str = self.path.to_string_lossy();

        let mut file = if path_str.contains('*') {
            let glob_err = || PolarsError::ComputeError("invalid glob pattern given".into());
            let mut paths = glob::glob(&path_str).map_err(|_| glob_err())?;

            match paths.next() {
                Some(globresult) => {
                    path = globresult.map_err(|_| glob_err())?;
                }
                None => {
                    return Err(PolarsError::ComputeError(
                        "globbing pattern did not match any files".into(),
                    ));
                }
            }
            std::fs::File::open(&path)
        } else {
            std::fs::File::open(&self.path)
        }?;
        let reader_bytes = get_reader_bytes(&mut file).expect("could not mmap file");
        let mut skip_rows = self.skip_rows;

        let (schema, _, _) = infer_file_schema(
            &reader_bytes,
            self.delimiter,
            self.infer_schema_length,
            self.has_header,
            // we set it to None and modify them after the schema is updated
            None,
            &mut skip_rows,
            self.skip_rows_after_header,
            self.comment_char,
            self.quote_char,
            self.eol_char,
            None,
            self.parse_dates,
        )?;
        let mut schema = f(schema)?;

        // the dtypes set may be for the new names, so update again
        if let Some(overwrite_schema) = self.schema_overwrite {
            for (name, dtype) in overwrite_schema.iter() {
                schema.with_column(name.clone(), dtype.clone());
            }
        }

        Ok(self.with_schema(Arc::new(schema)))
    }

    pub fn finish_impl(self) -> PolarsResult<LazyFrame> {
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_csv(
            self.path,
            self.delimiter,
            self.has_header,
            self.ignore_errors,
            self.skip_rows,
            self.n_rows,
            self.cache,
            self.schema,
            self.schema_overwrite,
            self.low_memory,
            self.comment_char,
            self.quote_char,
            self.eol_char,
            self.null_values,
            self.infer_schema_length,
            self.rechunk,
            self.skip_rows_after_header,
            self.encoding,
            self.row_count,
            self.parse_dates,
        )?
        .build()
        .into();
        lf.opt_state.file_caching = true;
        Ok(lf)
    }

    pub fn finish(self) -> PolarsResult<LazyFrame> {
        let path_str = self.path.to_string_lossy();
        if path_str.contains('*') {
            let paths = glob::glob(&path_str)
                .map_err(|_| PolarsError::ComputeError("invalid glob pattern given".into()))?;

            let lfs = paths
                .map(|r| {
                    let path = r.map_err(|e| PolarsError::ComputeError(format!("{e}").into()))?;
                    let mut builder = self.clone();
                    builder.path = path;
                    // do no rechunk yet.
                    builder.rechunk = false;
                    builder.finish_impl()
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            // set to false, as the csv parser has full thread utilization
            concat_impl(&lfs, self.rechunk, false, true)
                .map_err(|_| PolarsError::ComputeError("no matching files found".into()))
        } else {
            self.finish_impl()
        }
    }
}
