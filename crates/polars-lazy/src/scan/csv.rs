use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::csv::utils::infer_file_schema;
use polars_io::csv::{CommentPrefix, CsvEncoding, NullValues};
use polars_io::utils::get_reader_bytes;
use polars_io::RowCount;

use crate::frame::LazyFileListReader;
use crate::prelude::*;

/// To load files once this is fully configured, use
/// [LazyFileListReader::load_multiple],
/// [LazyFileListReader::load_specific_file] or other related methods.
#[derive(Clone)]
#[cfg(feature = "csv")]
pub struct LazyCsvReader<'a> {
    separator: u8,
    has_header: bool,
    ignore_errors: bool,
    skip_rows: usize,
    n_rows: Option<usize>,
    cache: bool,
    schema: Option<SchemaRef>,
    schema_overwrite: Option<&'a Schema>,
    low_memory: bool,
    comment_prefix: Option<CommentPrefix>,
    quote_char: Option<u8>,
    eol_char: u8,
    null_values: Option<NullValues>,
    missing_is_null: bool,
    truncate_ragged_lines: bool,
    infer_schema_length: Option<usize>,
    rechunk: bool,
    skip_rows_after_header: usize,
    encoding: CsvEncoding,
    row_count: Option<RowCount>,
    try_parse_dates: bool,
    raise_if_empty: bool,
}

#[cfg(feature = "csv")]
impl<'a> LazyCsvReader<'a> {
    pub fn new() -> Self {
        LazyCsvReader {
            separator: b',',
            has_header: true,
            ignore_errors: false,
            skip_rows: 0,
            n_rows: None,
            cache: true,
            schema: None,
            schema_overwrite: None,
            low_memory: false,
            comment_prefix: None,
            quote_char: Some(b'"'),
            eol_char: b'\n',
            null_values: None,
            missing_is_null: true,
            infer_schema_length: Some(100),
            rechunk: true,
            skip_rows_after_header: 0,
            encoding: CsvEncoding::Utf8,
            row_count: None,
            try_parse_dates: false,
            raise_if_empty: true,
            truncate_ragged_lines: false,
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
    pub fn with_schema(mut self, schema: Option<SchemaRef>) -> Self {
        self.schema = schema;
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

    /// Set the CSV file's column separator as a byte character
    #[must_use]
    pub fn with_separator(mut self, separator: u8) -> Self {
        self.separator = separator;
        self
    }

    /// Set the comment prefix for this instance. Lines starting with this prefix will be ignored.
    #[must_use]
    pub fn with_comment_prefix(mut self, comment_prefix: Option<&str>) -> Self {
        self.comment_prefix = comment_prefix.map(|s| {
            if s.len() == 1 && s.chars().next().unwrap().is_ascii() {
                CommentPrefix::Single(s.as_bytes()[0])
            } else {
                CommentPrefix::Multi(s.to_string())
            }
        });
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

    /// Set  [`CsvEncoding`]
    #[must_use]
    pub fn with_encoding(mut self, enc: CsvEncoding) -> Self {
        self.encoding = enc;
        self
    }

    /// Automatically try to parse dates/datetimes and time.
    /// If parsing fails, columns remain of dtype `[DataType::Utf8]`.
    #[cfg(feature = "temporal")]
    pub fn with_try_parse_dates(mut self, toggle: bool) -> Self {
        self.try_parse_dates = toggle;
        self
    }

    /// Raise an error if CSV is empty (otherwise return an empty frame)
    #[must_use]
    pub fn raise_if_empty(mut self, toggle: bool) -> Self {
        self.raise_if_empty = toggle;
        self
    }

    /// Truncate lines that are longer than the schema.
    #[must_use]
    pub fn truncate_ragged_lines(mut self, toggle: bool) -> Self {
        self.truncate_ragged_lines = toggle;
        self
    }

    /// Modify a schema before we run the lazy scanning.
    ///
    /// Important! Run this function latest in the builder!
    pub fn with_schema_modify<F>(mut self, f: F) -> PolarsResult<Self>
    where
        F: Fn(Schema) -> PolarsResult<Schema>,
    {
        let mut mmap_bytes_reader = self.specific_reader().expect("There should be a specific ReaderFactory set at this point").mmapbytesreader()?;
        let reader_bytes = get_reader_bytes(&mut mmap_bytes_reader).expect("could not mmap file");
        let mut skip_rows = self.skip_rows;

        let (schema, _, _) = infer_file_schema(
            &reader_bytes,
            self.separator,
            self.infer_schema_length,
            self.has_header,
            // we set it to None and modify them after the schema is updated
            None,
            &mut skip_rows,
            self.skip_rows_after_header,
            self.comment_prefix.as_ref(),
            self.quote_char,
            self.eol_char,
            None,
            self.try_parse_dates,
            self.raise_if_empty,
        )?;
        let mut schema = f(schema)?;

        // the dtypes set may be for the new names, so update again
        if let Some(overwrite_schema) = self.schema_overwrite {
            for (name, dtype) in overwrite_schema.iter() {
                schema.with_column(name.clone(), dtype.clone());
            }
        }

        Ok(self.with_schema(Some(Arc::new(schema))))
    }
}

impl LazyFileListReader for LazyCsvReader<'_> {
    fn load_specific(self, location: ScanLocation) -> PolarsResult<LazyFrame> {
        let mut lf: LazyFrame = LogicalPlanBuilder::scan_csv(
            location.mmapbytesreader(),
            self.separator,
            self.has_header,
            self.ignore_errors,
            self.skip_rows,
            self.n_rows,
            self.cache,
            self.schema,
            self.schema_overwrite,
            self.low_memory,
            self.comment_prefix,
            self.quote_char,
            self.eol_char,
            self.null_values,
            self.infer_schema_length,
            self.rechunk,
            self.skip_rows_after_header,
            self.encoding,
            self.row_count,
            self.try_parse_dates,
            self.raise_if_empty,
            self.truncate_ragged_lines,
        )?
        .build()
        .into();
        lf.opt_state.file_caching = true;
        Ok(lf)
    }

    fn rechunk(&self) -> bool {
        self.rechunk
    }

    /// Rechunk the memory to contiguous chunks when parsing is done.
    #[must_use]
    fn with_rechunk(mut self, toggle: bool) -> Self {
        self.rechunk = toggle;
        self
    }

    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    fn n_rows(&self) -> Option<usize> {
        self.n_rows
    }

    /// Add a `row_count` column.
    fn row_count(&self) -> Option<&RowCount> {
        self.row_count.as_ref()
    }

    fn concat_impl(&self, lfs: Vec<LazyFrame>) -> PolarsResult<LazyFrame> {
        // set to false, as the csv parser has full thread utilization
        concat_impl(&lfs, self.rechunk(), false, true, false)
    }
}
