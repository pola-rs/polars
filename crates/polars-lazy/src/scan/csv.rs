use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::csv::read::{
    infer_file_schema, CommentPrefix, CsvEncoding, CsvParseOptions, CsvReadOptions, NullValues,
};
use polars_io::utils::get_reader_bytes;
use polars_io::RowIndex;

use crate::prelude::*;

#[derive(Clone)]
#[cfg(feature = "csv")]
pub struct LazyCsvReader {
    path: PathBuf,
    paths: Arc<[PathBuf]>,
    glob: bool,
    cache: bool,
    read_options: CsvReadOptions,
    cloud_options: Option<CloudOptions>,
}

#[cfg(feature = "csv")]
impl LazyCsvReader {
    /// Re-export to shorten code.
    fn map_parse_options<F: Fn(CsvParseOptions) -> CsvParseOptions>(mut self, map_func: F) -> Self {
        self.read_options = self.read_options.map_parse_options(map_func);
        self
    }

    pub fn new_paths(paths: Arc<[PathBuf]>) -> Self {
        Self::new("").with_paths(paths)
    }

    pub fn new(path: impl AsRef<Path>) -> Self {
        LazyCsvReader {
            path: path.as_ref().to_owned(),
            paths: Arc::new([]),
            glob: true,
            cache: true,
            read_options: Default::default(),
            cloud_options: Default::default(),
        }
    }

    /// Skip this number of rows after the header location.
    #[must_use]
    pub fn with_skip_rows_after_header(mut self, offset: usize) -> Self {
        self.read_options.skip_rows_after_header = offset;
        self
    }

    /// Add a row index column.
    #[must_use]
    pub fn with_row_index(mut self, row_index: Option<RowIndex>) -> Self {
        self.read_options.row_index = row_index;
        self
    }

    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    #[must_use]
    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.read_options.n_rows = num_rows;
        self
    }

    /// Set the number of rows to use when inferring the csv schema.
    /// the default is 100 rows.
    /// Setting to `None` will do a full table scan, very slow.
    #[must_use]
    pub fn with_infer_schema_length(mut self, num_rows: Option<usize>) -> Self {
        self.read_options.infer_schema_length = num_rows;
        self
    }

    /// Continue with next batch when a ParserError is encountered.
    #[must_use]
    pub fn with_ignore_errors(mut self, ignore: bool) -> Self {
        self.read_options.ignore_errors = ignore;
        self
    }

    /// Set the CSV file's schema
    #[must_use]
    pub fn with_schema(mut self, schema: Option<SchemaRef>) -> Self {
        self.read_options.schema = schema;
        self
    }

    /// Skip the first `n` rows during parsing. The header will be parsed at row `n`.
    #[must_use]
    pub fn with_skip_rows(mut self, skip_rows: usize) -> Self {
        self.read_options.skip_rows = skip_rows;
        self
    }

    /// Overwrite the schema with the dtypes in this given Schema. The given schema may be a subset
    /// of the total schema.
    #[must_use]
    pub fn with_dtype_overwrite(mut self, schema: Option<SchemaRef>) -> Self {
        self.read_options.schema_overwrite = schema;
        self
    }

    /// Set whether the CSV file has headers
    #[must_use]
    pub fn with_has_header(mut self, has_header: bool) -> Self {
        self.read_options.has_header = has_header;
        self
    }

    /// Set the CSV file's column separator as a byte character
    #[must_use]
    pub fn with_separator(self, separator: u8) -> Self {
        self.map_parse_options(|opts| opts.with_separator(separator))
    }

    /// Set the comment prefix for this instance. Lines starting with this prefix will be ignored.
    #[must_use]
    pub fn with_comment_prefix(self, comment_prefix: Option<&str>) -> Self {
        self.map_parse_options(|opts| {
            opts.with_comment_prefix(comment_prefix.map(|s| {
                if s.len() == 1 && s.chars().next().unwrap().is_ascii() {
                    CommentPrefix::Single(s.as_bytes()[0])
                } else {
                    CommentPrefix::Multi(Arc::from(s))
                }
            }))
        })
    }

    /// Set the `char` used as quote char. The default is `b'"'`. If set to `[None]` quoting is disabled.
    #[must_use]
    pub fn with_quote_char(self, quote_char: Option<u8>) -> Self {
        self.map_parse_options(|opts| opts.with_quote_char(quote_char))
    }

    /// Set the `char` used as end of line. The default is `b'\n'`.
    #[must_use]
    pub fn with_eol_char(self, eol_char: u8) -> Self {
        self.map_parse_options(|opts| opts.with_eol_char(eol_char))
    }

    /// Set values that will be interpreted as missing/ null.
    #[must_use]
    pub fn with_null_values(self, null_values: Option<NullValues>) -> Self {
        self.map_parse_options(|opts| opts.with_null_values(null_values.clone()))
    }

    /// Treat missing fields as null.
    pub fn with_missing_is_null(self, missing_is_null: bool) -> Self {
        self.map_parse_options(|opts| opts.with_missing_is_null(missing_is_null))
    }

    /// Cache the DataFrame after reading.
    #[must_use]
    pub fn with_cache(mut self, cache: bool) -> Self {
        self.cache = cache;
        self
    }

    /// Reduce memory usage at the expense of performance
    #[must_use]
    pub fn with_low_memory(mut self, low_memory: bool) -> Self {
        self.read_options.low_memory = low_memory;
        self
    }

    /// Set  [`CsvEncoding`]
    #[must_use]
    pub fn with_encoding(self, encoding: CsvEncoding) -> Self {
        self.map_parse_options(|opts| opts.with_encoding(encoding))
    }

    /// Automatically try to parse dates/datetimes and time.
    /// If parsing fails, columns remain of dtype `[DataType::String]`.
    #[cfg(feature = "temporal")]
    pub fn with_try_parse_dates(self, try_parse_dates: bool) -> Self {
        self.map_parse_options(|opts| opts.with_try_parse_dates(try_parse_dates))
    }

    /// Raise an error if CSV is empty (otherwise return an empty frame)
    #[must_use]
    pub fn with_raise_if_empty(mut self, raise_if_empty: bool) -> Self {
        self.read_options.raise_if_empty = raise_if_empty;
        self
    }

    /// Truncate lines that are longer than the schema.
    #[must_use]
    pub fn with_truncate_ragged_lines(self, truncate_ragged_lines: bool) -> Self {
        self.map_parse_options(|opts| opts.with_truncate_ragged_lines(truncate_ragged_lines))
    }

    #[must_use]
    pub fn with_decimal_comma(self, decimal_comma: bool) -> Self {
        self.map_parse_options(|opts| opts.with_decimal_comma(decimal_comma))
    }

    #[must_use]
    /// Expand path given via globbing rules.
    pub fn with_glob(mut self, toggle: bool) -> Self {
        self.glob = toggle;
        self
    }

    pub fn with_cloud_options(mut self, cloud_options: Option<CloudOptions>) -> Self {
        self.cloud_options = cloud_options;
        self
    }

    /// Modify a schema before we run the lazy scanning.
    ///
    /// Important! Run this function latest in the builder!
    pub fn with_schema_modify<F>(mut self, f: F) -> PolarsResult<Self>
    where
        F: Fn(Schema) -> PolarsResult<Schema>,
    {
        let mut file = if let Some(mut paths) = self.iter_paths()? {
            let path = match paths.next() {
                Some(globresult) => globresult?,
                None => polars_bail!(ComputeError: "globbing pattern did not match any files"),
            };
            polars_utils::open_file(path)
        } else {
            polars_utils::open_file(&self.path)
        }?;
        let reader_bytes = get_reader_bytes(&mut file).expect("could not mmap file");
        let skip_rows = self.read_options.skip_rows;
        let parse_options = self.read_options.get_parse_options();

        let (schema, _, _) = infer_file_schema(
            &reader_bytes,
            parse_options.separator,
            self.read_options.infer_schema_length,
            self.read_options.has_header,
            // we set it to None and modify them after the schema is updated
            None,
            skip_rows,
            self.read_options.skip_rows_after_header,
            parse_options.comment_prefix.as_ref(),
            parse_options.quote_char,
            parse_options.eol_char,
            None,
            parse_options.try_parse_dates,
            self.read_options.raise_if_empty,
            &mut self.read_options.n_threads,
            parse_options.decimal_comma,
        )?;
        let mut schema = f(schema)?;

        // the dtypes set may be for the new names, so update again
        if let Some(overwrite_schema) = &self.read_options.schema_overwrite {
            for (name, dtype) in overwrite_schema.iter() {
                schema.with_column(name.clone(), dtype.clone());
            }
        }

        Ok(self.with_schema(Some(Arc::new(schema))))
    }
}

impl LazyFileListReader for LazyCsvReader {
    /// Get the final [LazyFrame].
    fn finish(mut self) -> PolarsResult<LazyFrame> {
        if !self.glob {
            return self.finish_no_glob();
        }
        if let Some(paths) = self.iter_paths()? {
            let paths = paths
                .into_iter()
                .collect::<PolarsResult<Arc<[PathBuf]>>>()?;
            self.paths = paths;
        }
        self.finish_no_glob()
    }

    fn finish_no_glob(self) -> PolarsResult<LazyFrame> {
        let paths = if self.paths.is_empty() {
            Arc::new([self.path])
        } else {
            self.paths
        };

        let mut lf: LazyFrame =
            DslBuilder::scan_csv(paths, self.read_options, self.cache, self.cloud_options)?
                .build()
                .into();
        lf.opt_state.file_caching = true;
        Ok(lf)
    }

    fn glob(&self) -> bool {
        self.glob
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    fn with_path(mut self, path: PathBuf) -> Self {
        self.path = path;
        self
    }

    fn with_paths(mut self, paths: Arc<[PathBuf]>) -> Self {
        self.paths = paths;
        self
    }

    fn with_n_rows(mut self, n_rows: impl Into<Option<usize>>) -> Self {
        self.read_options.n_rows = n_rows.into();
        self
    }

    fn with_row_index(mut self, row_index: impl Into<Option<RowIndex>>) -> Self {
        self.read_options.row_index = row_index.into();
        self
    }

    fn rechunk(&self) -> bool {
        self.read_options.rechunk
    }

    /// Rechunk the memory to contiguous chunks when parsing is done.
    #[must_use]
    fn with_rechunk(mut self, rechunk: bool) -> Self {
        self.read_options.rechunk = rechunk;
        self
    }

    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    fn n_rows(&self) -> Option<usize> {
        self.read_options.n_rows
    }

    /// Return the row index settings.
    fn row_index(&self) -> Option<&RowIndex> {
        self.read_options.row_index.as_ref()
    }

    fn concat_impl(&self, lfs: Vec<LazyFrame>) -> PolarsResult<LazyFrame> {
        // set to false, as the csv parser has full thread utilization
        let args = UnionArgs {
            rechunk: self.rechunk(),
            parallel: false,
            to_supertypes: false,
            from_partitioned_ds: true,
            ..Default::default()
        };
        concat_impl(&lfs, args)
    }
}
