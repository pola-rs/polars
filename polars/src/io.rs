use derive_builder::Builder;
use polars_core::schema::SchemaRef;
use polars_io::mmap::MmapBytesReader;
pub use polars_io::{RowCount, *};

/// Read a CSV file into a DataFrame.
#[cfg(feature = "csv")]
pub mod read_csv {
    use polars_io::prelude::{CsvEncoding, CsvReader, NullValues};

    use super::*;
    #[macro_export]
    /// Read a CSV file into a DataFrame.
    /// This macro is a convenience macro to read a CSV file into a DataFrame.
    /// See [`CsvReaderOptions`] for all available parameters.
    ///
    /// examples:
    /// ```rust no_run
    /// # use polars::prelude::*;
    ///
    /// # fn main() -> PolarsResult<DataFrame> {
    /// let df = polars::read_csv!("foo.csv")?;
    /// let df = polars::read_csv!("foo.csv", delimiter = b';')?;
    /// let df = polars::read_csv!("foo.csv", delimiter = b';', has_header = true)?;
    /// // Or reading from a PathBuf
    /// let path = std::path::PathBuf::from("foo.csv");
    /// let df = polars::read_csv!(&path)?;
    /// # }
    /// ```
    ///
    /// You can also consume a reader:
    /// ```rust no_run
    /// # use polars::prelude::*;
    /// # fn main() -> PolarsResult<DataFrame> {
    /// let buf = br#"foo,bar\n1,2"#;
    /// let cursor = std::io::Cursor::new(buf);
    /// let df = polars::read_csv!(cursor)?;
    /// // It also works with &mut readers
    /// let df = polars::read_csv!(&mut cursor)?;
    /// # }
    macro_rules! read_csv {
        // read_csv!("foo.csv")
        ($path:literal) => {
            $crate::prelude::CsvReader::from_path($path).and_then(|mut rdr| {
                rdr.finish()
            })
        };
        // read_csv!("foo.csv", ...options)
        ($path:literal, $($field:ident = $value:expr),* $(,)?) => {
            $crate::prelude::CsvReader::from_path($path).and_then(|rdr| {
                let mut options = $crate::io::read_csv::CsvReaderOptions::builder();
                $(
                    options = options.$field($value);
                )*
                let rdr = $crate::io::read_csv::set_options(rdr, options);
                rdr.finish()
            })
        };

        // read_csv!(&mut reader)
        (&mut $reader:expr) => {
            $crate::prelude::CsvReader::new($reader).finish()
        };

        // read_csv!(&mut reader, ...options)
        (&mut $reader:expr, $($field:ident = $value:expr),* $(,)?) => {{
            let rdr = $crate::prelude::CsvReader::new($reader);
            let mut options = $crate::io::read_csv::CsvReaderOptions::new();
            $(
                options = options.$field($value);
            )*
            let rdr = $crate::io::read_csv::set_options(rdr, options);
            rdr.finish()
        }};

        // read_csv!(&path)
        (&$path:expr) => {
            $crate::prelude::CsvReader::from_path($path).and_then(|mut rdr| {
                rdr.finish()
            })
        };

        // read_csv!(&path, ...options)
        (&$path:expr, $($field:ident = $value:expr),* $(,)?) => {
            $crate::prelude::CsvReader::from_path($path).and_then(|rdr| {
                let mut options = $crate::io::read_csv::CsvReaderOptionsBuilder::default();
                $(
                    options = options.$field($value);
                )*
                let options = options.build().unwrap();

                let rdr = $crate::io::read_csv::set_options(rdr, options);
                rdr.finish()
            })
        };

        // read_csv!(reader)
        ($read:expr) => {
            $crate::prelude::CsvReader::new($read).finish()
        };

        // read_csv!(reader, ...options)
        ($reader:expr, $($field:ident = $value:expr),* $(,)?) => {{
            let rdr = $crate::prelude::CsvReader::new($reader);
            let mut options = $crate::io::read_csv::CsvReaderOptionsBuilder::default();
            $(
                options = options.$field($value);
            )*
            let options = options.build().unwrap();
            let rdr = $crate::io::read_csv::set_options(rdr, options);
            rdr.finish()
        }};

  }

    #[must_use]
    #[derive(Builder)]
    #[builder(pattern = "immutable")]
    pub struct CsvReaderOptions {
        /// Sets the chunk size used by the parser. This influences performance
        chunk_size: usize,

        /// Columns to select/ project
        #[builder(setter(strip_option))]
        columns: Option<Vec<String>>,
        /// Set the comment character. Lines starting with this character will be ignored.
        #[builder(setter(strip_option))]
        comment_char: Option<u8>,
        /// Set the CSV file's column delimiter as a byte character
        #[builder(setter(strip_option))]
        delimiter: Option<u8>,
        /// Overwrite the schema with the dtypes in this given Schema. The given schema may be a subset
        /// of the total schema.
        #[builder(setter(strip_option))]
        dtypes: Option<SchemaRef>,
        /// Set  [`CsvEncoding`]
        encoding: CsvEncoding,
        /// Single byte end of line character. Defaults to `b'\n'`.
        eol_char: u8,
        /// Set whether the CSV file has headers
        has_header: bool,
        /// Continue with next batch when a ParserError is encountered.
        ignore_errors: bool,
        /// Set the Maximum number of rows read for schema inference
        #[builder(setter(strip_option))]
        infer_schema_length: Option<usize>,
        /// Reduce memory consumption at the expense of performance
        low_memory: bool,
        /// Treat missing fields as null.
        missing_is_null: bool,
        /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
        /// be guaranteed.
        #[builder(setter(strip_option))]
        n_rows: Option<usize>,
        /// Set the number of threads used in CSV reading. The default uses the number of cores of
        /// your cpu.
        ///
        /// Note that this only works if this is initialized with `CsvReader::from_path`.
        /// Note that the number of cores is the maximum allowed number of threads.
        #[builder(setter(strip_option))]
        n_threads: Option<usize>,
        /// Set values that will be interpreted as missing/ null. Note that any value you set as null value
        /// will not be escaped, so if quotation marks are part of the null value you should include them.
        /// example:
        /// ```rust no_run
        /// // NULL will be interpreted as a null value
        /// read_csv!("data.csv", null_values = "NULL");
        /// // NULL or null will be interpreted as a null value
        /// read_csv!("data.csv", null_values = ("NULL", "null"));
        /// // "NULL" or null will be interpreted as a null value
        /// read_csv!("data.csv", null_values = &["\"NULL\"", "null"]);
        /// ```
        ///
        #[builder(setter(strip_option, into))]
        null_values: Option<NullValues>,
        /// Set the reader's column projection. This counts from 0, meaning that
        /// `vec![0, 4]` would select the 1st and 5th column.
        #[builder(setter(strip_option))]
        projection: Option<Vec<usize>>,
        /// Set the `char` used as quote char. The default is `b'"'`. If set to `[None]` quoting is disabled.
        quote_char: Option<u8>,
        /// Rechunk the DataFrame to contiguous memory after the CSV is parsed.
        rechunk: bool,
        /// Add a `row_count` column.
        #[builder(setter(strip_option, into))]
        row_count: Option<RowCount>,
        /// Sets the size of the sample taken from the CSV file. The sample is used to get statistic about
        /// the file. These statistics are used to try to optimally allocate up front. Increasing this may
        /// improve performance.
        sample_size: usize,
        /// Set the CSV file's schema. This only accepts datatypes that are implemented
        /// in the csv parser and expects a complete Schema.
        ///
        /// It is recommended to use [with_dtypes](Self::with_dtypes) instead.
        // #[setters(strip_option)]
        schema: Option<SchemaRef>,
        /// Skip these rows after the header
        skip_rows_after_header: usize,

        /// Skip the first `n` rows during parsing. The header will be parsed at `n` lines.
        skip_rows_before_header: usize,
        /// Automatically try to parse dates/ datetimes and time. If parsing fails, columns remain of dtype `[DataType::Utf8]`.
        #[cfg(feature = "temporal")]
        try_parse_dates: bool,
    }

    pub fn set_options<'a, R: MmapBytesReader + 'a>(
        rdr: CsvReader<'a, R>,
        options: CsvReaderOptions,
    ) -> CsvReader<'a, R> {
        let mut rdr = rdr
            .with_skip_rows_after_header(options.skip_rows_after_header)
            .low_memory(options.low_memory)
            .infer_schema(options.infer_schema_length)
            .sample_size(options.sample_size)
            .with_dtypes(options.dtypes)
            .with_projection(options.projection)
            .with_columns(options.columns)
            .with_row_count(options.row_count)
            .with_chunk_size(options.chunk_size)
            .with_encoding(options.encoding)
            .with_n_rows(options.n_rows)
            .with_ignore_errors(options.ignore_errors)
            .with_skip_rows(options.skip_rows_before_header)
            .with_rechunk(options.rechunk)
            .has_header(options.has_header)
            .with_comment_char(options.comment_char)
            .with_end_of_line_char(options.eol_char)
            .with_null_values(options.null_values)
            .with_missing_is_null(options.missing_is_null)
            .with_quote_char(options.quote_char)
            .with_n_threads(options.n_threads);

        #[cfg(feature = "temporal")]
        {
            rdr = rdr.with_try_parse_dates(options.try_parse_dates);
        }
        if let Some(schema) = options.schema {
            rdr = rdr.with_schema(schema);
        }
        if let Some(delim) = options.delimiter {
            rdr = rdr.with_delimiter(delim);
        }

        rdr
    }

    impl Default for CsvReaderOptions {
        fn default() -> Self {
            Self {
                chunk_size: 1 << 18,
                columns: None,
                comment_char: None,
                delimiter: None,
                dtypes: None,
                encoding: CsvEncoding::Utf8,
                eol_char: b'\n',
                has_header: true,
                ignore_errors: false,
                infer_schema_length: Some(128),
                low_memory: false,
                missing_is_null: true,
                n_rows: None,
                n_threads: None,
                null_values: None,
                projection: None,
                quote_char: Some(b'"'),
                rechunk: true,
                row_count: None,
                sample_size: 1024,
                schema: None,
                skip_rows_after_header: 0,
                skip_rows_before_header: 0,
                try_parse_dates: false,
            }
        }
    }

    impl CsvReaderOptions {
        pub fn new() -> Self {
            Self::default()
        }
    }
}

#[cfg(all(feature = "lazy", feature = "csv"))]
pub mod scan_csv {
    use polars_core::prelude::{PolarsError, PolarsResult};
    use polars_core::schema::SchemaRef;
    use polars_io::prelude::{CsvEncoding, NullValues};
    use polars_io::RowCount;
    use polars_lazy::prelude::{LazyCsvReader, LazyFileListReader};

    use super::*;

    #[macro_export]
    /// Lazily load a csv file into a LazyFrame
    /// ```rust no_run
    /// # use polars::prelude::*;
    /// # fn main() {
    /// let df = polars::scan_csv!("foo.csv", has_header = true)?
    ///     .select(&[
    ///         col("A"),
    ///         col("B"),
    ///     ])
    ///     .filter(col("A").gt(lit(2)))
    ///     .collect();
    /// }
    macro_rules! scan_csv {
      ($path:expr) => {
          $crate::prelude::LazyCsvReader::new($path).finish()
      };
      ($path:expr, $($field:ident = $value:expr),* $(,)?) => {{
          let rdr = $crate::prelude::LazyCsvReader::new($path);
          let mut options = $crate::io::scan_csv::LazyCsvOptionsBuilder::default();
          $(
              options = options.$field($value);
          )*
          let options = options.build().unwrap();
          let rdr = $crate::io::scan_csv::set_options(rdr, options);

          rdr.finish()
        }}
    }

    impl From<PolarsError> for LazyCsvOptionsBuilderError {
        fn from(value: PolarsError) -> Self {
            LazyCsvOptionsBuilderError::ValidationError(value.to_string())
        }
    }
    #[must_use]
    #[derive(Builder)]
    #[builder(pattern = "immutable")]
    pub struct LazyCsvOptions {
        /// Cache the DataFrame after reading.
        cache: bool,
        /// Set the comment character. Lines starting with this character will be ignored.
        #[builder(setter(strip_option, into))]
        comment_char: Option<u8>,
        /// Set the CSV file's column delimiter as a byte character
        /// example:
        /// ```rust no_run
        /// scan_csv!("foo.csv", delimiter = b',');
        /// ```
        delimiter: u8,
        /// Set  [`CsvEncoding`]
        #[builder(setter(custom))]
        encoding: CsvEncoding,
        /// Single byte end of line character. Defaults to `b'\n'`.
        eol_char: u8,
        /// Set whether the CSV file has headers
        has_header: bool,
        /// Continue with next batch when a ParserError is encountered.
        ignore_errors: bool,
        /// Set the Maximum number of rows read for schema inference
        #[builder(setter(strip_option))]
        infer_schema_length: Option<usize>,
        /// Reduce memory consumption at the expense of performance
        low_memory: bool,
        /// Treat missing fields as null.
        missing_is_null: bool,
        /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
        /// be guaranteed.
        #[builder(setter(strip_option))]
        n_rows: Option<usize>,
        /// Set values that will be interpreted as missing/ null. Note that any value you set as null value
        /// will not be escaped, so if quotation marks are part of the null value you should include them.
        /// ```rust no_run
        /// // NULL will be interpreted as a null value
        /// scan_csv!("data.csv", null_values = "NULL");
        /// // NULL or null will be interpreted as a null value
        /// scan_csv!("data.csv", null_values = ("NULL", "null"));
        /// // "NULL" or null will be interpreted as a null value
        /// scan_csv!("data.csv", null_values = &["\"NULL\"", "null"]);
        /// ```
        #[builder(setter(strip_option, into))]
        null_values: Option<NullValues>,
        /// Set the `char` used as quote char. The default is `b'"'`. If set to `[None]` quoting is disabled.
        quote_char: Option<u8>,
        /// Rechunk the DataFrame to contiguous memory after the CSV is parsed.
        rechunk: bool,
        /// Add a `row_count` column with a provided name
        ///
        /// ```rust no_run
        /// // add a row count column named "id"
        /// polars::scan_csv!("foo.csv", row_count = "id")?;
        /// // add a row count column named "id" with an offset of 100
        /// polars::scan_csv!("foo.csv", row_count = ("id", 100))?;
        /// // use the RowCount struct
        /// polars::scan_csv!("foo.csv", row_count = RowCount {name: "id", offset: 100})?;
        /// ```
        #[builder(setter(strip_option, into))]
        row_count: Option<RowCount>,
        /// Set the CSV file's schema. This only accepts datatypes that are implemented
        /// in the csv parser and expects a complete Schema.
        #[builder(setter(strip_option))]
        schema: Option<SchemaRef>,
        /// Skip these rows after the header
        skip_rows_after_header: usize,
        /// Skip the first `n` rows during parsing. The header will be parsed at `n` lines.
        skip_rows: usize,
        /// Automatically try to parse dates/ datetimes and time. If parsing fails, columns remain of dtype `[DataType::Utf8]`.
        #[cfg(feature = "temporal")]
        try_parse_dates: bool,
    }

    impl LazyCsvOptionsBuilder {
        /// set the encoding
        /// example:
        /// ```rust no_run
        /// scan_csv!("foo.csv", encoding = "utf8");
        /// scan_csv!("foo.csv", encoding = "utf8-lossy");
        /// ```
        pub fn encoding<VALUE>(&self, value: VALUE) -> Self
        where
            VALUE: AsRef<str>,
        {
            let mut opts = self.clone();
            let v = value.as_ref().parse().unwrap_or_else(|_| CsvEncoding::Utf8);

            opts.encoding = Some(v);
            opts
        }
    }

    impl Default for LazyCsvOptions {
        fn default() -> Self {
            Self {
                delimiter: b',',
                has_header: true,
                ignore_errors: false,
                skip_rows: 0,
                n_rows: None,
                cache: true,
                schema: None,
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
                try_parse_dates: false,
            }
        }
    }

    impl LazyCsvOptions {
        pub fn new() -> Self {
            Self::default()
        }
    }

    pub fn set_options<'a>(rdr: LazyCsvReader<'a>, options: LazyCsvOptions) -> LazyCsvReader<'a> {
        let mut rdr = rdr
            .has_header(options.has_header)
            .with_rechunk(options.rechunk)
            .with_skip_rows(options.skip_rows)
            .with_skip_rows_after_header(options.skip_rows_after_header)
            .with_row_count(options.row_count)
            .with_n_rows(options.n_rows)
            .with_infer_schema_length(options.infer_schema_length)
            .with_ignore_errors(options.ignore_errors)
            .with_skip_rows(options.skip_rows_after_header)
            .with_delimiter(options.delimiter)
            .with_comment_char(options.comment_char)
            .with_quote_char(options.quote_char)
            .with_end_of_line_char(options.eol_char)
            .with_null_values(options.null_values)
            .with_missing_is_null(options.missing_is_null)
            .with_cache(options.cache)
            .low_memory(options.low_memory)
            .with_encoding(options.encoding);

        #[cfg(feature = "temporal")]
        {
            rdr = rdr.with_try_parse_dates(options.try_parse_dates)
        }

        if let Some(schema) = options.schema {
            rdr = rdr.with_schema(schema);
        }

        rdr
    }
}

#[cfg(feature = "parquet")]
pub mod read_parquet {
    use polars_core::utils::arrow::io::parquet::write::FileMetaData;
    use polars_io::mmap::MmapBytesReader;
    use polars_io::RowCount;

    use crate::prelude::*;
    #[macro_export]
    macro_rules! read_parquet {
        (&mut $path:expr) => {
            $crate::prelude::ParquetReader::new($path).finish()
        };
        ($path:expr) => {
            $crate::prelude::ParquetReader::from_path($path).and_then(|mut rdr| {
                rdr.finish()
            })
        };
        (&mut $path:expr, $($field:ident = $value:expr),* $(,)?) => {{
            let rdr = $crate::prelude::ParquetReader::new($path);
            let mut options = $crate::io::read_parquet::ParquetReaderOptions::default();
            $(
                options = options.$field($value);
            )*
            let rdr = $crate::io::read_parquet::set_options(rdr, options);
            rdr.finish()
        }};
        ($path:expr, $($field:ident = $value:expr),* $(,)?) => {
            $crate::prelude::ParquetReader::from_path($path).and_then(|rdr| {
                let mut options = $crate::io::read_parquet::ParquetReaderOptions::default();
                $(
                    options = options.$field($value);
                )*
                let rdr = $crate::io::read_parquet::set_options(rdr, options);
                rdr.finish()
            })
        };
    }

    pub fn set_options<R: MmapBytesReader>(
        rdr: ParquetReader<R>,
        options: ParquetReaderOptions,
    ) -> ParquetReader<R> {
        rdr.with_row_count(options.row_count)
            .with_n_rows(options.n_rows)
            .with_columns(options.columns)
            .with_projection(options.projection)
            .read_parallel(options.parallel)
            .set_low_memory(options.low_memory)
            .use_statistics(options.use_statistics)
    }

    pub struct ParquetReaderOptions {
        rechunk: bool,
        parallel: ParallelStrategy,
        low_memory: bool,
        use_statistics: bool,
        n_rows: Option<usize>,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        row_count: Option<RowCount>,
        metadata: Option<FileMetaData>,
    }

    impl Default for ParquetReaderOptions {
        fn default() -> Self {
            Self {
                rechunk: false,
                n_rows: None,
                columns: None,
                projection: None,
                parallel: Default::default(),
                row_count: None,
                low_memory: false,
                metadata: None,
                use_statistics: true,
            }
        }
    }

    impl ParquetReaderOptions {
        impl_set! {
            rechunk: bool,
            parallel: ParallelStrategy,
            low_memory: bool,
            use_statistics: bool,
        }

        impl_set_option! {
            n_rows: usize,
            row_count: RowCount,
            metadata: FileMetaData,
        }
        pub fn columns<S: Into<String>, T: IntoIterator<Item = S>>(mut self, columns: T) -> Self {
            self.columns = Some(columns.into_iter().map(|s| s.into()).collect());
            self
        }
        pub fn projection<T: Into<Vec<usize>>>(mut self, projection: T) -> Self {
            self.projection = Some(projection.into());
            self
        }
    }
}

#[cfg(all(feature = "lazy", feature = "parquet"))]
pub mod scan_parquet {
    use std::path::PathBuf;

    use polars_core::cloud::CloudOptions;
    use polars_core::utils::arrow::io::parquet::write::FileMetaData;
    use polars_io::RowCount;

    use crate::prelude::*;
    pub struct LazyParquetOptions {
        rechunk: bool,
        low_memory: bool,
        cache: bool,
        use_statistics: bool,
        parallel: ParallelStrategy,
        n_rows: Option<usize>,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        row_count: Option<RowCount>,
        metadata: Option<FileMetaData>,
        cloud_options: Option<CloudOptions>,
    }

    impl Default for LazyParquetOptions {
        fn default() -> Self {
            Self {
                rechunk: false,
                n_rows: None,
                columns: None,
                projection: None,
                parallel: Default::default(),
                row_count: None,
                low_memory: false,
                metadata: None,
                use_statistics: true,
                cloud_options: None,
                cache: true,
            }
        }
    }
    #[macro_export]
    /// Lazily load a parquet file into a LazyFrame
    /// ```rust no_run
    /// # use polars::prelude::*;
    /// # fn main() {
    /// let df = polars::scan_parquet!("foo.parquet", parallel = true)?
    ///     .select(&[
    ///         col("A"),
    ///         col("B"),
    ///     ])
    ///     .filter(col("A").gt(lit(2)))
    ///     .collect();
    /// }
    macro_rules! scan_parquet {
      ($path:expr) => {
          $crate::prelude::LazyParquetReader::new($path, Default::default()).finish()
      };
      ($path:expr, $($field:ident = $value:expr),* $(,)?) => {{

          let mut options = $crate::io::scan_parquet::LazyParquetOptions::default();
          $(
              options = options.$field($value);
          )*
          let rdr = $crate::io::scan_parquet::new_reader($path, options);

          rdr.finish()
        }}
    }

    pub fn new_reader(path: PathBuf, options: LazyParquetOptions) -> LazyParquetReader {
        let args = ScanArgsParquet {
            n_rows: options.n_rows,
            cache: options.cache,
            parallel: options.parallel,
            rechunk: options.rechunk,
            row_count: options.row_count,
            low_memory: options.low_memory,
            cloud_options: options.cloud_options,
            use_statistics: options.use_statistics,
        };

        LazyParquetReader::new(path, args)
    }

    impl LazyParquetOptions {
        impl_set! {
            rechunk: bool,
            low_memory: bool,
            cache: bool,
            use_statistics: bool,
            parallel: ParallelStrategy,
        }

        impl_set_option! {
            n_rows: usize,
            row_count: RowCount,
            metadata: FileMetaData,
            cloud_options: CloudOptions
        }
        pub fn columns<S: Into<String>, T: IntoIterator<Item = S>>(mut self, columns: T) -> Self {
            self.columns = Some(columns.into_iter().map(|s| s.into()).collect());
            self
        }
        pub fn projection<T: Into<Vec<usize>>>(mut self, projection: T) -> Self {
            self.projection = Some(projection.into());
            self
        }
    }
}

#[cfg(feature = "ipc")]
pub mod read_ipc {
    use polars_io::mmap::MmapBytesReader;
    use polars_io::RowCount;

    use crate::prelude::*;
    #[macro_export]
    macro_rules! read_ipc {
        (&mut $path:expr) => {
            $crate::prelude::IpcReader::new($path).finish()
        };
        ($path:expr) => {
            $crate::prelude::IpcReader::from_path($path).and_then(|mut rdr| {
                rdr.finish()
            })
        };
        (&mut $path:expr, $($field:ident = $value:expr),* $(,)?) => {{
            let rdr = $crate::prelude::IpcReader::new($path);
            let mut options = $crate::io::read_ipc::IpcReaderOptions::default();
            $(
                options = options.$field($value);
            )*
            let rdr = $crate::io::read_ipc::set_options(rdr, options);
            rdr.finish()
        }};
        ($path:expr, $($field:ident = $value:expr),* $(,)?) => {
            $crate::prelude::IpcReader::from_path($path).and_then(|rdr| {
                let mut options = $crate::io::read_ipc::IpcReaderOptions::default();
                $(
                    options = options.$field($value);
                )*
                let rdr = $crate::io::read_ipc::set_options(rdr, options);
                rdr.finish()
            })
        };
    }

    pub fn set_options<R: MmapBytesReader>(
        rdr: IpcReader<R>,
        options: IpcReaderOptions,
    ) -> IpcReader<R> {
        rdr.with_row_count(options.row_count)
            .with_n_rows(options.n_rows)
            .with_columns(options.columns)
            .with_projection(options.projection)
            .memory_mapped(options.memmap)
    }

    pub struct IpcReaderOptions {
        rechunk: bool,
        cache: bool,
        memmap: bool,
        n_rows: Option<usize>,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        row_count: Option<RowCount>,
    }

    impl Default for IpcReaderOptions {
        fn default() -> Self {
            Self {
                rechunk: false,
                cache: true,
                memmap: true,
                ..Default::default()
            }
        }
    }

    impl IpcReaderOptions {
        impl_set! {
            rechunk: bool,
            cache: bool,
            memmap: bool,
        }

        impl_set_option! {
            n_rows: usize,
            row_count: RowCount,
        }

        pub fn columns<S: Into<String>, T: IntoIterator<Item = S>>(mut self, columns: T) -> Self {
            self.columns = Some(columns.into_iter().map(|s| s.into()).collect());
            self
        }

        pub fn projection<T: Into<Vec<usize>>>(mut self, projection: T) -> Self {
            self.projection = Some(projection.into());
            self
        }
    }
}

#[cfg(all(feature = "lazy", feature = "ipc"))]
pub mod scan_ipc {
    use std::path::PathBuf;

    use polars_io::RowCount;

    use crate::prelude::*;
    pub struct LazyIpcOptions {
        n_rows: Option<usize>,
        cache: bool,
        rechunk: bool,
        row_count: Option<RowCount>,
        memmap: bool,
    }

    impl Default for LazyIpcOptions {
        fn default() -> Self {
            Self {
                n_rows: None,
                cache: true,
                rechunk: true,
                row_count: None,
                memmap: true,
            }
        }
    }

    #[macro_export]
    /// Lazily load a ipc file into a LazyFrame
    /// ```rust no_run
    /// # use polars::prelude::*;
    /// # fn main() {
    /// let df = polars::scan_ipc!("foo.arrow", memmap = true)?
    ///     .select(&[
    ///         col("A"),
    ///         col("B"),
    ///     ])
    ///     .filter(col("A").gt(lit(2)))
    ///     .collect();
    /// }
    macro_rules! scan_ipc {
      ($path:expr) => {
          $crate::prelude::LazyIpcReader::new($path, Default::default()).finish()
      };
      ($path:expr, $($field:ident = $value:expr),* $(,)?) => {{

          let mut options = $crate::io::scan_ipc::LazyIpcOptions::default();
          $(
              options = options.$field($value);
          )*
          let rdr = $crate::io::scan_ipc::new_reader($path, options);

          rdr.finish()
        }}
    }

    pub fn new_reader(path: PathBuf, options: LazyIpcOptions) -> LazyIpcReader {
        let args = ScanArgsIpc {
            n_rows: options.n_rows,
            cache: options.cache,
            rechunk: options.rechunk,
            row_count: options.row_count,
            memmap: options.memmap,
        };

        LazyIpcReader::new(path, args)
    }

    impl LazyIpcOptions {
        impl_set! {
            rechunk: bool,
            cache: bool,
        }

        impl_set_option! {
            n_rows: usize,
            row_count: RowCount,
        }
    }
}
