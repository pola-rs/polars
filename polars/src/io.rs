use polars_core::schema::SchemaRef;
use polars_io::mmap::MmapBytesReader;
pub use polars_io::{RowCount, *};

macro_rules! impl_set {
    ($($field:ident: $arg_type:ty),* $(,)?) => {
        $(
            pub fn $field(mut self, value: $arg_type) -> Self {
                self.$field = value;
                self
            }
        )*
    };
}

macro_rules! impl_set_option {
    ($($field:ident: $arg_type:ty),* $(,)?) => {
        $(
            pub fn $field(mut self, value: $arg_type) -> Self {
                self.$field = Some(value);
                self
            }
        )*
    };
}

/// Read a CSV file into a DataFrame.
#[cfg(feature = "csv")]
pub mod read_csv {
    use polars_io::prelude::{CsvEncoding, CsvReader, NullValues};

    use super::*;
    #[macro_export]
    /// Read a CSV file into a DataFrame.
    /// This macro is a convenience macro to read a CSV file into a DataFrame.
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
    /// _Note: PathBuf must be taken by reference: `&path`. This is because the macro is not able to disambiguate between a PathBuf and a string literal_
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
                let mut options = $crate::io::read_csv::CsvReaderOptions::new();
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
                let mut options = $crate::io::read_csv::CsvReaderOptions::new();
                $(
                    options = options.$field($value);
                )*
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
            let mut options = $crate::io::read_csv::CsvReaderOptions::new();
            $(
                options = options.$field($value);
            )*
            let rdr = $crate::io::read_csv::set_options(rdr, options);
            rdr.finish()
        }};

  }
    #[must_use]
    pub struct CsvReaderOptions {
        rechunk: bool,
        n_rows: Option<usize>,
        infer_schema_length: Option<usize>,
        skip_rows_before_header: usize,
        projection: Option<Vec<usize>>,
        columns: Option<Vec<String>>,
        delimiter: Option<u8>,
        has_header: bool,
        ignore_errors: bool,
        schema: Option<SchemaRef>,
        encoding: CsvEncoding,
        n_threads: Option<usize>,
        schema_overwrite: Option<SchemaRef>,
        sample_size: usize,
        chunk_size: usize,
        low_memory: bool,
        comment_char: Option<u8>,
        eol_char: u8,
        null_values: Option<NullValues>,
        missing_is_null: bool,
        quote_char: Option<u8>,
        skip_rows_after_header: usize,
        try_parse_dates: bool,
        row_count: Option<RowCount>,
    }

    pub fn set_options<'a, R: MmapBytesReader + 'a>(
        rdr: CsvReader<'a, R>,
        options: CsvReaderOptions,
    ) -> CsvReader<'a, R> {
        let mut rdr = rdr
            .with_skip_rows_after_header(options.skip_rows_after_header)
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
            .with_missing_is_null(options.missing_is_null);
        if let Some(schema) = options.schema {
            rdr = rdr.with_schema(schema);
        }
        if let Some(delim) = options.delimiter {
            rdr = rdr.with_delimiter(delim);
        }

        rdr
    }

    impl CsvReaderOptions {
        pub fn new() -> Self {
            Self {
                rechunk: true,
                n_rows: None,
                infer_schema_length: Some(128),
                skip_rows_before_header: 0,
                projection: None,
                delimiter: None,
                has_header: true,
                ignore_errors: false,
                schema: None,
                columns: None,
                encoding: CsvEncoding::Utf8,
                n_threads: None,
                schema_overwrite: None,
                sample_size: 1024,
                chunk_size: 1 << 18,
                low_memory: false,
                comment_char: None,
                eol_char: b'\n',
                null_values: None,
                missing_is_null: true,
                quote_char: Some(b'"'),
                skip_rows_after_header: 0,
                try_parse_dates: false,
                row_count: None,
            }
        }
        impl_set! {
            rechunk: bool,
            skip_rows_before_header: usize,
            has_header: bool,
            ignore_errors: bool,
            encoding: CsvEncoding,
            sample_size: usize,
            chunk_size: usize,
            low_memory: bool,
            eol_char: u8,
            missing_is_null: bool,
            skip_rows_after_header: usize,
            try_parse_dates: bool
        }

        impl_set_option! {
            n_rows: usize,
            infer_schema_length: usize,
            projection: Vec<usize>,
            columns: Vec<String>,
            delimiter: u8,
            schema: SchemaRef,
            n_threads: usize,
            schema_overwrite: SchemaRef,
            comment_char: u8,
            quote_char: u8,
            row_count: RowCount
        }

        pub fn null_values<T: Into<NullValues>>(mut self, null_values: T) -> Self {
            self.null_values = Some(null_values.into());
            self
        }
    }
}

#[cfg(all(feature = "lazy", feature = "csv"))]
pub mod scan_csv {
    use polars_core::schema::SchemaRef;
    use polars_io::prelude::{CsvEncoding, NullValues};
    use polars_io::RowCount;
    use polars_lazy::prelude::LazyCsvReader;

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
          let mut options = $crate::io::scan_csv::LazyCsvOptions::new();
          $(
              options = options.$field($value);
          )*
          let rdr = $crate::io::scan_csv::set_options(rdr, options);

          rdr.finish()
        }}
    }

    pub struct LazyCsvOptions {
        delimiter: u8,
        has_header: bool,
        ignore_errors: bool,
        skip_rows: usize,
        n_rows: Option<usize>,
        cache: bool,
        schema: Option<SchemaRef>,
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
        try_parse_dates: bool,
    }

    impl LazyCsvOptions {
        pub fn new() -> Self {
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

        impl_set! {
            delimiter: u8,
            has_header: bool,
            ignore_errors: bool,
            skip_rows: usize,
            cache: bool,
            low_memory: bool,
            eol_char: u8,
            missing_is_null: bool,
            rechunk: bool,
            skip_rows_after_header: usize,
            encoding: CsvEncoding,
            try_parse_dates: bool
        }
        impl_set_option! {
            n_rows: usize,
            schema: SchemaRef,
            comment_char: u8,
            quote_char: u8,
            infer_schema_length: usize,
            row_count: RowCount
        }

        pub fn null_values<T: Into<NullValues>>(mut self, null_values: T) -> Self {
            self.null_values = Some(null_values.into());
            self
        }
    }

    pub fn set_options<'a>(rdr: LazyCsvReader<'a>, options: LazyCsvOptions) -> LazyCsvReader<'a> {
        let mut rdr = rdr
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
            .with_encoding(options.encoding)
            .with_try_parse_dates(options.try_parse_dates);

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
