use polars_core::schema::SchemaRef;
use polars_io::mmap::MmapBytesReader;
use polars_io::prelude::{CsvEncoding, CsvReader, NullValues};
use polars_io::RowCount;
pub use polars_io::*;

/// Read a CSV file into a DataFrame.
#[cfg(feature = "csv-file")]
pub mod read_csv {
    use super::*;

    #[macro_export]
    macro_rules! read_csv {
        (&mut $path:expr) => {
            $crate::prelude::CsvReader::new($path).finish()
        };
        ($path:expr) => {
            $crate::prelude::CsvReader::from_path($path).and_then(|mut rdr| {
                rdr.finish()
            })
        };
        ($path:expr, $($field:ident = $value:expr),* $(,)?) => {
            $crate::prelude::CsvReader::from_path($path).and_then(|rdr| {
                let mut options = $crate::io::read_csv::CsvReaderOptions::new();
                $(
                    options = options.$field($value);
                )*
                let rdr = $crate::io::read_csv::set_options(rdr, options);
                rdr.finish()
            })
        }
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

        pub fn rechunk(mut self, rechunk: bool) -> Self {
            self.rechunk = rechunk;
            self
        }

        pub fn n_rows(mut self, n_rows: usize) -> Self {
            self.n_rows = Some(n_rows);
            self
        }
        pub fn infer_schema_length(mut self, infer_schema_length: usize) -> Self {
            self.infer_schema_length = Some(infer_schema_length);
            self
        }
        pub fn skip_rows_before_header(mut self, skip_rows_before_header: usize) -> Self {
            self.skip_rows_before_header = skip_rows_before_header;
            self
        }
        pub fn projection(mut self, projection: Vec<usize>) -> Self {
            self.projection = Some(projection);
            self
        }
        pub fn columns(mut self, columns: Vec<String>) -> Self {
            self.columns = Some(columns);
            self
        }
        pub fn delimiter(mut self, delimiter: u8) -> Self {
            self.delimiter = Some(delimiter);
            self
        }
        pub fn has_header(mut self, has_header: bool) -> Self {
            self.has_header = has_header;
            self
        }
        pub fn ignore_errors(mut self, ignore_errors: bool) -> Self {
            self.ignore_errors = ignore_errors;
            self
        }
        pub fn schema(mut self, schema: SchemaRef) -> Self {
            self.schema = Some(schema);
            self
        }
        pub fn encoding(mut self, encoding: CsvEncoding) -> Self {
            self.encoding = encoding;
            self
        }
        pub fn n_threads(mut self, n_threads: usize) -> Self {
            self.n_threads = Some(n_threads);
            self
        }
        pub fn schema_overwrite(mut self, schema_overwrite: SchemaRef) -> Self {
            self.schema_overwrite = Some(schema_overwrite);
            self
        }
        pub fn sample_size(mut self, sample_size: usize) -> Self {
            self.sample_size = sample_size;
            self
        }
        pub fn chunk_size(mut self, chunk_size: usize) -> Self {
            self.chunk_size = chunk_size;
            self
        }
        pub fn low_memory(mut self, low_memory: bool) -> Self {
            self.low_memory = low_memory;
            self
        }
        pub fn comment_char(mut self, comment_char: u8) -> Self {
            self.comment_char = Some(comment_char);
            self
        }
        pub fn eol_char(mut self, eol_char: u8) -> Self {
            self.eol_char = eol_char;
            self
        }
        pub fn null_values(mut self, null_values: NullValues) -> Self {
            self.null_values = Some(null_values);
            self
        }
        pub fn missing_is_null(mut self, missing_is_null: bool) -> Self {
            self.missing_is_null = missing_is_null;
            self
        }
        pub fn quote_char(mut self, quote_char: u8) -> Self {
            self.quote_char = Some(quote_char);
            self
        }
        pub fn skip_rows_after_header(mut self, skip_rows_after_header: usize) -> Self {
            self.skip_rows_after_header = skip_rows_after_header;
            self
        }
        pub fn try_parse_dates(mut self, try_parse_dates: bool) -> Self {
            self.try_parse_dates = try_parse_dates;
            self
        }
        pub fn row_count(mut self, row_count: RowCount) -> Self {
            self.row_count = Some(row_count);
            self
        }
    }
}

#[cfg(all(feature = "lazy", feature = "csv-file"))]
pub mod scan_csv {
    use polars_core::schema::SchemaRef;
    use polars_io::prelude::{CsvEncoding, NullValues};
    use polars_io::RowCount;
    use polars_lazy::prelude::LazyCsvReader;

    #[macro_export]
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

    #[allow(dead_code)]
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
        pub fn delimiter(mut self, delimiter: u8) -> Self {
            self.delimiter = delimiter;
            self
        }
        pub fn has_header(mut self, has_header: bool) -> Self {
            self.has_header = has_header;
            self
        }
        pub fn ignore_errors(mut self, ignore_errors: bool) -> Self {
            self.ignore_errors = ignore_errors;
            self
        }
        pub fn skip_rows(mut self, skip_rows: usize) -> Self {
            self.skip_rows = skip_rows;
            self
        }
        pub fn n_rows(mut self, n_rows: usize) -> Self {
            self.n_rows = Some(n_rows);
            self
        }
        pub fn cache(mut self, cache: bool) -> Self {
            self.cache = cache;
            self
        }
        pub fn schema(mut self, schema: SchemaRef) -> Self {
            self.schema = Some(schema);
            self
        }
        pub fn low_memory(mut self, low_memory: bool) -> Self {
            self.low_memory = low_memory;
            self
        }
        pub fn comment_char(mut self, comment_char: u8) -> Self {
            self.comment_char = Some(comment_char);
            self
        }
        pub fn quote_char(mut self, quote_char: u8) -> Self {
            self.quote_char = Some(quote_char);
            self
        }
        pub fn eol_char(mut self, eol_char: u8) -> Self {
            self.eol_char = eol_char;
            self
        }
        pub fn null_values<T: Into<NullValues>>(mut self, null_values: T) -> Self {
            self.null_values = Some(null_values.into());
            self
        }
        pub fn missing_is_null(mut self, missing_is_null: bool) -> Self {
            self.missing_is_null = missing_is_null;
            self
        }
        pub fn infer_schema_length(mut self, infer_schema_length: usize) -> Self {
            self.infer_schema_length = Some(infer_schema_length);
            self
        }
        pub fn rechunk(mut self, rechunk: bool) -> Self {
            self.rechunk = rechunk;
            self
        }
        pub fn skip_rows_after_header(mut self, skip_rows_after_header: usize) -> Self {
            self.skip_rows_after_header = skip_rows_after_header;
            self
        }
        pub fn encoding(mut self, encoding: CsvEncoding) -> Self {
            self.encoding = encoding;
            self
        }
        pub fn row_count(mut self, row_count: RowCount) -> Self {
            self.row_count = Some(row_count);
            self
        }
        pub fn try_parse_dates(mut self, try_parse_dates: bool) -> Self {
            self.try_parse_dates = try_parse_dates;
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
    use std::io::Read;

    use polars_core::prelude::*;
    use polars_core::utils::arrow::io::parquet::write::FileMetaData;
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
        ($path:expr, $($field:ident = $value:expr),* $(,)?) => {
            $crate::prelude::CsvReader::from_path($path).and_then(|rdr| {
                let mut options = $crate::io::read_csv::CsvReaderOptions::new();
                $(
                    options = options.$field($value);
                )*
                let rdr = $crate::io::read_csv::set_options(rdr, options);
                rdr.finish()
            })
        }
  }

    pub struct ParquetReaderOptions {
        rechunk: bool,
        n_rows: Option<usize>,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        parallel: ParallelStrategy,
        row_count: Option<RowCount>,
        low_memory: bool,
        metadata: Option<FileMetaData>,
        use_statistics: bool,
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
        pub fn rechunk(mut self, rechunk: bool) -> Self {
            self.rechunk = rechunk;
            self
        }
        pub fn n_rows(mut self, n_rows: usize) -> Self {
            self.n_rows = Some(n_rows);
            self
        }
        pub fn columns<T: Into<Vec<String>>>(mut self, columns: T) -> Self {
            self.columns = Some(columns.into());
            self
        }
        pub fn projection<T: Into<Vec<usize>>>(mut self, projection: T) -> Self {
            self.projection = Some(projection.into());
            self
        }
        pub fn parallel(mut self, parallel: ParallelStrategy) -> Self {
            self.parallel = parallel;
            self
        }
        pub fn row_count(mut self, row_count: RowCount) -> Self {
            self.row_count = Some(row_count);
            self
        }
        pub fn low_memory(mut self, low_memory: bool) -> Self {
            self.low_memory = low_memory;
            self
        }
        pub fn metadata(mut self, metadata: FileMetaData) -> Self {
            self.metadata = Some(metadata);
            self
        }
        pub fn use_statistics(mut self, use_statistics: bool) -> Self {
            self.use_statistics = use_statistics;
            self
        }
    }
}
