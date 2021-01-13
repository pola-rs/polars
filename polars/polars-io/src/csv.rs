//! # (De)serializing CSV files
//!
//! ## Write a DataFrame to a csv file.
//!
//! ## Example
//!
//! ```
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::fs::File;
//!
//! fn example(df: &mut DataFrame) -> Result<()> {
//!     let mut file = File::create("example.csv").expect("could not create file");
//!
//!     CsvWriter::new(&mut file)
//!     .has_headers(true)
//!     .with_delimiter(b',')
//!     .finish(df)
//! }
//! ```
//!
//! ## Read a csv file to a DataFrame
//!
//! ## Example
//!
//! ```
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::io::Cursor;
//!
//! let s = r#"
//! "sepal.length","sepal.width","petal.length","petal.width","variety"
//! 5.1,3.5,1.4,.2,"Setosa"
//! 4.9,3,1.4,.2,"Setosa"
//! 4.7,3.2,1.3,.2,"Setosa"
//! 4.6,3.1,1.5,.2,"Setosa"
//! 5,3.6,1.4,.2,"Setosa"
//! 5.4,3.9,1.7,.4,"Setosa"
//! 4.6,3.4,1.4,.3,"Setosa"
//! "#;
//!
//! let file = Cursor::new(s);
//! let df = CsvReader::new(file)
//! .infer_schema(Some(100))
//! .has_header(true)
//! .with_batch_size(100)
//! .finish()
//! .unwrap();
//!
//! assert_eq!("sepal.length", df.get_columns()[0].name());
//! # assert_eq!(1, df.column("sepal.length").unwrap().chunks().len());
//! ```
use crate::fork::csv::{build_csv_reader, SequentialReader};
use crate::{SerReader, SerWriter};
pub use arrow::csv::WriterBuilder;
use polars_core::prelude::*;
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::sync::Arc;

/// Write a DataFrame to csv.
pub struct CsvWriter<'a, W: Write> {
    /// File or Stream handler
    buffer: &'a mut W,
    /// Builds an Arrow CSV Writer
    writer_builder: WriterBuilder,
    buffer_size: usize,
}

impl<'a, W> SerWriter<'a, W> for CsvWriter<'a, W>
where
    W: Write,
{
    fn new(buffer: &'a mut W) -> Self {
        CsvWriter {
            buffer,
            writer_builder: WriterBuilder::new(),
            buffer_size: 1000,
        }
    }

    fn finish(self, df: &mut DataFrame) -> Result<()> {
        let mut csv_writer = self.writer_builder.build(self.buffer);

        let iter = df.iter_record_batches(self.buffer_size);
        for batch in iter {
            csv_writer.write(&batch)?
        }
        Ok(())
    }
}

impl<'a, W> CsvWriter<'a, W>
where
    W: Write,
{
    /// Set whether to write headers
    pub fn has_headers(mut self, has_headers: bool) -> Self {
        self.writer_builder = self.writer_builder.has_headers(has_headers);
        self
    }

    /// Set the CSV file's column delimiter as a byte character
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.writer_builder = self.writer_builder.with_delimiter(delimiter);
        self
    }

    /// Set the CSV file's date format
    pub fn with_date_format(mut self, format: String) -> Self {
        self.writer_builder = self.writer_builder.with_date_format(format);
        self
    }

    /// Set the CSV file's time format
    pub fn with_time_format(mut self, format: String) -> Self {
        self.writer_builder = self.writer_builder.with_time_format(format);
        self
    }

    /// Set the CSV file's timestamp formatch array in
    pub fn with_timestamp_format(mut self, format: String) -> Self {
        self.writer_builder = self.writer_builder.with_timestamp_format(format);
        self
    }

    /// Set the size of the write buffers. Batch size is the amount of rows written at once.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.buffer_size = batch_size;
        self
    }
}

#[derive(Copy, Clone)]
pub enum CsvEncoding {
    Utf8,
    LossyUtf8,
}

/// Create a new DataFrame by reading a csv file.
///
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// use polars_io::prelude::*;
/// use std::fs::File;
///
/// fn example() -> Result<DataFrame> {
///
///     CsvReader::from_path("iris_csv")?
///             .infer_schema(None)
///             .has_header(true)
///             .finish()
/// }
/// ```
pub struct CsvReader<'a, R>
where
    R: Read + Seek,
{
    /// File or Stream object
    reader: R,
    /// Aggregates chunk afterwards to a single chunk.
    pub rechunk: bool,
    /// Stop reading from the csv after this number of rows is reached
    stop_after_n_rows: Option<usize>,
    // used by error ignore logic
    max_records: Option<usize>,
    skip_rows: usize,
    /// Optional indexes of the columns to project
    projection: Option<Vec<usize>>,
    /// Optional column names to project/ select.
    columns: Option<Vec<String>>,
    batch_size: usize,
    delimiter: Option<u8>,
    has_header: bool,
    ignore_parser_errors: bool,
    schema: Option<Arc<Schema>>,
    encoding: CsvEncoding,
    n_threads: Option<usize>,
    path: Option<String>,
    schema_overwrite: Option<&'a Schema>,
}

impl<'a, R> CsvReader<'a, R>
where
    R: 'static + Read + Seek + Sync + Send,
{
    pub fn with_encoding(mut self, enc: CsvEncoding) -> Self {
        self.encoding = enc;
        self
    }

    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    pub fn with_stop_after_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.stop_after_n_rows = num_rows;
        self
    }

    /// Continue with next batch when a ParserError is encountered.
    pub fn with_ignore_parser_errors(mut self, ignore: bool) -> Self {
        self.ignore_parser_errors = ignore;
        self
    }

    /// Set the CSV file's schema
    pub fn with_schema(mut self, schema: Arc<Schema>) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Skip the first `n` rows during parsing.
    pub fn with_skip_rows(mut self, skip_rows: usize) -> Self {
        self.skip_rows = skip_rows;
        self
    }

    /// Rechunk the DataFrame to contiguous memory after the CSV is parsed.
    pub fn with_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    /// Set whether the CSV file has headers
    pub fn has_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Set the CSV file's column delimiter as a byte character
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = Some(delimiter);
        self
    }

    /// Overwrite the schema with the dtypes in this given Schema. The given schema may be a subset
    /// of the total schema.
    pub fn with_dtype_overwrite(mut self, schema: Option<&'a Schema>) -> Self {
        self.schema_overwrite = schema;
        self
    }

    /// Set the CSV reader to infer the schema of the file
    pub fn infer_schema(mut self, max_records: Option<usize>) -> Self {
        // used by error ignore logic
        self.max_records = max_records;
        self
    }

    /// Set the batch size (number of records to load at one time)
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the reader's column projection
    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
        self
    }

    /// Columns to select/ project
    pub fn with_columns(mut self, columns: Option<Vec<String>>) -> Self {
        self.columns = columns;
        self
    }

    /// Set the number of threads used in CSV reading. The default uses the number of cores of
    /// your cpu.
    ///
    /// Note that this only works if this is initialized with `CsvReader::from_path`.
    /// Note that the number of cores is the maximum allowed number of threads.
    pub fn with_n_threads(mut self, n: Option<usize>) -> Self {
        self.n_threads = n;
        self
    }

    pub fn with_path(mut self, path: Option<String>) -> Self {
        self.path = path;
        self
    }

    pub fn build_inner_reader(self) -> Result<SequentialReader<R>> {
        build_csv_reader(
            self.reader,
            self.stop_after_n_rows,
            self.skip_rows,
            self.projection,
            self.batch_size,
            self.max_records,
            self.delimiter,
            self.has_header,
            self.ignore_parser_errors,
            self.schema,
            self.columns,
            self.encoding,
            self.n_threads,
            self.path,
            self.schema_overwrite,
        )
    }
}

impl<'a> CsvReader<'a, File> {
    /// This is the recommended way to create a csv reader as this allows for fastest parsing.
    pub fn from_path(path: &str) -> Result<Self> {
        let f = std::fs::File::open(path)?;
        Ok(Self::new(f).with_path(Some(path.to_string())))
    }
}

impl<'a, R> SerReader<R> for CsvReader<'a, R>
where
    R: 'static + Read + Seek + Sync + Send,
{
    /// Create a new CsvReader from a file/ stream
    fn new(reader: R) -> Self {
        CsvReader {
            reader,
            rechunk: true,
            stop_after_n_rows: None,
            max_records: Some(100),
            skip_rows: 0,
            projection: None,
            batch_size: 32,
            delimiter: None,
            has_header: true,
            ignore_parser_errors: false,
            schema: None,
            columns: None,
            encoding: CsvEncoding::Utf8,
            n_threads: None,
            path: None,
            schema_overwrite: None,
        }
    }

    /// Read the file and create the DataFrame.
    fn finish(self) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        let mut csv_reader = self.build_inner_reader()?;
        let df = csv_reader.as_df(None, None)?;
        match rechunk {
            true => Ok(df.agg_chunks()),
            false => Ok(df),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn write_csv() {
        let mut buf: Vec<u8> = Vec::new();
        let mut df = create_df();

        CsvWriter::new(&mut buf)
            .has_headers(true)
            .finish(&mut df)
            .expect("csv written");
        let csv = std::str::from_utf8(&buf).unwrap();
        assert_eq!("days,temp\n0,22.1\n1,19.9\n2,7\n3,2\n4,3\n", csv);
    }

    #[test]
    fn test_read_csv_file() {
        let path = "../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv";
        let file = std::fs::File::open(path).unwrap();
        let df = CsvReader::new(file)
            .with_path(Some(path.to_string()))
            .finish()
            .unwrap();
        dbg!(df);
    }

    #[test]
    fn test_parser() {
        use std::io::Cursor;

        let s = r#"
 "sepal.length","sepal.width","petal.length","petal.width","variety"
 5.1,3.5,1.4,.2,"Setosa"
 4.9,3,1.4,.2,"Setosa"
 4.7,3.2,1.3,.2,"Setosa"
 4.6,3.1,1.5,.2,"Setosa"
 5,3.6,1.4,.2,"Setosa"
 5.4,3.9,1.7,.4,"Setosa"
 4.6,3.4,1.4,.3,"Setosa"#;

        let file = Cursor::new(s);
        let df = CsvReader::new(file)
            .infer_schema(Some(100))
            .has_header(true)
            .with_ignore_parser_errors(true)
            .with_batch_size(100)
            .finish()
            .unwrap();
        dbg!(df.select_at_idx(0).unwrap().n_chunks());

        let s = r#"
         "sepal.length","sepal.width","petal.length","petal.width","variety"
         5.1,3.5,1.4,.2,"Setosa"
         5.1,3.5,1.4,.2,"Setosa"#;

        let file = Cursor::new(s);

        // just checks if unwrap doesn't panic
        CsvReader::new(file)
            // we also check if infer schema ignores errors
            .infer_schema(Some(10))
            .has_header(true)
            .with_batch_size(2)
            .with_ignore_parser_errors(true)
            .finish()
            .unwrap();
    }
}
