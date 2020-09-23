//! # (De)serializing CSV files
//!
//! ## Write a DataFrame to a csv file.
//!
//! ## Example
//!
//! ```
//! use polars::prelude::*;
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
//! use polars::prelude::*;
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
use crate::prelude::*;
use crate::{frame::ser::finish_reader, utils::clone};
pub use arrow::csv::{ReaderBuilder, WriterBuilder};
use arrow::error::ArrowError;
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

/// Creates a DataFrame after reading a csv.
pub struct CsvReader<R>
where
    R: Read + Seek,
{
    /// File or Stream object
    reader: R,
    /// Builds an Arrow csv reader
    reader_builder: ReaderBuilder,
    /// Aggregates chunk afterwards to a single chunk.
    rechunk: bool,
    /// Continue with next batch when a ParserError is encountered.
    ignore_parser_error: bool,
    // use by error ignore logic
    max_records: Option<usize>,
}

impl<R> SerReader<R> for CsvReader<R>
where
    R: Read + Seek,
{
    /// Create a new CsvReader from a file/ stream
    fn new(reader: R) -> Self {
        CsvReader {
            reader,
            reader_builder: ReaderBuilder::new(),
            rechunk: true,
            ignore_parser_error: false,
            max_records: None,
        }
    }

    /// Rechunk to one contiguous chunk of memory after all data is read
    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    /// Continue with next batch when a ParserError is encountered.
    fn with_ignore_parser_error(mut self) -> Self {
        self.ignore_parser_error = true;
        self
    }

    /// Read the file and create the DataFrame.
    fn finish(mut self) -> Result<DataFrame> {
        // It could be that we could not infer schema due to invalid lines.
        // If we have a CsvError or ParserError we half the number of lines we use for
        // schema inference
        let reader = if self.ignore_parser_error && self.max_records.is_some() {
            if self.max_records < Some(1) {
                return Err(PolarsError::Other("Could not infer schema".into()));
            }
            let reader_val;
            loop {
                let rb = clone(&self.reader_builder);
                match rb.build(clone(&self.reader)) {
                    Err(ArrowError::CsvError(_)) | Err(ArrowError::ParseError(_)) => {
                        self.max_records = self.max_records.map(|v| v / 2);
                        self.reader_builder = self.reader_builder.infer_schema(self.max_records);
                    }
                    Err(e) => return Err(PolarsError::ArrowError(e)),
                    Ok(reader) => {
                        reader_val = reader;
                        break;
                    }
                }
            }
            reader_val
        } else {
            self.reader_builder.build(self.reader)?
        };
        finish_reader(reader, self.rechunk, self.ignore_parser_error)
    }
}

impl<R> CsvReader<R>
where
    R: Read + Seek,
{
    /// Create a new DataFrame by reading a csv file.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// use std::fs::File;
    ///
    /// fn example() -> Result<DataFrame> {
    ///     let file = File::open("iris.csv").expect("could not open file");
    ///
    ///     CsvReader::new(file)
    ///             .infer_schema(None)
    ///             .has_header(true)
    ///             .finish()
    /// }
    /// ```

    /// Set the CSV file's schema
    pub fn with_schema(mut self, schema: Arc<Schema>) -> Self {
        self.reader_builder = self.reader_builder.with_schema(schema);
        self
    }

    /// Set whether the CSV file has headers
    pub fn has_header(mut self, has_header: bool) -> Self {
        self.reader_builder = self.reader_builder.has_header(has_header);
        self
    }

    /// Set the CSV file's column delimiter as a byte character
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.reader_builder = self.reader_builder.with_delimiter(delimiter);
        self
    }

    /// Set the CSV reader to infer the schema of the file
    pub fn infer_schema(mut self, max_records: Option<usize>) -> Self {
        // used by error ignore logic
        self.max_records = max_records;
        self.reader_builder = self.reader_builder.infer_schema(max_records);
        self
    }

    /// Set the batch size (number of records to load at one time)
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.reader_builder = self.reader_builder.with_batch_size(batch_size);
        self
    }

    /// Set the reader's column projection
    pub fn with_projection(mut self, projection: Vec<usize>) -> Self {
        self.reader_builder = self.reader_builder.with_projection(projection);
        self
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
    fn test_parser_error_ignore() {
        use std::io::Cursor;

        let s = r#"
         "sepal.length","sepal.width","petal.length","petal.width","variety"
         5.1,3.5,1.4,.2,"Setosa"
         5.1,3.5,1.4,.2,"Setosa"
         4.9,3,1.4,.2,"Setosa", "extra-column"
         "#;

        let file = Cursor::new(s);

        // just checks if unwrap doesn't panic
        CsvReader::new(file)
            // we also check if infer schema ignores errors
            .infer_schema(Some(10))
            .has_header(true)
            .with_batch_size(2)
            .with_ignore_parser_error()
            .finish()
            .unwrap();
    }
}
