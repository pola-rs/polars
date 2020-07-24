use crate::frame::ser::finish_reader;
use crate::prelude::*;
pub use arrow::csv::{ReaderBuilder, WriterBuilder};
use std::io::{Read, Seek, Write};
use std::sync::Arc;

/// Write a DataFrame to csv.
pub struct CsvWriter<'a, W: Write> {
    writer: &'a mut W,
    writer_builder: WriterBuilder,
}

impl<'a, W> CsvWriter<'a, W>
where
    W: Write,
{
    /// Write a DataFrame to a csv file.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// use std::fs::File;
    ///
    /// fn example(df: &DataFrame) -> Result<()> {
    ///     let mut file = File::create("example.csv").expect("could not create file");
    ///
    ///     CsvWriter::new(&mut file)
    ///     .has_headers(true)
    ///     .with_delimiter(b',')
    ///     .finish(df)
    /// }
    /// ```
    pub fn new(writer: &'a mut W) -> Self {
        CsvWriter {
            writer,
            writer_builder: WriterBuilder::new(),
        }
    }

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

    pub fn finish(self, df: &DataFrame) -> Result<()> {
        let mut csv_writer = self.writer_builder.build(self.writer);
        let record_batches = df.as_record_batches()?;

        for batch in &record_batches {
            csv_writer.write(batch)?
        }

        Ok(())
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
        }
    }

    /// Rechunk to one contiguous chunk of memory after all data is read
    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    /// Read the file and create the DataFrame.
    fn finish(self) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        finish_reader(self.reader_builder.build(self.reader)?, rechunk)
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
    use std::io::Cursor;

    #[test]
    fn write_csv() {
        let mut buf: Vec<u8> = Vec::new();
        let df = create_df();

        // TODO: headers not working: https://github.com/apache/arrow/pull/7554
        CsvWriter::new(&mut buf)
            .has_headers(true)
            .finish(&df)
            .expect("csv written");
        let csv = std::str::from_utf8(&buf).unwrap();
        assert_eq!("0,22.1\n1,19.9\n2,7\n3,2\n4,3\n", csv);
    }

    #[test]
    fn read_csv() {
        let s = r#"
"sepal.length","sepal.width","petal.length","petal.width","variety"
5.1,3.5,1.4,.2,"Setosa"
4.9,3,1.4,.2,"Setosa"
4.7,3.2,1.3,.2,"Setosa"
4.6,3.1,1.5,.2,"Setosa"
5,3.6,1.4,.2,"Setosa"
5.4,3.9,1.7,.4,"Setosa"
4.6,3.4,1.4,.3,"Setosa"
"#;

        let file = Cursor::new(s);
        let df = CsvReader::new(file)
            .infer_schema(Some(100))
            .has_header(true)
            .with_batch_size(100)
            .finish()
            .unwrap();

        assert_eq!("sepal.length", df.schema.fields()[0].name());
        assert_eq!(1, df.f_column("sepal.length").chunks().len());
        println!("{:?}", df)
    }
}
