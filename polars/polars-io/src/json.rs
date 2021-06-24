//! # (De)serialize JSON files.
//!
//! ## Write a DataFrame to a JSON file
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
//!     JsonWriter::new(&mut file)
//!     .finish(df)
//! }
//! ```
//!
//! ## Read JSON to a DataFrame
//!
//! ## Example
//!
//! ```
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::io::Cursor;
//!
//! let basic_json = r#"{"a":1, "b":2.0, "c":false, "d":"4"}
//! {"a":-10, "b":-3.5, "c":true, "d":"4"}
//! {"a":2, "b":0.6, "c":false, "d":"text"}
//! {"a":1, "b":2.0, "c":false, "d":"4"}
//! {"a":7, "b":-3.5, "c":true, "d":"4"}
//! {"a":1, "b":0.6, "c":false, "d":"text"}
//! {"a":1, "b":2.0, "c":false, "d":"4"}
//! {"a":5, "b":-3.5, "c":true, "d":"4"}
//! {"a":1, "b":0.6, "c":false, "d":"text"}
//! {"a":1, "b":2.0, "c":false, "d":"4"}
//! {"a":1, "b":-3.5, "c":true, "d":"4"}
//! {"a":100000000000000, "b":0.6, "c":false, "d":"text"}"#;
//! let file = Cursor::new(basic_json);
//! let df = JsonReader::new(file)
//! .infer_schema(Some(3))
//! .with_batch_size(3)
//! .finish()
//! .unwrap();
//!
//! println!("{:?}", df);
//! ```
//! >>> Outputs:
//!
//! ```text
//! +-----+--------+-------+--------+
//! | a   | b      | c     | d      |
//! | --- | ---    | ---   | ---    |
//! | i64 | f64    | bool  | str    |
//! +=====+========+=======+========+
//! | 1   | 2      | false | "4"    |
//! +-----+--------+-------+--------+
//! | -10 | -3.5e0 | true  | "4"    |
//! +-----+--------+-------+--------+
//! | 2   | 0.6    | false | "text" |
//! +-----+--------+-------+--------+
//! | 1   | 2      | false | "4"    |
//! +-----+--------+-------+--------+
//! | 7   | -3.5e0 | true  | "4"    |
//! +-----+--------+-------+--------+
//! | 1   | 0.6    | false | "text" |
//! +-----+--------+-------+--------+
//! | 1   | 2      | false | "4"    |
//! +-----+--------+-------+--------+
//! | 5   | -3.5e0 | true  | "4"    |
//! +-----+--------+-------+--------+
//! | 1   | 0.6    | false | "text" |
//! +-----+--------+-------+--------+
//! | 1   | 2      | false | "4"    |
//! +-----+--------+-------+--------+
//! ```
//!
use crate::finish_reader;
use crate::prelude::*;
use crate::utils::to_arrow_compatible_df;
pub use arrow::json::LineDelimitedWriter;
pub use arrow::json::ReaderBuilder;
use polars_core::prelude::*;
use std::io::{Read, Seek, Write};
use std::sync::Arc;

// Write a DataFrame to JSON
pub struct JsonWriter<'a, W: Write> {
    /// File or Stream handler
    buffer: &'a mut W,
}

impl<'a, W> SerWriter<'a, W> for JsonWriter<'a, W>
where
    W: Write,
{
    fn new(buffer: &'a mut W) -> Self {
        JsonWriter { buffer }
    }

    fn finish(self, df: &DataFrame) -> Result<()> {
        let df = to_arrow_compatible_df(df);
        let mut json_writer = LineDelimitedWriter::new(self.buffer);

        let batches = df.as_record_batches()?;
        json_writer.write_batches(&batches)?;
        json_writer.finish()?;

        Ok(())
    }
}

pub struct JsonReader<R>
where
    R: Read + Seek,
{
    reader: R,
    reader_builder: ReaderBuilder,
    rechunk: bool,
}

impl<R> SerReader<R> for JsonReader<R>
where
    R: Read + Seek,
{
    fn new(reader: R) -> Self {
        JsonReader {
            reader,
            reader_builder: ReaderBuilder::new(),
            rechunk: true,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(self) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        finish_reader(
            self.reader_builder.build(self.reader)?,
            rechunk,
            None,
            None,
            None,
        )
    }
}

impl<R> JsonReader<R>
where
    R: Read + Seek,
{
    /// Set the JSON file's schema
    pub fn with_schema(mut self, schema: &Schema) -> Self {
        self.reader_builder = self.reader_builder.with_schema(Arc::new(schema.to_arrow()));
        self
    }

    /// Set the JSON reader to infer the schema of the file
    pub fn infer_schema(mut self, max_records: Option<usize>) -> Self {
        self.reader_builder = self.reader_builder.infer_schema(max_records);
        self
    }

    /// Set the batch size (number of records to load at one time)
    /// This heavily influences loading time.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.reader_builder = self.reader_builder.with_batch_size(batch_size);
        self
    }

    /// Set the reader's column projection
    pub fn with_projection(mut self, projection: Vec<String>) -> Self {
        self.reader_builder = self.reader_builder.with_projection(projection);
        self
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use std::io::Cursor;

    #[test]
    fn write_json() {
        let mut buf: Vec<u8> = Vec::new();
        let mut df = create_df();

        JsonWriter::new(&mut buf).finish(&mut df);

        let json = std::str::from_utf8(&buf).unwrap();
        assert_eq!(
            r#"{"days":0,"temp":22.1}
{"days":1,"temp":19.9}
{"days":2,"temp":7.0}
{"days":3,"temp":2.0}
{"days":4,"temp":3.0}
"#,
            json
        );
    }

    #[test]
    fn read_json() {
        let basic_json = r#"{"a":1, "b":2.0, "c":false, "d":"4"}
{"a":-10, "b":-3.5, "c":true, "d":"4"}
{"a":2, "b":0.6, "c":false, "d":"text"}
{"a":1, "b":2.0, "c":false, "d":"4"}
{"a":7, "b":-3.5, "c":true, "d":"4"}
{"a":1, "b":0.6, "c":false, "d":"text"}
{"a":1, "b":2.0, "c":false, "d":"4"}
{"a":5, "b":-3.5, "c":true, "d":"4"}
{"a":1, "b":0.6, "c":false, "d":"text"}
{"a":1, "b":2.0, "c":false, "d":"4"}
{"a":1, "b":-3.5, "c":true, "d":"4"}
{"a":100000000000000, "b":0.6, "c":false, "d":"text"}"#;
        let file = Cursor::new(basic_json);
        let df = JsonReader::new(file)
            .infer_schema(Some(3))
            .with_batch_size(3)
            .finish()
            .unwrap();

        println!("{:?}", df);
        assert_eq!("a", df.get_columns()[0].name());
        assert_eq!("d", df.get_columns()[3].name());
        assert_eq!((12, 4), df.shape());
    }
}
