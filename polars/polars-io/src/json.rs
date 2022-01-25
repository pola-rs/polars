//! # (De)serialize JSON files.
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
//! {"a":1, "b":0.6, "c":false, "d":"text"}"#;
//! let file = Cursor::new(basic_json);
//! let df = JsonReader::new(file)
//! .infer_schema_len(Some(3))
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
use crate::prelude::*;
pub use arrow::{error::Result as ArrowResult, io::json::read, io::json::write};
use polars_core::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical;
use std::convert::TryFrom;
use std::io::{BufRead, Seek, Write};

pub enum JsonFormat {
    Json,
    JsonLines,
}

// Write a DataFrame to JSON
#[must_use]
pub struct JsonWriter<W: Write> {
    /// File or Stream handler
    buffer: W,
    json_format: JsonFormat,
}

impl<W: Write> JsonWriter<W> {
    pub fn with_json_format(mut self, format: JsonFormat) -> Self {
        self.json_format = format;
        self
    }
}

impl<W> SerWriter<W> for JsonWriter<W>
where
    W: Write,
{
    fn new(buffer: W) -> Self {
        JsonWriter {
            buffer,
            json_format: JsonFormat::JsonLines,
        }
    }

    fn finish(mut self, df: &mut DataFrame) -> Result<()> {
        df.rechunk();
        let batches = df.iter_chunks().map(Ok);
        let names = df.get_column_names_owned();

        match self.json_format {
            JsonFormat::JsonLines => {
                let format = write::LineDelimited::default();

                let blocks =
                    write::Serializer::new(batches, names, Vec::with_capacity(1024), format);
                write::write(&mut self.buffer, format, blocks)?;
            }
            JsonFormat::Json => {
                let format = write::JsonArray::default();
                let blocks =
                    write::Serializer::new(batches, names, Vec::with_capacity(1024), format);
                write::write(&mut self.buffer, format, blocks)?;
            }
        }

        Ok(())
    }
}

#[must_use]
pub struct JsonReader<R>
where
    R: BufRead + Seek,
{
    reader: R,
    rechunk: bool,
    infer_schema_len: Option<usize>,
    batch_size: usize,
    projection: Option<Vec<String>>,
    schema: Option<ArrowSchema>,
}

impl<R> SerReader<R> for JsonReader<R>
where
    R: BufRead + Seek,
{
    fn new(reader: R) -> Self {
        JsonReader {
            reader,
            rechunk: true,
            infer_schema_len: Some(100),
            batch_size: 8192,
            projection: None,
            schema: None,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> Result<DataFrame> {
        let rechunk = self.rechunk;

        let fields = if let Some(schema) = self.schema {
            schema.fields
        } else {
            read::infer_and_reset(&mut self.reader, self.infer_schema_len)?
        };
        let projection = self
            .projection
            .map(|projection| {
                projection
                    .iter()
                    .map(|name| {
                        fields
                            .iter()
                            .position(|fld| &fld.name == name)
                            .ok_or_else(|| PolarsError::NotFound(name.into()))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;

        let mut dfs = vec![];

        // at most  rows. This container can be re-used across batches.
        let mut rows = vec![String::default(); self.batch_size];
        loop {
            let read = read::read_rows(&mut self.reader, &mut rows)?;
            if read == 0 {
                break;
            }
            let read_rows = &rows[..read];
            let rb = read::deserialize(read_rows, &fields)?;
            let df = DataFrame::try_from((rb, fields.as_slice()))?;
            let cols = df.get_columns();

            if let Some(projection) = &projection {
                let cols = projection
                    .iter()
                    .map(|idx| cols[*idx].clone())
                    .collect::<Vec<_>>();
                dfs.push(DataFrame::new_no_checks(cols))
            } else {
                dfs.push(df)
            }
        }

        let mut out = accumulate_dataframes_vertical(dfs.into_iter())?;
        if rechunk {
            out.rechunk();
        }
        Ok(out)
    }
}

impl<R> JsonReader<R>
where
    R: BufRead + Seek,
{
    /// Set the JSON file's schema
    pub fn with_schema(mut self, schema: &Schema) -> Self {
        self.schema = Some(schema.to_arrow());
        self
    }

    /// Set the JSON reader to infer the schema of the file
    pub fn infer_schema_len(mut self, max_records: Option<usize>) -> Self {
        self.infer_schema_len = max_records;
        self
    }

    /// Set the batch size (number of records to load at one time)
    /// This heavily influences loading time.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the reader's column projection
    pub fn with_projection(mut self, projection: Option<Vec<String>>) -> Self {
        self.projection = projection;
        self
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use std::io::Cursor;

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
            .infer_schema_len(Some(3))
            .with_batch_size(3)
            .finish()
            .unwrap();

        println!("{:?}", df);
        assert_eq!("a", df.get_columns()[0].name());
        assert_eq!("d", df.get_columns()[3].name());
        assert_eq!((12, 4), df.shape());
    }
}
