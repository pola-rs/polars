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
//! .with_json_format(JsonFormat::JsonLines)
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
use arrow::array::{ArrayRef, StructArray};
use arrow::io::ndjson::read::FallibleStreamingIterator;
pub use arrow::{
    error::Result as ArrowResult,
    io::{json, ndjson},
};
use polars_arrow::conversion::chunk_to_struct;
use polars_arrow::kernels::concatenate::concatenate_owned_unchecked;
use polars_core::prelude::*;
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
        let fields = df.iter().map(|s| s.field().to_arrow()).collect::<Vec<_>>();
        let batches = df
            .iter_chunks()
            .map(|chunk| Ok(Arc::new(chunk_to_struct(chunk, fields.clone())) as ArrayRef));

        match self.json_format {
            JsonFormat::JsonLines => {
                let serializer = ndjson::write::Serializer::new(batches, vec![]);
                let writer = ndjson::write::FileWriter::new(&mut self.buffer, serializer);
                writer.collect::<ArrowResult<()>>()?;
            }
            JsonFormat::Json => {
                let serializer = json::write::Serializer::new(batches, vec![]);
                json::write::write(&mut self.buffer, serializer)?;
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
    json_format: JsonFormat,
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
            json_format: JsonFormat::Json,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> Result<DataFrame> {
        let out = match self.json_format {
            JsonFormat::Json => {
                let v = serde_json::from_reader(&mut self.reader)
                    .map_err(|e| PolarsError::ComputeError(format!("{:?}", e).into()))?;
                // likely struct type
                let dtype = json::read::infer(&v)?;
                let arr = json::read::deserialize(&v, dtype)?;
                let arr = arr.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
                    PolarsError::ComputeError("only can deserialize json objects".into())
                })?;
                DataFrame::try_from(arr.clone())
            }
            JsonFormat::JsonLines => {
                let dtype = ndjson::read::infer(&mut self.reader, self.infer_schema_len)?;
                self.reader.rewind()?;

                let mut reader = ndjson::read::FileReader::new(
                    &mut self.reader,
                    vec!["".to_string(); self.batch_size],
                    None,
                );
                let mut arrays = vec![];
                // `next` is IO-bounded
                while let Some(rows) = reader.next()? {
                    // `deserialize` is CPU-bounded
                    let array = ndjson::read::deserialize(rows, dtype.clone())?;
                    arrays.push(array);
                }
                let arr = concatenate_owned_unchecked(&arrays)?;
                let arr = arr.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
                    PolarsError::ComputeError("only can deserialize json objects".into())
                })?;
                DataFrame::try_from(arr.clone())
            }
        }?;

        if let Some(proj) = &self.projection {
            out.select(proj)
        } else {
            Ok(out)
        }
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

    pub fn with_json_format(mut self, format: JsonFormat) -> Self {
        self.json_format = format;
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
            .with_json_format(JsonFormat::JsonLines)
            .with_batch_size(3)
            .finish()
            .unwrap();

        assert_eq!("a", df.get_columns()[0].name());
        assert_eq!("d", df.get_columns()[3].name());
        assert_eq!((12, 4), df.shape());
    }
}
