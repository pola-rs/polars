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
use std::convert::TryFrom;
use std::io::Write;
use std::ops::Deref;

use arrow::array::StructArray;
pub use arrow::error::Result as ArrowResult;
pub use arrow::io::json;
use polars_arrow::conversion::chunk_to_struct;
use polars_core::prelude::*;

use crate::mmap::{MmapBytesReader, ReaderBytes};
use crate::prelude::*;

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

    fn finish(&mut self, df: &mut DataFrame) -> PolarsResult<()> {
        df.rechunk();
        let fields = df.iter().map(|s| s.field().to_arrow()).collect::<Vec<_>>();
        let batches = df
            .iter_chunks()
            .map(|chunk| Ok(Box::new(chunk_to_struct(chunk, fields.clone())) as ArrayRef));

        match self.json_format {
            JsonFormat::JsonLines => {
                let serializer = arrow_ndjson::write::Serializer::new(batches, vec![]);
                let writer = arrow_ndjson::write::FileWriter::new(&mut self.buffer, serializer);
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
    R: MmapBytesReader,
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
    R: MmapBytesReader,
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

    fn finish(self) -> PolarsResult<DataFrame> {
        let rb: ReaderBytes = (&self.reader).into();

        let out = match self.json_format {
            JsonFormat::Json => {
                let bytes = rb.deref();
                let json_value = json::read::json_deserializer::parse(bytes)
                    .map_err(|err| PolarsError::ComputeError(format!("{err:?}").into()))?;
                // likely struct type
                let dtype = json::read::infer(&json_value)?;
                let arr = json::read::deserialize(&json_value, dtype)?;
                let arr = arr.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
                    PolarsError::ComputeError("only can deserialize json objects".into())
                })?;
                DataFrame::try_from(arr.clone())
            }
            JsonFormat::JsonLines => {
                let mut json_reader = CoreJsonReader::new(
                    rb,
                    None,
                    None,
                    None,
                    1024, // sample size
                    1 << 18,
                    false,
                    self.infer_schema_len,
                )?;
                let mut df: DataFrame = json_reader.as_df()?;
                if self.rechunk {
                    df.as_single_chunk_par();
                }
                Ok(df)
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
    R: MmapBytesReader,
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
