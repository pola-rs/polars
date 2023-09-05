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
use arrow::legacy::conversion::chunk_to_struct;
use polars_json::json::write::FallibleStreamingIterator;
use polars_core::error::to_compute_err;
use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
use polars_json::json::infer;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use polars_json::{json, ndjson};
use simd_json::BorrowedValue;

use crate::mmap::{MmapBytesReader, ReaderBytes};
use crate::prelude::*;

/// The format to use to write the DataFrame to JSON: `Json` (a JSON array) or `JsonLines` (each row output on a
/// separate line). In either case, each row is serialized as a JSON object whose keys are the column names and whose
/// values are the row's corresponding values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum JsonFormat {
    /// A single JSON array containing each DataFrame row as an object. The length of the array is the number of rows in
    /// the DataFrame.
    ///
    /// Use this to create valid JSON that can be deserialized back into an array in one fell swoop.
    Json,
    /// Each DataFrame row is serialized as a JSON object on a separate line. The number of lines in the output is the
    /// number of rows in the DataFrame.
    ///
    /// The [JSON Lines](https://jsonlines.org) format makes it easy to read records in a streaming fashion, one (line)
    /// at a time. But the output in its entirety is not valid JSON; only the individual lines are.
    ///
    /// It is recommended to use the file extension `.jsonl` when saving as JSON Lines.
    #[default]
    JsonLines,
}

/// Writes a DataFrame to JSON.
///
/// Under the hood, this uses [`arrow2::io::json`](https://docs.rs/arrow2/latest/arrow2/io/json/write/fn.write.html).
/// `arrow2` generally serializes types that are not JSON primitives, such as Date and DateTime, as their
/// `Display`-formatted versions. For instance, a (naive) DateTime column is formatted as the String `"yyyy-mm-dd
/// HH:MM:SS"`. To control how non-primitive columns are serialized, convert them to String or another primitive type
/// before serializing.
#[must_use]
pub struct JsonWriter<W: Write> {
    /// File or Stream handler
    buffer: W,
    json_format: JsonFormat,
}

impl<W: Write> JsonWriter<W>
where
    W: Write + 'static,
{
    pub fn with_json_format(mut self, format: JsonFormat) -> Self {
        self.json_format = format;
        self
    }
}

impl<W> SerWriter<W> for JsonWriter<W>
where
    W: Write,
{
    /// Create a new `JsonWriter` writing to `buffer` with format `JsonFormat::JsonLines`. To specify a different
    /// format, use e.g., [`JsonWriter::new(buffer).with_json_format(JsonFormat::Json)`](JsonWriter::with_json_format).
    fn new(buffer: W) -> Self {
        JsonWriter {
            buffer,
            json_format: JsonFormat::JsonLines,
        }
    }

    fn finish(&mut self, df: &mut DataFrame) -> PolarsResult<()> {
        df.align_chunks();
        let fields = df.iter().map(|s| s.field().to_arrow()).collect::<Vec<_>>();
        let batches = df
            .iter_chunks()
            .map(|chunk| Ok(Box::new(chunk_to_struct(chunk, fields.clone())) as ArrayRef));

        match self.json_format {
            JsonFormat::JsonLines => {
                let serializer = polars_json::ndjson::write::Serializer::new(batches, vec![]);
                let writer =
                    polars_json::ndjson::write::FileWriter::new(&mut self.buffer, serializer);
                writer.collect::<PolarsResult<()>>()?;
            },
            JsonFormat::Json => {
                let serializer = polars_json::json::write::Serializer::new(batches, vec![]);
                polars_json::json::write::write(&mut self.buffer, serializer)?;
            },
        }

        Ok(())
    }
}

pub trait BatchedWriter<W: Write> {
    fn write_batch(&mut self, df: &DataFrame) -> PolarsResult<()>;
    fn finish(&mut self) -> PolarsResult<()>;
}

pub struct JsonBatchedWriter<W: Write> {
    writer: W,
    is_first_row: bool,
}

impl<W> JsonBatchedWriter<W>
where
    W: Write,
{
    pub fn new(writer: W) -> Self {
        JsonBatchedWriter {
            writer,
            is_first_row: true,
        }
    }
}

impl<W> BatchedWriter<W> for JsonBatchedWriter<W>
where
    W: Write,
{
    /// Write a batch to the json writer.
    ///
    /// # Panics
    /// The caller must ensure the chunks in the given [`DataFrame`] are aligned.
    fn write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        let fields = df.iter().map(|s| s.field().to_arrow()).collect::<Vec<_>>();
        let batches = df
            .iter_chunks()
            .map(|chunk| Ok(Box::new(chunk_to_struct(chunk, fields.clone())) as ArrayRef));
        let mut serializer = json::write::Serializer::new(batches, vec![]);

        while let Some(block) = serializer.next()? {
            if self.is_first_row {
                self.writer.write_all(&[b'['])?;
                self.is_first_row = false;
            } else {
                self.writer.write_all(&[b','])?;
            }
            self.writer.write_all(block)?;
        }
        Ok(())
    }

    /// Writes the footer of the Json file.
    fn finish(&mut self) -> PolarsResult<()> {
        self.writer.write_all(&[b']'])?;
        Ok(())
    }
}

pub struct JsonLinesBatchedWriter<W: Write> {
    writer: W,
}

impl<W> JsonLinesBatchedWriter<W>
where
    W: Write,
{
    pub fn new(writer: W) -> Self {
        JsonLinesBatchedWriter { writer }
    }
}

impl<W> BatchedWriter<W> for JsonLinesBatchedWriter<W>
where
    W: Write,
{
    /// Write a batch to the json writer.
    ///
    /// # Panics
    /// The caller must ensure the chunks in the given [`DataFrame`] are aligned.
    fn write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        let fields = df.iter().map(|s| s.field().to_arrow()).collect::<Vec<_>>();
        let batches = df
            .iter_chunks()
            .map(|chunk| Ok(Box::new(chunk_to_struct(chunk, fields.clone())) as ArrayRef));
        let mut serializer = ndjson::write::Serializer::new(batches, vec![]);
        while let Some(block) = serializer.next()? {
            self.writer.write_all(block)?;
        }
        Ok(())
    }

    /// Writes the footer of the Json file.
    fn finish(&mut self) -> PolarsResult<()> {
        Ok(())
    }
}

/// Reads JSON in one of the formats in [`JsonFormat`] into a DataFrame.
#[must_use]
pub struct JsonReader<'a, R>
where
    R: MmapBytesReader,
{
    reader: R,
    rechunk: bool,
    ignore_errors: bool,
    infer_schema_len: Option<usize>,
    batch_size: usize,
    projection: Option<Vec<String>>,
    schema: Option<SchemaRef>,
    schema_overwrite: Option<&'a Schema>,
    json_format: JsonFormat,
}

impl<'a, R> SerReader<R> for JsonReader<'a, R>
where
    R: MmapBytesReader,
{
    fn new(reader: R) -> Self {
        JsonReader {
            reader,
            rechunk: true,
            ignore_errors: false,
            infer_schema_len: Some(100),
            batch_size: 8192,
            projection: None,
            schema: None,
            schema_overwrite: None,
            json_format: JsonFormat::Json,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    /// Take the SerReader and return a parsed DataFrame.
    ///
    /// Because JSON values specify their types (number, string, etc), no upcasting or conversion is performed between
    /// incompatible types in the input. In the event that a column contains mixed dtypes, is it unspecified whether an
    /// error is returned or whether elements of incompatible dtypes are replaced with `null`.
    fn finish(self) -> PolarsResult<DataFrame> {
        let rb: ReaderBytes = (&self.reader).into();

        let out = match self.json_format {
            JsonFormat::Json => {
                polars_ensure!(!self.ignore_errors, InvalidOperation: "'ignore_errors' only supported in ndjson");
                let mut bytes = rb.deref().to_vec();
                let json_value =
                    simd_json::to_borrowed_value(&mut bytes).map_err(to_compute_err)?;

                // struct type
                let dtype = if let Some(mut schema) = self.schema {
                    if let Some(overwrite) = self.schema_overwrite {
                        let mut_schema = Arc::make_mut(&mut schema);
                        overwrite_schema(mut_schema, overwrite)?;
                    }

                    DataType::Struct(schema.iter_fields().collect()).to_arrow()
                } else {
                    // infer
                    let inner_dtype = if let BorrowedValue::Array(values) = &json_value {
                        // struct types may have missing fields so find supertype
                        values
                            .iter()
                            .take(self.infer_schema_len.unwrap_or(usize::MAX))
                            .map(|value| infer(value).map(|dt| DataType::from(&dt)))
                            .reduce(|l, r| {
                                let l = l?;
                                let r = r?;
                                try_get_supertype(&l, &r)
                            })
                            .unwrap()?
                            .to_arrow()
                    } else {
                        infer(&json_value)?
                    };

                    if let Some(overwrite) = self.schema_overwrite {
                        let ArrowDataType::Struct(fields) = inner_dtype else {
                            polars_bail!(ComputeError: "can only deserialize json objects")
                        };

                        let mut schema = Schema::from_iter(fields.iter());
                        overwrite_schema(&mut schema, overwrite)?;

                        DataType::Struct(
                            schema
                                .into_iter()
                                .map(|(name, dt)| Field::new(&name, dt))
                                .collect(),
                        )
                        .to_arrow()
                    } else {
                        inner_dtype
                    }
                };

                let dtype = if let BorrowedValue::Array(_) = &json_value {
                    ArrowDataType::LargeList(Box::new(arrow::datatypes::Field::new(
                        "item", dtype, true,
                    )))
                } else {
                    dtype
                };

                let arr = polars_json::json::deserialize(&json_value, dtype)?;
                let arr = arr.as_any().downcast_ref::<StructArray>().ok_or_else(
                    || polars_err!(ComputeError: "can only deserialize json objects"),
                )?;
                DataFrame::try_from(arr.clone())
            },
            JsonFormat::JsonLines => {
                let mut json_reader = CoreJsonReader::new(
                    rb,
                    None,
                    self.schema,
                    self.schema_overwrite,
                    None,
                    1024, // sample size
                    1 << 18,
                    false,
                    self.infer_schema_len,
                    self.ignore_errors,
                )?;
                let mut df: DataFrame = json_reader.as_df()?;
                if self.rechunk {
                    df.as_single_chunk_par();
                }
                Ok(df)
            },
        }?;

        // TODO! Ensure we don't materialize the columns we don't need
        if let Some(proj) = &self.projection {
            out.select(proj)
        } else {
            Ok(out)
        }
    }
}

impl<'a, R> JsonReader<'a, R>
where
    R: MmapBytesReader,
{
    /// Set the JSON file's schema
    pub fn with_schema(mut self, schema: SchemaRef) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Overwrite parts of the inferred schema.
    pub fn with_schema_overwrite(mut self, schema: &'a Schema) -> Self {
        self.schema_overwrite = Some(schema);
        self
    }

    /// Set the JSON reader to infer the schema of the file. Currently, this is only used when reading from
    /// [`JsonFormat::JsonLines`], as [`JsonFormat::Json`] reads in the entire array anyway.
    ///
    /// When using [`JsonFormat::JsonLines`], `max_records = None` will read the entire buffer in order to infer the
    /// schema, `Some(1)` would look only at the first record, `Some(2)` the first two records, etc.
    ///
    /// It is an error to pass `max_records = Some(0)`, as a schema cannot be inferred from 0 records when deserializing
    /// from JSON (unlike CSVs, there is no header row to inspect for column names).
    pub fn infer_schema_len(mut self, max_records: Option<usize>) -> Self {
        self.infer_schema_len = max_records;
        self
    }

    /// Set the batch size (number of records to load at one time)
    ///
    /// This heavily influences loading time.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the reader's column projection: the names of the columns to keep after deserialization. If `None`, all
    /// columns are kept.
    ///
    /// Setting `projection` to the columns you want to keep is more efficient than deserializing all of the columns and
    /// then dropping the ones you don't want.
    pub fn with_projection(mut self, projection: Option<Vec<String>>) -> Self {
        self.projection = projection;
        self
    }

    pub fn with_json_format(mut self, format: JsonFormat) -> Self {
        self.json_format = format;
        self
    }

    /// Return a `null` if an error occurs during parsing.
    pub fn with_ignore_errors(mut self, ignore: bool) -> Self {
        self.ignore_errors = ignore;
        self
    }
}
