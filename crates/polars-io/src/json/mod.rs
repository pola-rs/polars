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
//! use std::num::NonZeroUsize;
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
//! .infer_schema_len(NonZeroUsize::new(3))
//! .with_batch_size(NonZeroUsize::new(3).unwrap())
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
pub(crate) mod infer;

use std::io::Write;
use std::num::NonZeroUsize;
use std::ops::Deref;

use arrow::array::LIST_VALUES_NAME;
use arrow::legacy::conversion::chunk_to_struct;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::error::to_compute_err;
use polars_core::prelude::*;
use polars_error::{PolarsResult, polars_bail};
use polars_json::json::write::FallibleStreamingIterator;
use simd_json::BorrowedValue;

use crate::mmap::{MmapBytesReader, ReaderBytes};
use crate::prelude::*;

/// The format to use to write the DataFrame to JSON: `Json` (a JSON array)
/// or `JsonLines` (each row output on a separate line).
///
/// In either case, each row is serialized as a JSON object whose keys are the column names and
/// whose values are the row's corresponding values.
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
    /// Create a new `JsonWriter` writing to `buffer` with format `JsonFormat::JsonLines`. To specify a different
    /// format, use e.g., [`JsonWriter::new(buffer).with_json_format(JsonFormat::Json)`](JsonWriter::with_json_format).
    fn new(buffer: W) -> Self {
        JsonWriter {
            buffer,
            json_format: JsonFormat::JsonLines,
        }
    }

    fn finish(&mut self, df: &mut DataFrame) -> PolarsResult<()> {
        df.align_chunks_par();
        let fields = df.columns()
            .iter()
            .map(|s| {
                #[cfg(feature = "object")]
                polars_ensure!(!matches!(s.dtype(), DataType::Object(_)), ComputeError: "cannot write 'Object' datatype to json");
                Ok(s.field().to_arrow(CompatLevel::newest()))
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        let batches = df
            .iter_chunks(CompatLevel::newest(), false)
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

pub struct BatchedWriter<W: Write> {
    writer: W,
}

impl<W> BatchedWriter<W>
where
    W: Write,
{
    pub fn new(writer: W) -> Self {
        BatchedWriter { writer }
    }
    /// Write a batch to the json writer.
    ///
    /// # Panics
    /// The caller must ensure the chunks in the given [`DataFrame`] are aligned.
    pub fn write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        let fields = df.columns()
            .iter()
            .map(|s| {
                #[cfg(feature = "object")]
                polars_ensure!(!matches!(s.dtype(), DataType::Object(_)), ComputeError: "cannot write 'Object' datatype to json");
                Ok(s.field().to_arrow(CompatLevel::newest()))
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        let chunks = df.iter_chunks(CompatLevel::newest(), false);
        let batches =
            chunks.map(|chunk| Ok(Box::new(chunk_to_struct(chunk, fields.clone())) as ArrayRef));
        let mut serializer = polars_json::ndjson::write::Serializer::new(batches, vec![]);
        while let Some(block) = serializer.next()? {
            self.writer.write_all(block)?;
        }
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
    infer_schema_len: Option<NonZeroUsize>,
    batch_size: NonZeroUsize,
    projection: Option<Vec<PlSmallStr>>,
    schema: Option<SchemaRef>,
    schema_overwrite: Option<&'a Schema>,
    json_format: JsonFormat,
}

pub fn remove_bom(bytes: &[u8]) -> PolarsResult<&[u8]> {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        // UTF-8 BOM
        Ok(&bytes[3..])
    } else if bytes.starts_with(&[0xFE, 0xFF]) || bytes.starts_with(&[0xFF, 0xFE]) {
        // UTF-16 BOM
        polars_bail!(ComputeError: "utf-16 not supported")
    } else {
        Ok(bytes)
    }
}
impl<R> SerReader<R> for JsonReader<'_, R>
where
    R: MmapBytesReader,
{
    fn new(reader: R) -> Self {
        JsonReader {
            reader,
            rechunk: true,
            ignore_errors: false,
            infer_schema_len: Some(NonZeroUsize::new(100).unwrap()),
            batch_size: NonZeroUsize::new(8192).unwrap(),
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
    fn finish(mut self) -> PolarsResult<DataFrame> {
        let pre_rb: ReaderBytes = (&mut self.reader).into();
        let bytes = remove_bom(pre_rb.deref())?;
        let rb = ReaderBytes::Borrowed(bytes);
        let out = match self.json_format {
            JsonFormat::Json => {
                polars_ensure!(!self.ignore_errors, InvalidOperation: "'ignore_errors' only supported in ndjson");
                let mut bytes = rb.deref().to_vec();
                let owned = &mut vec![];
                #[expect(deprecated)] // JSON is not a row-format
                compression::maybe_decompress_bytes(&bytes, owned)?;
                // the easiest way to avoid ownership issues is by implicitly figuring out if
                // decompression happened (owned is only populated on decompress), then pick which bytes to parse
                let json_value = if owned.is_empty() {
                    simd_json::to_borrowed_value(&mut bytes).map_err(to_compute_err)?
                } else {
                    simd_json::to_borrowed_value(owned).map_err(to_compute_err)?
                };
                if let BorrowedValue::Array(array) = &json_value {
                    if array.is_empty() & self.schema.is_none() & self.schema_overwrite.is_none() {
                        return Ok(DataFrame::empty());
                    }
                }

                let allow_extra_fields_in_struct = self.schema.is_some();

                let mut schema = if let Some(schema) = self.schema {
                    Arc::unwrap_or_clone(schema)
                } else {
                    // Infer.
                    let inner_dtype = if let BorrowedValue::Array(values) = &json_value {
                        infer::json_values_to_supertype(
                            values,
                            self.infer_schema_len
                                .unwrap_or(NonZeroUsize::new(usize::MAX).unwrap()),
                        )?
                    } else {
                        DataType::from_arrow_dtype(&polars_json::json::infer(&json_value)?)
                    };

                    let DataType::Struct(fields) = inner_dtype else {
                        polars_bail!(ComputeError: "can only deserialize json objects")
                    };

                    Schema::from_iter(fields)
                };

                if let Some(overwrite) = self.schema_overwrite {
                    overwrite_schema(&mut schema, overwrite)?;
                }

                let mut needs_cast = false;
                let deserialize_schema = schema
                    .iter()
                    .map(|(name, dt)| {
                        Field::new(
                            name.clone(),
                            dt.clone().map_leaves(&mut |leaf_dt| {
                                // Deserialize enums and categoricals as strings first.
                                match leaf_dt {
                                    #[cfg(feature = "dtype-categorical")]
                                    DataType::Enum(..) | DataType::Categorical(..) => {
                                        needs_cast = true;
                                        DataType::String
                                    },
                                    leaf_dt => leaf_dt,
                                }
                            }),
                        )
                    })
                    .collect();

                let arrow_dtype =
                    DataType::Struct(deserialize_schema).to_arrow(CompatLevel::newest());

                let arrow_dtype = if let BorrowedValue::Array(_) = &json_value {
                    ArrowDataType::LargeList(Box::new(arrow::datatypes::Field::new(
                        LIST_VALUES_NAME,
                        arrow_dtype,
                        true,
                    )))
                } else {
                    arrow_dtype
                };

                let arr = polars_json::json::deserialize(
                    &json_value,
                    arrow_dtype,
                    allow_extra_fields_in_struct,
                )?;

                let arr = arr.as_any().downcast_ref::<StructArray>().ok_or_else(
                    || polars_err!(ComputeError: "can only deserialize json objects"),
                )?;

                let mut df = DataFrame::try_from(arr.clone())?;

                if df.width() == 0 && df.height() <= 1 {
                    // read_json("{}")
                    unsafe { df.set_height(0) };
                }

                if needs_cast {
                    for (col, dt) in unsafe { df.columns_mut() }
                        .iter_mut()
                        .zip(schema.iter_values())
                    {
                        *col = col.cast_with_options(
                            dt,
                            if self.ignore_errors {
                                CastOptions::NonStrict
                            } else {
                                CastOptions::Strict
                            },
                        )?;
                    }
                }

                df
            },
            JsonFormat::JsonLines => {
                let mut json_reader = CoreJsonReader::new(
                    rb,
                    None,
                    self.schema,
                    self.schema_overwrite,
                    None,
                    1024, // sample size
                    NonZeroUsize::new(1 << 18).unwrap(),
                    false,
                    self.infer_schema_len,
                    self.ignore_errors,
                    None,
                    None,
                    None,
                )?;
                let mut df: DataFrame = json_reader.as_df()?;
                if self.rechunk {
                    df.rechunk_mut_par();
                }

                df
            },
        };

        // TODO! Ensure we don't materialize the columns we don't need
        if let Some(proj) = self.projection.as_deref() {
            out.select(proj.iter().cloned())
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
    pub fn infer_schema_len(mut self, max_records: Option<NonZeroUsize>) -> Self {
        self.infer_schema_len = max_records;
        self
    }

    /// Set the batch size (number of records to load at one time)
    ///
    /// This heavily influences loading time.
    pub fn with_batch_size(mut self, batch_size: NonZeroUsize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the reader's column projection: the names of the columns to keep after deserialization. If `None`, all
    /// columns are kept.
    ///
    /// Setting `projection` to the columns you want to keep is more efficient than deserializing all of the columns and
    /// then dropping the ones you don't want.
    pub fn with_projection(mut self, projection: Option<Vec<PlSmallStr>>) -> Self {
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
