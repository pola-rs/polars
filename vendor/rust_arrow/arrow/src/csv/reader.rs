// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! CSV Reader
//!
//! This CSV reader allows CSV files to be read into the Arrow memory model. Records are
//! loaded in batches and are then converted from row-based data to columnar data.
//!
//! Example:
//!
//! ```
//! use some::csv;
//! use some::datatypes::{DataType, Field, Schema};
//! use std::fs::File;
//! use std::sync::Arc;
//!
//! let schema = Schema::new(vec![
//!     Field::new("city", DataType::Utf8, false),
//!     Field::new("lat", DataType::Float64, false),
//!     Field::new("lng", DataType::Float64, false),
//! ]);
//!
//! let file = File::open("test/data/uk_cities.csv").unwrap();
//!
//! let mut csv = csv::Reader::new(file, Arc::new(schema), false, None, 1024, None);
//! let batch = csv.next().unwrap().unwrap();
//! ```

use lazy_static::lazy_static;
use regex::{Regex, RegexBuilder};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::sync::Arc;

use csv as csv_crate;

use crate::array::{ArrayRef, PrimitiveBuilder, StringBuilder};
use crate::datatypes::*;
use crate::error::{ArrowError, Result};
use crate::record_batch::RecordBatch;

use self::csv_crate::{StringRecord, StringRecordsIntoIter};

lazy_static! {
    static ref DECIMAL_RE: Regex = Regex::new(r"^-?(\d+\.\d+)$").unwrap();
    static ref INTEGER_RE: Regex = Regex::new(r"^-?(\d+)$").unwrap();
    static ref BOOLEAN_RE: Regex = RegexBuilder::new(r"^(true)$|^(false)$")
        .case_insensitive(true)
        .build()
        .unwrap();
}

/// Infer the data type of a record
fn infer_field_schema(string: &str) -> DataType {
    // when quoting is enabled in the reader, these quotes aren't escaped, we default to
    // Utf8 for them
    if string.starts_with('"') {
        return DataType::Utf8;
    }
    // match regex in a particular order
    if BOOLEAN_RE.is_match(string) {
        DataType::Boolean
    } else if DECIMAL_RE.is_match(string) {
        DataType::Float64
    } else if INTEGER_RE.is_match(string) {
        DataType::Int64
    } else {
        DataType::Utf8
    }
}

/// Infer the schema of a CSV file by reading through the first n records of the file,
/// with `max_read_records` controlling the maximum number of records to read.
///
/// If `max_read_records` is not set, the whole file is read to infer its schema.
///
/// Return infered schema and number of records used for inference.
fn infer_file_schema<R: Read + Seek>(
    reader: &mut BufReader<R>,
    delimiter: u8,
    max_read_records: Option<usize>,
    has_header: bool,
) -> Result<(Schema, usize)> {
    let mut csv_reader = csv_crate::ReaderBuilder::new()
        .delimiter(delimiter)
        .from_reader(reader);

    // get or create header names
    // when has_header is false, creates default column names with column_ prefix
    let headers: Vec<String> = if has_header {
        let headers = &csv_reader.headers()?.clone();
        headers.iter().map(|s| s.to_string()).collect()
    } else {
        let first_record_count = &csv_reader.headers()?.len();
        (0..*first_record_count)
            .map(|i| format!("column_{}", i + 1))
            .collect()
    };

    // save the csv reader position after reading headers
    let position = csv_reader.position().clone();

    let header_length = headers.len();
    // keep track of inferred field types
    let mut column_types: Vec<HashSet<DataType>> = vec![HashSet::new(); header_length];
    // keep track of columns with nulls
    let mut nulls: Vec<bool> = vec![false; header_length];

    // return csv reader position to after headers
    csv_reader.seek(position)?;

    let mut records_count = 0;
    let mut fields = vec![];

    for result in csv_reader
        .records()
        .take(max_read_records.unwrap_or(std::usize::MAX))
    {
        let record = result?;
        records_count += 1;

        for i in 0..header_length {
            if let Some(string) = record.get(i) {
                if string == "" {
                    nulls[i] = true;
                } else {
                    column_types[i].insert(infer_field_schema(string));
                }
            }
        }
    }

    // build schema from inference results
    for i in 0..header_length {
        let possibilities = &column_types[i];
        let has_nulls = nulls[i];
        let field_name = &headers[i];

        // determine data type based on possible types
        // if there are incompatible types, use DataType::Utf8
        match possibilities.len() {
            1 => {
                for dtype in possibilities.iter() {
                    fields.push(Field::new(&field_name, dtype.clone(), has_nulls));
                }
            }
            2 => {
                if possibilities.contains(&DataType::Int64)
                    && possibilities.contains(&DataType::Float64)
                {
                    // we have an integer and double, fall down to double
                    fields.push(Field::new(&field_name, DataType::Float64, has_nulls));
                } else {
                    // default to Utf8 for conflicting datatypes (e.g bool and int)
                    fields.push(Field::new(&field_name, DataType::Utf8, has_nulls));
                }
            }
            _ => fields.push(Field::new(&field_name, DataType::Utf8, has_nulls)),
        }
    }

    // return the reader seek back to the start
    csv_reader.into_inner().seek(SeekFrom::Start(0))?;

    Ok((Schema::new(fields), records_count))
}

/// Infer schema from a list of CSV files by reading through first n records
/// with `max_read_records` controlling the maximum number of records to read.
///
/// Files will be read in the given order untill n records have been reached.
///
/// If `max_read_records` is not set, all files will be read fully to infer the schema.
pub fn infer_schema_from_files(
    files: &Vec<String>,
    delimiter: u8,
    max_read_records: Option<usize>,
    has_header: bool,
) -> Result<Schema> {
    let mut schemas = vec![];
    let mut records_to_read = max_read_records.unwrap_or(std::usize::MAX);

    for fname in files.iter() {
        let (schema, records_read) = infer_file_schema(
            &mut BufReader::new(File::open(fname)?),
            delimiter,
            Some(records_to_read),
            has_header,
        )?;
        if records_read == 0 {
            continue;
        }
        schemas.push(schema.clone());
        records_to_read -= records_read;
        if records_to_read <= 0 {
            break;
        }
    }

    Schema::try_merge(&schemas)
}

/// CSV file reader
pub struct Reader<R: Read> {
    /// Explicit schema for the CSV file
    schema: Arc<Schema>,
    /// Optional projection for which columns to load (zero-based column indices)
    projection: Option<Vec<usize>>,
    /// File reader
    record_iter: StringRecordsIntoIter<BufReader<R>>,
    /// Batch size (number of records to load each time)
    batch_size: usize,
    /// Current line number, used in error reporting
    line_number: usize,
}

impl<R: Read> Reader<R> {
    /// Create a new CsvReader from any value that implements the `Read` trait.
    ///
    /// If reading a `File` or an input that supports `std::io::Read` and `std::io::Seek`;
    /// you can customise the Reader, such as to enable schema inference, use
    /// `ReaderBuilder`.
    pub fn new(
        reader: R,
        schema: Arc<Schema>,
        has_header: bool,
        delimiter: Option<u8>,
        batch_size: usize,
        projection: Option<Vec<usize>>,
    ) -> Self {
        Self::from_buf_reader(
            BufReader::new(reader),
            schema,
            has_header,
            delimiter,
            batch_size,
            projection,
        )
    }

    /// Returns the schema of the reader, useful for getting the schema without reading
    /// record batches
    pub fn schema(&self) -> Arc<Schema> {
        match &self.projection {
            Some(projection) => {
                let fields = self.schema.fields();
                let projected_fields: Vec<Field> =
                    projection.iter().map(|i| fields[*i].clone()).collect();

                Arc::new(Schema::new(projected_fields))
            }
            None => self.schema.clone(),
        }
    }

    /// Create a new CsvReader from a `BufReader<R: Read>
    ///
    /// This constructor allows you more flexibility in what records are processed by the
    /// csv reader.
    pub fn from_buf_reader(
        buf_reader: BufReader<R>,
        schema: Arc<Schema>,
        has_header: bool,
        delimiter: Option<u8>,
        batch_size: usize,
        projection: Option<Vec<usize>>,
    ) -> Self {
        let mut reader_builder = csv_crate::ReaderBuilder::new();
        reader_builder.has_headers(has_header);

        match delimiter {
            Some(c) => {
                reader_builder.delimiter(c);
            }
            _ => (),
        }

        let csv_reader = reader_builder.from_reader(buf_reader);
        let record_iter = csv_reader.into_records();
        Self {
            schema,
            projection,
            record_iter,
            batch_size,
            line_number: if has_header { 1 } else { 0 },
        }
    }

    /// Read the next batch of rows
    pub fn next(&mut self) -> Result<Option<RecordBatch>> {
        // read a batch of rows into memory
        let mut rows: Vec<StringRecord> = Vec::with_capacity(self.batch_size);
        for i in 0..self.batch_size {
            match self.record_iter.next() {
                Some(Ok(r)) => {
                    rows.push(r);
                }
                Some(Err(e)) => {
                    return Err(ArrowError::ParseError(format!(
                        "Error parsing line {}: {:?}",
                        self.line_number + i,
                        e
                    )));
                }
                None => break,
            }
        }

        // return early if no data was loaded
        if rows.is_empty() {
            return Ok(None);
        }

        let projection: Vec<usize> = match self.projection {
            Some(ref v) => v.clone(),
            None => self
                .schema
                .fields()
                .iter()
                .enumerate()
                .map(|(i, _)| i)
                .collect(),
        };

        let rows = &rows[..];
        let arrays: Result<Vec<ArrayRef>> = projection
            .iter()
            .map(|i| {
                let i = *i;
                let field = self.schema.field(i);
                match field.data_type() {
                    &DataType::Boolean => {
                        self.build_primitive_array::<BooleanType>(rows, i)
                    }
                    &DataType::Int8 => self.build_primitive_array::<Int8Type>(rows, i),
                    &DataType::Int16 => self.build_primitive_array::<Int16Type>(rows, i),
                    &DataType::Int32 => self.build_primitive_array::<Int32Type>(rows, i),
                    &DataType::Int64 => self.build_primitive_array::<Int64Type>(rows, i),
                    &DataType::UInt8 => self.build_primitive_array::<UInt8Type>(rows, i),
                    &DataType::UInt16 => {
                        self.build_primitive_array::<UInt16Type>(rows, i)
                    }
                    &DataType::UInt32 => {
                        self.build_primitive_array::<UInt32Type>(rows, i)
                    }
                    &DataType::UInt64 => {
                        self.build_primitive_array::<UInt64Type>(rows, i)
                    }
                    &DataType::Float32 => {
                        self.build_primitive_array::<Float32Type>(rows, i)
                    }
                    &DataType::Float64 => {
                        self.build_primitive_array::<Float64Type>(rows, i)
                    }
                    &DataType::Utf8 => {
                        let mut builder = StringBuilder::new(rows.len());
                        for row in rows.iter() {
                            match row.get(i) {
                                Some(s) => builder.append_value(s).unwrap(),
                                _ => builder.append(false).unwrap(),
                            }
                        }
                        Ok(Arc::new(builder.finish()) as ArrayRef)
                    }
                    other => Err(ArrowError::ParseError(format!(
                        "Unsupported data type {:?}",
                        other
                    ))),
                }
            })
            .collect();

        self.line_number += rows.len();

        let schema_fields = self.schema.fields();

        let projected_fields: Vec<Field> = projection
            .iter()
            .map(|i| schema_fields[*i].clone())
            .collect();

        let projected_schema = Arc::new(Schema::new(projected_fields));

        arrays.and_then(|arr| RecordBatch::try_new(projected_schema, arr).map(Some))
    }

    fn build_primitive_array<T: ArrowPrimitiveType>(
        &self,
        rows: &[StringRecord],
        col_idx: usize,
    ) -> Result<ArrayRef> {
        let mut builder = PrimitiveBuilder::<T>::new(rows.len());
        let is_boolean_type =
            *self.schema.field(col_idx).data_type() == DataType::Boolean;
        for (row_index, row) in rows.iter().enumerate() {
            match row.get(col_idx) {
                Some(s) if !s.is_empty() => {
                    let t = if is_boolean_type {
                        s.to_lowercase().parse::<T::Native>()
                    } else {
                        s.parse::<T::Native>()
                    };
                    match t {
                        Ok(v) => builder.append_value(v)?,
                        Err(_) => {
                            // TODO: we should surface the underlying error here.
                            return Err(ArrowError::ParseError(format!(
                                "Error while parsing value {} for column {} at line {}",
                                s,
                                col_idx,
                                self.line_number + row_index
                            )));
                        }
                    }
                }
                _ => builder.append_null()?,
            }
        }
        Ok(Arc::new(builder.finish()))
    }
}

/// CSV file reader builder
pub struct ReaderBuilder {
    /// Optional schema for the CSV file
    ///
    /// If the schema is not supplied, the reader will try to infer the schema
    /// based on the CSV structure.
    schema: Option<Arc<Schema>>,
    /// Whether the file has headers or not
    ///
    /// If schema inference is run on a file with no headers, default column names
    /// are created.
    has_header: bool,
    /// An optional column delimiter. Defaults to `b','`
    delimiter: Option<u8>,
    /// Optional maximum number of records to read during schema inference
    ///
    /// If a number is not provided, all the records are read.
    max_records: Option<usize>,
    /// Batch size (number of records to load each time)
    ///
    /// The default batch size when using the `ReaderBuilder` is 1024 records
    batch_size: usize,
    /// Optional projection for which columns to load (zero-based column indices)
    projection: Option<Vec<usize>>,
}

impl Default for ReaderBuilder {
    fn default() -> Self {
        Self {
            schema: None,
            has_header: false,
            delimiter: None,
            max_records: None,
            batch_size: 1024,
            projection: None,
        }
    }
}

impl ReaderBuilder {
    /// Create a new builder for configuring CSV parsing options.
    ///
    /// To convert a builder into a reader, call `ReaderBuilder::build`
    ///
    /// # Example
    ///
    /// ```
    /// extern crate some;
    ///
    /// use some::csv;
    /// use std::fs::File;
    ///
    /// fn example() -> csv::Reader<File> {
    ///     let file = File::open("test/data/uk_cities_with_headers.csv").unwrap();
    ///
    ///     // create a builder, inferring the schema with the first 100 records
    ///     let builder = csv::ReaderBuilder::new().infer_schema(Some(100));
    ///
    ///     let reader = builder.build(file).unwrap();
    ///
    ///     reader
    /// }
    /// ```
    pub fn new() -> ReaderBuilder {
        ReaderBuilder::default()
    }

    /// Set the CSV file's schema
    pub fn with_schema(mut self, schema: Arc<Schema>) -> Self {
        self.schema = Some(schema);
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

    /// Set the CSV reader to infer the schema of the file
    pub fn infer_schema(mut self, max_records: Option<usize>) -> Self {
        // remove any schema that is set
        self.schema = None;
        self.max_records = max_records;
        self
    }

    /// Set the batch size (number of records to load at one time)
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the reader's column projection
    pub fn with_projection(mut self, projection: Vec<usize>) -> Self {
        self.projection = Some(projection);
        self
    }

    /// Create a new `Reader` from the `ReaderBuilder`
    pub fn build<R: Read + Seek>(self, reader: R) -> Result<Reader<R>> {
        // check if schema should be inferred
        let mut buf_reader = BufReader::new(reader);
        let delimiter = self.delimiter.unwrap_or(b',');
        let schema = match self.schema {
            Some(schema) => schema,
            None => {
                let (inferred_schema, _) = infer_file_schema(
                    &mut buf_reader,
                    delimiter,
                    self.max_records,
                    self.has_header,
                )?;

                Arc::new(inferred_schema)
            }
        };
        let csv_reader = csv_crate::ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(self.has_header)
            .from_reader(buf_reader);
        let record_iter = csv_reader.into_records();
        Ok(Reader {
            schema,
            projection: self.projection.clone(),
            record_iter,
            batch_size: self.batch_size,
            line_number: if self.has_header { 1 } else { 0 },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::File;
    use std::io::{Cursor, Write};
    use tempfile::NamedTempFile;

    use crate::array::*;
    use crate::datatypes::Field;

    #[test]
    fn test_csv() {
        let schema = Schema::new(vec![
            Field::new("city", DataType::Utf8, false),
            Field::new("lat", DataType::Float64, false),
            Field::new("lng", DataType::Float64, false),
        ]);

        let file = File::open("test/data/uk_cities.csv").unwrap();

        let mut csv =
            Reader::new(file, Arc::new(schema.clone()), false, None, 1024, None);
        assert_eq!(Arc::new(schema), csv.schema());
        let batch = csv.next().unwrap().unwrap();
        assert_eq!(37, batch.num_rows());
        assert_eq!(3, batch.num_columns());

        // access data from a primitive array
        let lat = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(57.653484, lat.value(0));

        // access data from a string array (ListArray<u8>)
        let city = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        assert_eq!("Aberdeen, Aberdeen City, UK", city.value(13));
    }

    #[test]
    fn test_csv_from_buf_reader() {
        let schema = Schema::new(vec![
            Field::new("city", DataType::Utf8, false),
            Field::new("lat", DataType::Float64, false),
            Field::new("lng", DataType::Float64, false),
        ]);

        let file_with_headers =
            File::open("test/data/uk_cities_with_headers.csv").unwrap();
        let file_without_headers = File::open("test/data/uk_cities.csv").unwrap();
        let both_files = file_with_headers
            .chain(Cursor::new("\n".to_string()))
            .chain(file_without_headers);
        let mut csv = Reader::from_buf_reader(
            BufReader::new(both_files),
            Arc::new(schema),
            true,
            None,
            1024,
            None,
        );
        let batch = csv.next().unwrap().unwrap();
        assert_eq!(74, batch.num_rows());
        assert_eq!(3, batch.num_columns());
    }

    #[test]
    fn test_csv_with_schema_inference() {
        let file = File::open("test/data/uk_cities_with_headers.csv").unwrap();

        let builder = ReaderBuilder::new().has_header(true).infer_schema(None);

        let mut csv = builder.build(file).unwrap();
        let expected_schema = Schema::new(vec![
            Field::new("city", DataType::Utf8, false),
            Field::new("lat", DataType::Float64, false),
            Field::new("lng", DataType::Float64, false),
        ]);
        assert_eq!(Arc::new(expected_schema), csv.schema());
        let batch = csv.next().unwrap().unwrap();
        assert_eq!(37, batch.num_rows());
        assert_eq!(3, batch.num_columns());

        // access data from a primitive array
        let lat = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(57.653484, lat.value(0));

        // access data from a string array (ListArray<u8>)
        let city = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        assert_eq!("Aberdeen, Aberdeen City, UK", city.value(13));
    }

    #[test]
    fn test_csv_with_schema_inference_no_headers() {
        let file = File::open("test/data/uk_cities.csv").unwrap();

        let builder = ReaderBuilder::new().infer_schema(None);

        let mut csv = builder.build(file).unwrap();

        // csv field names should be 'column_{number}'
        let schema = csv.schema();
        assert_eq!("column_1", schema.field(0).name());
        assert_eq!("column_2", schema.field(1).name());
        assert_eq!("column_3", schema.field(2).name());
        let batch = csv.next().unwrap().unwrap();
        let batch_schema = batch.schema();

        assert_eq!(&schema, batch_schema);
        assert_eq!(37, batch.num_rows());
        assert_eq!(3, batch.num_columns());

        // access data from a primitive array
        let lat = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(57.653484, lat.value(0));

        // access data from a string array (ListArray<u8>)
        let city = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        assert_eq!("Aberdeen, Aberdeen City, UK", city.value(13));
    }

    #[test]
    fn test_csv_with_projection() {
        let schema = Schema::new(vec![
            Field::new("city", DataType::Utf8, false),
            Field::new("lat", DataType::Float64, false),
            Field::new("lng", DataType::Float64, false),
        ]);

        let file = File::open("test/data/uk_cities.csv").unwrap();

        let mut csv =
            Reader::new(file, Arc::new(schema), false, None, 1024, Some(vec![0, 1]));
        let projected_schema = Arc::new(Schema::new(vec![
            Field::new("city", DataType::Utf8, false),
            Field::new("lat", DataType::Float64, false),
        ]));
        assert_eq!(projected_schema.clone(), csv.schema());
        let batch = csv.next().unwrap().unwrap();
        assert_eq!(&projected_schema, batch.schema());
        assert_eq!(37, batch.num_rows());
        assert_eq!(2, batch.num_columns());
    }

    #[test]
    fn test_nulls() {
        let schema = Schema::new(vec![
            Field::new("c_int", DataType::UInt64, false),
            Field::new("c_float", DataType::Float32, false),
            Field::new("c_string", DataType::Utf8, false),
        ]);

        let file = File::open("test/data/null_test.csv").unwrap();

        let mut csv = Reader::new(file, Arc::new(schema), true, None, 1024, None);
        let batch = csv.next().unwrap().unwrap();

        assert_eq!(false, batch.column(1).is_null(0));
        assert_eq!(false, batch.column(1).is_null(1));
        assert_eq!(true, batch.column(1).is_null(2));
        assert_eq!(false, batch.column(1).is_null(3));
        assert_eq!(false, batch.column(1).is_null(4));
    }

    #[test]
    fn test_nulls_with_inference() {
        let file = File::open("test/data/various_types.csv").unwrap();

        let builder = ReaderBuilder::new()
            .infer_schema(None)
            .has_header(true)
            .with_delimiter(b'|')
            .with_batch_size(512)
            .with_projection(vec![0, 1, 2, 3]);

        let mut csv = builder.build(file).unwrap();
        let batch = csv.next().unwrap().unwrap();

        assert_eq!(5, batch.num_rows());
        assert_eq!(4, batch.num_columns());

        let schema = batch.schema();

        assert_eq!(&DataType::Int64, schema.field(0).data_type());
        assert_eq!(&DataType::Float64, schema.field(1).data_type());
        assert_eq!(&DataType::Float64, schema.field(2).data_type());
        assert_eq!(&DataType::Boolean, schema.field(3).data_type());

        assert_eq!(false, schema.field(0).is_nullable());
        assert_eq!(true, schema.field(1).is_nullable());
        assert_eq!(true, schema.field(2).is_nullable());
        assert_eq!(false, schema.field(3).is_nullable());

        assert_eq!(false, batch.column(1).is_null(0));
        assert_eq!(false, batch.column(1).is_null(1));
        assert_eq!(true, batch.column(1).is_null(2));
        assert_eq!(false, batch.column(1).is_null(3));
        assert_eq!(false, batch.column(1).is_null(4));
    }

    #[test]
    fn test_parse_invalid_csv() {
        let file = File::open("test/data/various_types_invalid.csv").unwrap();

        let schema = Schema::new(vec![
            Field::new("c_int", DataType::UInt64, false),
            Field::new("c_float", DataType::Float32, false),
            Field::new("c_string", DataType::Utf8, false),
            Field::new("c_bool", DataType::Boolean, false),
        ]);

        let builder = ReaderBuilder::new()
            .with_schema(Arc::new(schema))
            .has_header(true)
            .with_delimiter(b'|')
            .with_batch_size(512)
            .with_projection(vec![0, 1, 2, 3]);

        let mut csv = builder.build(file).unwrap();
        match csv.next() {
            Err(e) => assert_eq!(
                "ParseError(\"Error while parsing value 4.x4 for column 1 at line 4\")",
                format!("{:?}", e)
            ),
            Ok(_) => panic!("should have failed"),
        }
    }

    #[test]
    fn test_infer_field_schema() {
        assert_eq!(infer_field_schema("A"), DataType::Utf8);
        assert_eq!(infer_field_schema("\"123\""), DataType::Utf8);
        assert_eq!(infer_field_schema("10"), DataType::Int64);
        assert_eq!(infer_field_schema("10.2"), DataType::Float64);
        assert_eq!(infer_field_schema("true"), DataType::Boolean);
        assert_eq!(infer_field_schema("false"), DataType::Boolean);
    }

    #[test]
    fn test_infer_schema_from_multiple_files() -> Result<()> {
        let mut csv1 = NamedTempFile::new()?;
        let mut csv2 = NamedTempFile::new()?;
        let csv3 = NamedTempFile::new()?; // empty csv file should be skipped
        let mut csv4 = NamedTempFile::new()?;
        writeln!(csv1, "c1,c2,c3")?;
        writeln!(csv1, "1,\"foo\",0.5")?;
        writeln!(csv1, "3,\"bar\",1")?;
        // reading csv2 will set c2 to optional
        writeln!(csv2, "c1,c2,c3,c4")?;
        writeln!(csv2, "10,,3.14,true")?;
        // reading csv4 will set c3 to optional
        writeln!(csv4, "c1,c2,c3")?;
        writeln!(csv4, "10,\"foo\",")?;

        let schema = infer_schema_from_files(
            &vec![
                csv3.path().to_str().unwrap().to_string(),
                csv1.path().to_str().unwrap().to_string(),
                csv2.path().to_str().unwrap().to_string(),
                csv4.path().to_str().unwrap().to_string(),
            ],
            b',',
            Some(3), // only csv1 and csv2 should be read
            true,
        )?;

        assert_eq!(schema.fields().len(), 4);
        assert_eq!(false, schema.field(0).is_nullable());
        assert_eq!(true, schema.field(1).is_nullable());
        assert_eq!(false, schema.field(2).is_nullable());
        assert_eq!(false, schema.field(3).is_nullable());

        assert_eq!(&DataType::Int64, schema.field(0).data_type());
        assert_eq!(&DataType::Utf8, schema.field(1).data_type());
        assert_eq!(&DataType::Float64, schema.field(2).data_type());
        assert_eq!(&DataType::Boolean, schema.field(3).data_type());

        Ok(())
    }
}
