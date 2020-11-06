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

use crate::prelude::*;
use arrow::datatypes::SchemaRef;
use arrow::error::Result as ArrowResult;
use csv::{StringRecord, StringRecordsIntoIter};
use lazy_static::lazy_static;
use rayon::prelude::*;
use regex::{Regex, RegexBuilder};
use seahash::SeaHasher;
use std::collections::HashSet;
use std::fmt;
use std::hash::BuildHasherDefault;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::sync::Arc;

lazy_static! {
    static ref DECIMAL_RE: Regex = Regex::new(r"^-?(\d+\.\d+)$").unwrap();
    static ref INTEGER_RE: Regex = Regex::new(r"^-?(\d+)$").unwrap();
    static ref BOOLEAN_RE: Regex = RegexBuilder::new(r"^(true)$|^(false)$")
        .case_insensitive(true)
        .build()
        .unwrap();
}

/// Infer the data type of a record
fn infer_field_schema(string: &str) -> ArrowDataType {
    // when quoting is enabled in the reader, these quotes aren't escaped, we default to
    // Utf8 for them
    if string.starts_with('"') {
        return ArrowDataType::Utf8;
    }
    // match regex in a particular order
    if BOOLEAN_RE.is_match(string) {
        ArrowDataType::Boolean
    } else if DECIMAL_RE.is_match(string) {
        ArrowDataType::Float64
    } else if INTEGER_RE.is_match(string) {
        ArrowDataType::Int64
    } else {
        ArrowDataType::Utf8
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
) -> ArrowResult<(Schema, usize)> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .from_reader(reader);

    // get or create header names
    // when has_header is false, creates default column names with column_ prefix
    let headers: Vec<String> = if has_header {
        let headers = csv_reader.headers()?;
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
    let mut column_types: Vec<HashSet<ArrowDataType, BuildHasherDefault<SeaHasher>>> =
        vec![HashSet::with_hasher(BuildHasherDefault::default()); header_length];
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
                if possibilities.contains(&ArrowDataType::Int64)
                    && possibilities.contains(&ArrowDataType::Float64)
                {
                    // we have an integer and double, fall down to double
                    fields.push(Field::new(&field_name, ArrowDataType::Float64, has_nulls));
                } else {
                    // default to Utf8 for conflicting datatypes (e.g bool and int)
                    fields.push(Field::new(&field_name, ArrowDataType::Utf8, has_nulls));
                }
            }
            _ => fields.push(Field::new(&field_name, ArrowDataType::Utf8, has_nulls)),
        }
    }

    // return the reader seek back to the start
    csv_reader.into_inner().seek(SeekFrom::Start(0))?;

    Ok((Schema::new(fields), records_count))
}

/// CSV file reader
pub struct Reader<R: Read> {
    /// Explicit schema for the CSV file
    schema: SchemaRef,
    /// Optional projection for which columns to load (zero-based column indices)
    projection: Option<Vec<usize>>,
    /// File reader
    record_iter: StringRecordsIntoIter<BufReader<R>>,
    /// Batch size (number of records to load each time)
    batch_size: usize,
    /// Current line number, used in error reporting
    line_number: usize,
    ignore_parser_errors: bool,
    header_offset: usize,
}

impl<R> fmt::Debug for Reader<R>
where
    R: Read,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reader")
            .field("schema", &self.schema)
            .field("projection", &self.projection)
            .field("batch_size", &self.batch_size)
            .field("line_number", &self.line_number)
            .finish()
    }
}

impl<R: Read + Sync> Reader<R> {
    /// Create a new CsvReader from any value that implements the `Read` trait.
    ///
    /// If reading a `File` or an input that supports `std::io::Read` and `std::io::Seek`;
    /// you can customise the Reader, such as to enable schema inference, use
    /// `ReaderBuilder`.
    pub fn new(
        reader: R,
        schema: SchemaRef,
        has_header: bool,
        delimiter: Option<u8>,
        batch_size: usize,
        projection: Option<Vec<usize>>,
        ignore_parser_errors: bool,
    ) -> Self {
        Self::from_buf_reader(
            BufReader::new(reader),
            schema,
            has_header,
            delimiter,
            batch_size,
            projection,
            ignore_parser_errors,
        )
    }

    /// Returns the schema of the reader, useful for getting the schema without reading
    /// record batches
    pub fn schema(&self) -> SchemaRef {
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
        schema: SchemaRef,
        has_header: bool,
        delimiter: Option<u8>,
        batch_size: usize,
        projection: Option<Vec<usize>>,
        ignore_parser_errors: bool,
    ) -> Self {
        let mut reader_builder = csv::ReaderBuilder::new();
        reader_builder.has_headers(has_header);

        if let Some(c) = delimiter {
            reader_builder.delimiter(c);
        }

        let csv_reader = reader_builder.from_reader(buf_reader);
        let record_iter = csv_reader.into_records();

        let header_offset = if has_header { 1 } else { 0 };

        Self {
            schema,
            projection,
            record_iter,
            batch_size,
            line_number: if has_header { 1 } else { 0 },
            ignore_parser_errors,
            header_offset,
        }
    }

    fn next_rows(&mut self, rows: &mut Vec<StringRecord>) -> Result<()> {
        for i in 0..self.batch_size {
            self.line_number += 1;
            match self.record_iter.next() {
                Some(Ok(r)) => {
                    rows.push(r);
                }
                Some(Err(e)) => {
                    if self.ignore_parser_errors {
                        continue;
                    } else {
                        return Err(PolarsError::Other(
                            format!("Error parsing line {}: {:?}", self.line_number + i, e).into(),
                        ));
                    }
                }
                None => break,
            }
        }
        Ok(())
    }

    fn field_to_builder(&self, i: usize, capacity: usize) -> Result<Builder> {
        let field = self.schema.field(i);
        let name = field.name();

        let builder = match field.data_type() {
            &ArrowDataType::Boolean => {
                Builder::Boolean(PrimitiveChunkedBuilder::new(name, capacity))
            }
            &ArrowDataType::Int32 => Builder::Int32(PrimitiveChunkedBuilder::new(name, capacity)),
            &ArrowDataType::Int64 => Builder::Int64(PrimitiveChunkedBuilder::new(name, capacity)),
            &ArrowDataType::Float32 => {
                Builder::Float32(PrimitiveChunkedBuilder::new(name, capacity))
            }
            &ArrowDataType::Float64 => {
                Builder::Float64(PrimitiveChunkedBuilder::new(name, capacity))
            }
            &ArrowDataType::Utf8 => Builder::Utf8(Utf8ChunkedBuilder::new(name, capacity)),
            other => {
                return Err(PolarsError::Other(
                    format!("Unsupported data type {:?} when reading a csv", other).into(),
                ))
            }
        };
        Ok(builder)
    }

    fn add_to_primitive<T>(
        &self,
        rows: &[StringRecord],
        col_idx: usize,
        builder: &mut PrimitiveChunkedBuilder<T>,
    ) -> Result<()>
    where
        T: ArrowPrimitiveType,
    {
        let is_boolean_type = *self.schema.field(col_idx).data_type() == ArrowDataType::Boolean;

        if (rows.len() + builder.len()) > builder.capacity() {
            builder.reserve(builder.capacity() * 2)
        }

        for (row_index, row) in rows.iter().enumerate() {
            match row.get(col_idx) {
                Some(s) => {
                    if s.is_empty() {
                        builder.append_null();
                        continue;
                    }
                    let parsed = if is_boolean_type {
                        s.to_lowercase().parse::<T::Native>()
                    } else {
                        s.parse::<T::Native>()
                    };
                    match parsed {
                        Ok(e) => builder.append_value(e),
                        Err(_) => {
                            if self.ignore_parser_errors {
                                builder.append_null();
                                continue;
                            }
                            return Err(PolarsError::Other(
                                format!(
                                    // TODO: we should surface the underlying error here.
                                    "Error while parsing value {} for column {} at line {}",
                                    s,
                                    col_idx,
                                    self.line_number + row_index
                                )
                                .into(),
                            ));
                        }
                    }
                }
                None => builder.append_null(),
            }
        }
        Ok(())
    }

    fn add_to_utf8(
        &self,
        rows: &[StringRecord],
        col_idx: usize,
        builder: &mut Utf8ChunkedBuilder,
    ) -> Result<()> {
        for row in rows.iter() {
            let v = row.get(col_idx);
            builder.append_option(v);
        }
        Ok(())
    }

    fn init_builders(&self, projection: &Vec<usize>, capacity: usize) -> Result<Vec<Builder>> {
        projection
            .iter()
            .map(|&i| self.field_to_builder(i, capacity))
            .collect()
    }

    fn builders_to_df(&self, builders: Vec<Builder>) -> DataFrame {
        let columns = builders.into_iter().map(|b| b.into_series()).collect();
        DataFrame::new_no_checks(columns)
    }

    fn add_to_builders(
        &self,
        builders: &mut [Builder],
        projection: &[usize],
        rows: &[StringRecord],
    ) -> Result<()> {
        projection
            .par_iter()
            .zip(builders)
            .map(|(i, builder)| {
                let field = self.schema.field(*i);
                match field.data_type() {
                    ArrowDataType::Boolean => self.add_to_primitive(&rows, *i, builder.bool()),
                    ArrowDataType::Int32 => self.add_to_primitive(&rows, *i, builder.i32()),
                    ArrowDataType::Int64 => self.add_to_primitive(&rows, *i, builder.i64()),
                    ArrowDataType::Float32 => self.add_to_primitive(&rows, *i, builder.f32()),
                    ArrowDataType::Float64 => self.add_to_primitive(&rows, *i, builder.f64()),
                    ArrowDataType::Utf8 => self.add_to_utf8(&rows, *i, builder.utf8()),
                    _ => todo!(),
                }
            })
            .collect::<Result<_>>()?;

        Ok(())
    }

    pub fn into_df(
        mut self,
        capacity: usize,
        n_rows: Option<usize>,
        skip_rows: usize,
    ) -> Result<DataFrame> {
        let mut total_capacity = capacity;
        if skip_rows > 0 {
            for _ in 0..skip_rows {
                self.line_number += 1;
                let _ = self.record_iter.next();
            }
            total_capacity += skip_rows;
        }

        // only take projections once
        let projection = match self.projection.take() {
            Some(v) => v,
            None => self
                .schema
                .fields()
                .iter()
                .enumerate()
                .map(|(i, _)| i)
                .collect(),
        };
        self.batch_size = std::cmp::min(self.batch_size, capacity);
        if let Some(n) = n_rows {
            self.batch_size = std::cmp::min(self.batch_size, n);
        }

        let mut builders = self.init_builders(&projection, capacity)?;
        // we reuse this container to amortize allocations
        let mut rows = Vec::with_capacity(self.batch_size);
        let mut parsed_dfs = Vec::with_capacity(128);
        loop {
            rows.clear();
            self.next_rows(&mut rows)?;
            // stop when the whole file is processed
            if rows.len() == 0 {
                break;
            }
            if (self.line_number - self.header_offset) > total_capacity {
                let mut builders_tmp = self.init_builders(&projection, capacity)?;
                std::mem::swap(&mut builders_tmp, &mut builders);
                parsed_dfs.push(self.builders_to_df(builders_tmp));
                total_capacity += capacity;
            }

            self.add_to_builders(&mut builders, &projection, &rows)?;

            // stop after n_rows are processed
            if let Some(n_rows) = n_rows {
                if self.line_number >= n_rows {
                    break;
                }
            }
        }
        parsed_dfs.push(self.builders_to_df(builders));
        let mut iter = parsed_dfs.into_iter();
        let mut acc_df = iter.next().unwrap();
        while let Some(df) = iter.next() {
            acc_df.vstack(&df)?;
        }
        Ok(acc_df)
    }
}

enum Builder {
    Boolean(PrimitiveChunkedBuilder<BooleanType>),
    Int32(PrimitiveChunkedBuilder<Int32Type>),
    Int64(PrimitiveChunkedBuilder<Int64Type>),
    Float32(PrimitiveChunkedBuilder<Float32Type>),
    Float64(PrimitiveChunkedBuilder<Float64Type>),
    Utf8(Utf8ChunkedBuilder),
}

impl Builder {
    fn bool(&mut self) -> &mut PrimitiveChunkedBuilder<BooleanType> {
        match self {
            Builder::Boolean(builder) => builder,
            _ => panic!("implementation error"),
        }
    }
    fn i32(&mut self) -> &mut PrimitiveChunkedBuilder<Int32Type> {
        match self {
            Builder::Int32(builder) => builder,
            _ => panic!("implementation error"),
        }
    }
    fn i64(&mut self) -> &mut PrimitiveChunkedBuilder<Int64Type> {
        match self {
            Builder::Int64(builder) => builder,
            _ => panic!("implementation error"),
        }
    }
    fn f64(&mut self) -> &mut PrimitiveChunkedBuilder<Float64Type> {
        match self {
            Builder::Float64(builder) => builder,
            _ => panic!("implementation error"),
        }
    }
    fn f32(&mut self) -> &mut PrimitiveChunkedBuilder<Float32Type> {
        match self {
            Builder::Float32(builder) => builder,
            _ => panic!("implementation error"),
        }
    }
    fn utf8(&mut self) -> &mut Utf8ChunkedBuilder {
        match self {
            Builder::Utf8(builder) => builder,
            _ => panic!("implementation error"),
        }
    }

    fn into_series(self) -> Series {
        use Builder::*;
        match self {
            Utf8(b) => b.finish().into(),
            Int32(b) => b.finish().into(),
            Int64(b) => b.finish().into(),
            Float32(b) => b.finish().into(),
            Float64(b) => b.finish().into(),
            Boolean(b) => b.finish().into(),
        }
    }
}

/// CSV file reader builder
#[derive(Debug)]
pub struct ReaderBuilder {
    /// Optional schema for the CSV file
    ///
    /// If the schema is not supplied, the reader will try to infer the schema
    /// based on the CSV structure.
    schema: Option<SchemaRef>,
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
    ignore_errors: bool,
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
            ignore_errors: false,
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
    /// extern crate arrow;
    ///
    /// use arrow::csv;
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
    pub fn with_schema(mut self, schema: SchemaRef) -> Self {
        self.schema = Some(schema);
        self
    }

    pub fn with_ignore_parser_errors(mut self) -> Self {
        self.ignore_errors = true;
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
        let csv_reader = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(self.has_header)
            .from_reader(buf_reader);
        let record_iter = csv_reader.into_records();
        let header_offset = if self.has_header { 1 } else { 0 };
        Ok(Reader {
            schema,
            projection: self.projection.clone(),
            record_iter,
            batch_size: self.batch_size,
            line_number: if self.has_header { 1 } else { 0 },
            ignore_parser_errors: self.ignore_errors,
            header_offset,
        })
    }
}
