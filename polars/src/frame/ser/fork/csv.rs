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

use crate::frame::ser::csv::CsvEncoding;
use crate::prelude::*;
use crate::utils;
use ahash::RandomState;
use arrow::datatypes::SchemaRef;
use crossbeam::{
    channel::{bounded, TryRecvError},
    thread,
};
use csv::{ByteRecord, ByteRecordsIntoIter};
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

/// Is multiplied with batch_size to determine capacity of builders
const CAPACITY_MULTIPLIER: usize = 512;

fn all_digit(string: &str) -> bool {
    string.chars().all(|c| c.is_ascii_digit())
}
/// Infer the data type of a record
fn infer_field_schema(string: &str) -> ArrowDataType {
    // when quoting is enabled in the reader, these quotes aren't escaped, we default to
    // Utf8 for them
    if string.starts_with('"') {
        return ArrowDataType::Utf8;
    }
    // match regex in a particular order
    let lower = string.to_ascii_lowercase();
    if lower == "true" || lower == "false" {
        return ArrowDataType::Boolean;
    }
    let skip_minus = if string.starts_with('-') { 1 } else { 0 };

    let mut parts = string[skip_minus..].split('.');
    let (left, right) = (parts.next(), parts.next());
    let left_is_number = left.map_or(false, all_digit);
    if left_is_number && right.map_or(false, all_digit) {
        return ArrowDataType::Float64;
    } else if left_is_number {
        return ArrowDataType::Int64;
    }
    ArrowDataType::Utf8
}

fn parse_bytes_with_encoding(bytes: &[u8], encoding: CsvEncoding) -> Result<Cow<str>> {
    let s = match encoding {
        CsvEncoding::Utf8 => std::str::from_utf8(bytes)
            .map_err(anyhow::Error::from)?
            .into(),
        CsvEncoding::LossyUtf8 => String::from_utf8_lossy(bytes),
    };
    Ok(s)
}

/// Infer the schema of a CSV file by reading through the first n records of the file,
/// with `max_read_records` controlling the maximum number of records to read.
///
/// If `max_read_records` is not set, the whole file is read to infer its schema.
///
/// Return infered schema and number of records used for inference.
fn infer_file_schema<R: Read + Seek>(
    reader: &mut R,
    delimiter: u8,
    max_read_records: Option<usize>,
    has_header: bool,
) -> Result<(Schema, usize)> {
    // We use lossy utf8 here because we don't want the schema inference to fail on utf8.
    // It may later.
    let encoding = CsvEncoding::LossyUtf8;
    // set headers to false otherwise the csv crate, skips them.
    let csv_reader = init_csv_reader(reader, false, delimiter);

    let mut records = csv_reader.into_byte_records();
    let header_length;

    // get or create header names
    // when has_header is false, creates default column names with column_ prefix
    let headers: Vec<String> = if let Some(byterecord) = records.next() {
        let byterecord = byterecord.map_err(anyhow::Error::from)?;
        header_length = byterecord.len();
        if has_header {
            byterecord
                .iter()
                .map(|slice| {
                    let s = parse_bytes_with_encoding(slice, encoding)?;
                    Ok(s.into())
                })
                .collect::<Result<_>>()?
        } else {
            (0..header_length)
                .map(|i| format!("column_{}", i + 1))
                .collect()
        }
    } else {
        return Err(PolarsError::NoData("empty csv".into()));
    };

    // keep track of inferred field types
    let mut column_types: Vec<HashSet<ArrowDataType, RandomState>> =
        vec![HashSet::with_hasher(RandomState::new()); header_length];
    // keep track of columns with nulls
    let mut nulls: Vec<bool> = vec![false; header_length];

    let mut records_count = 0;
    let mut fields = Vec::with_capacity(header_length);

    // needed to prevent ownership going into the iterator loop
    let records_ref = &mut records;

    for result in records_ref.take(max_read_records.unwrap_or(std::usize::MAX)) {
        let record = result.map_err(anyhow::Error::from)?;
        records_count += 1;

        for i in 0..header_length {
            if let Some(slice) = record.get(i) {
                if slice.is_empty() {
                    nulls[i] = true;
                } else {
                    let s = parse_bytes_with_encoding(slice, encoding)?;
                    column_types[i].insert(infer_field_schema(&s));
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
    let csv_reader = records.into_reader();

    // return the reader seek back to the start
    csv_reader.into_inner().seek(SeekFrom::Start(0))?;

    Ok((Schema::new(fields), records_count))
}

fn init_csv_reader<R: Read>(reader: R, has_header: bool, delimiter: u8) -> csv::Reader<R> {
    let mut reader_builder = csv::ReaderBuilder::new();
    reader_builder.has_headers(has_header);
    reader_builder.delimiter(delimiter);
    reader_builder.from_reader(reader)
}

fn take_projection(projection: &mut Option<Vec<usize>>, schema: &SchemaRef) -> Vec<usize> {
    match projection.take() {
        Some(v) => v,
        None => schema.fields().iter().enumerate().map(|(i, _)| i).collect(),
    }
}

fn init_builders(
    projection: &[usize],
    capacity: usize,
    schema: &SchemaRef,
) -> Result<Vec<Builder>> {
    projection
        .iter()
        .map(|&i| field_to_builder(i, capacity, schema))
        .collect()
}

fn field_to_builder(i: usize, capacity: usize, schema: &SchemaRef) -> Result<Builder> {
    let field = schema.field(i);
    let name = field.name();

    let builder = match field.data_type() {
        &ArrowDataType::Boolean => Builder::Boolean(PrimitiveChunkedBuilder::new(name, capacity)),
        &ArrowDataType::Int16 => Builder::Int16(PrimitiveChunkedBuilder::new(name, capacity)),
        &ArrowDataType::Int32 => Builder::Int32(PrimitiveChunkedBuilder::new(name, capacity)),
        &ArrowDataType::Int64 => Builder::Int64(PrimitiveChunkedBuilder::new(name, capacity)),
        &ArrowDataType::UInt16 => Builder::UInt16(PrimitiveChunkedBuilder::new(name, capacity)),
        &ArrowDataType::UInt32 => Builder::UInt32(PrimitiveChunkedBuilder::new(name, capacity)),
        &ArrowDataType::UInt64 => Builder::UInt64(PrimitiveChunkedBuilder::new(name, capacity)),
        &ArrowDataType::Float32 => Builder::Float32(PrimitiveChunkedBuilder::new(name, capacity)),
        &ArrowDataType::Float64 => Builder::Float64(PrimitiveChunkedBuilder::new(name, capacity)),
        &ArrowDataType::Utf8 => Builder::Utf8(Utf8ChunkedBuilder::new(name, capacity)),
        other => {
            return Err(PolarsError::Other(
                format!("Unsupported data type {:?} when reading a csv", other).into(),
            ))
        }
    };
    Ok(builder)
}

fn next_rows<R: Read>(
    record_iter: &mut ByteRecordsIntoIter<R>,
    batch_size: usize,
    line_number: &mut usize,
    ignore_parser_errors: bool,
    n_rows: Option<usize>,
) -> Result<Vec<ByteRecord>> {
    let mut rows = Vec::with_capacity(batch_size);
    // if it is None, we set it larger than batch size such that it always evaluates false in
    // the conditional below
    let n_rows = n_rows.unwrap_or(*line_number + batch_size * 2);
    for i in 0..batch_size {
        *line_number += 1;

        if *line_number > n_rows {
            break;
        }

        match record_iter.next() {
            Some(Ok(r)) => {
                rows.push(r);
            }
            Some(Err(e)) => {
                if ignore_parser_errors {
                    continue;
                } else {
                    return Err(PolarsError::Other(
                        format!("Error parsing line {}: {:?}", *line_number + i, e).into(),
                    ));
                }
            }
            None => break,
        }
    }
    Ok(rows)
}

fn add_to_utf8_builder(
    rows: &[ByteRecord],
    col_idx: usize,
    builder: &mut Utf8ChunkedBuilder,
    encoding: CsvEncoding,
) -> Result<()> {
    for row in rows.iter() {
        let v = row.get(col_idx);
        match v {
            None => builder.append_null(),
            Some(bytes) => {
                let s = parse_bytes_with_encoding(bytes, encoding)?;
                builder.append_value(&s);
            }
        }
    }
    Ok(())
}

fn builders_to_df(builders: Vec<Builder>) -> DataFrame {
    let columns = builders.into_iter().map(|b| b.into_series()).collect();
    DataFrame::new_no_checks(columns)
}

/// CSV file reader
pub struct SequentialReader<R: Read> {
    /// Explicit schema for the CSV file
    schema: SchemaRef,
    /// Optional projection for which columns to load (zero-based column indices)
    projection: Option<Vec<usize>>,
    /// File reader
    record_iter: Option<ByteRecordsIntoIter<R>>,
    /// Batch size (number of records to load each time)
    batch_size: usize,
    /// Current line number, used in error reporting
    line_number: usize,
    ignore_parser_errors: bool,
    header_offset: usize,
    skip_rows: usize,
    n_rows: Option<usize>,
    encoding: CsvEncoding,
    one_thread: bool,
}

impl<R> fmt::Debug for SequentialReader<R>
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

impl<R: Read + Sync + Send> SequentialReader<R> {
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
    #[allow(clippy::too_many_arguments)]
    pub fn from_reader(
        reader: R,
        schema: SchemaRef,
        has_header: bool,
        delimiter: u8,
        batch_size: usize,
        projection: Option<Vec<usize>>,
        ignore_parser_errors: bool,
        n_rows: Option<usize>,
        skip_rows: usize,
        encoding: CsvEncoding,
        one_thread: bool,
    ) -> Self {
        let csv_reader = init_csv_reader(reader, has_header, delimiter);
        let record_iter = Some(csv_reader.into_byte_records());

        let header_offset = if has_header { 1 } else { 0 };

        Self {
            schema,
            projection,
            record_iter,
            batch_size,
            line_number: if has_header { 1 } else { 0 },
            ignore_parser_errors,
            header_offset,
            skip_rows,
            n_rows,
            encoding,
            one_thread,
        }
    }

    fn add_to_primitive<T>(
        &self,
        rows: &[ByteRecord],
        col_idx: usize,
        builder: &mut PrimitiveChunkedBuilder<T>,
    ) -> Result<()>
    where
        T: PolarsPrimitiveType,
    {
        let is_boolean_type = *self.schema.field(col_idx).data_type() == ArrowDataType::Boolean;

        for (row_index, row) in rows.iter().enumerate() {
            match row.get(col_idx) {
                Some(s) => {
                    if s.is_empty() {
                        builder.append_null();
                        continue;
                    }

                    let s = match std::str::from_utf8(s) {
                        Ok(s) => s,
                        Err(e) => {
                            if self.ignore_parser_errors {
                                builder.append_null();
                                continue;
                            }
                            return Err(PolarsError::Other(
                                format!(
                                    "Error while parsing value {} for column {} at line {}. Not valid UTF8",
                                    e,
                                    col_idx,
                                    self.line_number + row_index
                                )
                                    .into(),
                            ));
                        }
                    };

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

    fn add_to_builders(
        &self,
        builders: &mut [Builder],
        projection: &[usize],
        rows: &[ByteRecord],
    ) -> Result<()> {
        let dispatch = |(i, builder): (&usize, &mut Builder)| {
            let field = self.schema.field(*i);
            match field.data_type() {
                ArrowDataType::Boolean => self.add_to_primitive(rows, *i, builder.bool()),
                ArrowDataType::Int32 => self.add_to_primitive(rows, *i, builder.i32()),
                ArrowDataType::Int64 => self.add_to_primitive(rows, *i, builder.i64()),
                ArrowDataType::UInt64 => self.add_to_primitive(rows, *i, builder.u64()),
                ArrowDataType::UInt32 => self.add_to_primitive(rows, *i, builder.u32()),
                ArrowDataType::Float32 => self.add_to_primitive(rows, *i, builder.f32()),
                ArrowDataType::Float64 => self.add_to_primitive(rows, *i, builder.f64()),
                ArrowDataType::Utf8 => add_to_utf8_builder(rows, *i, builder.utf8(), self.encoding),
                _ => todo!(),
            }
        };

        // TODO! benchmark this
        let bp = std::env::var("POLARS_PAR_COLUMN_BP")
            .unwrap_or_else(|_| "".to_string())
            .parse()
            .unwrap_or(15);
        if projection.len() > bp {
            projection
                .par_iter()
                .zip(builders)
                .map(dispatch)
                .collect::<Result<_>>()?;
        } else {
            projection.iter().zip(builders).try_for_each(dispatch)?;
        }

        Ok(())
    }

    fn next_rows(
        &mut self,
        rows: &mut Vec<ByteRecord>,
        record_iter: &mut ByteRecordsIntoIter<R>,
    ) -> Result<()> {
        for i in 0..self.batch_size {
            self.line_number += 1;
            match record_iter.next() {
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

    fn one_thread(
        &mut self,
        builders: &mut Vec<Builder>,
        projection: &[usize],
        parsed_dfs: &mut Vec<DataFrame>,
        mut record_iter: ByteRecordsIntoIter<R>,
    ) -> Result<()> {
        let mut rows = Vec::with_capacity(self.batch_size);
        let mut count = 0;
        loop {
            count += 1;
            rows.clear();
            self.next_rows(&mut rows, &mut record_iter)?;
            // stop when the whole file is processed
            if rows.is_empty() {
                break;
            }
            if count % CAPACITY_MULTIPLIER == 0 {
                let mut builders_tmp = init_builders(
                    &projection,
                    self.batch_size * CAPACITY_MULTIPLIER,
                    &self.schema,
                )?;
                std::mem::swap(&mut builders_tmp, builders);
                parsed_dfs.push(builders_to_df(builders_tmp));
            }

            self.add_to_builders(builders, &projection, &rows)?;

            // stop after n_rows are processed
            if let Some(n_rows) = self.n_rows {
                if self.line_number > n_rows {
                    break;
                }
            }
        }
        Ok(())
    }

    fn two_threads(
        &mut self,
        builders: &mut Vec<Builder>,
        projection: &[usize],
        mut record_iter: ByteRecordsIntoIter<R>,
        parsed_dfs: &mut Vec<DataFrame>,
    ) -> Result<()> {
        let (tx, rx) = bounded(256);
        // used to end rampant thread
        let (tx_end, rx_end) = bounded(1);

        let _ = thread::scope(|s| {
            let batch_size = self.batch_size;
            // note that line number gets copied to the other thread.
            let mut line_number = self.line_number;
            let ignore_parser_errors = self.ignore_parser_errors;
            let n_rows = self.n_rows;

            let _ = s.spawn(move |_| {
                loop {
                    let rows = next_rows(
                        &mut record_iter,
                        batch_size,
                        &mut line_number,
                        ignore_parser_errors,
                        n_rows,
                    );
                    match &rows {
                        Ok(rows) => {
                            // stop when the whole file is processed
                            if rows.is_empty() {
                                break;
                            }
                        }
                        Err(_) => {
                            break;
                        }
                    }
                    match rx_end.try_recv() {
                        Ok(_) | Err(TryRecvError::Disconnected) => {
                            break;
                        }
                        // continue
                        Err(TryRecvError::Empty) => {}
                    }

                    tx.send(rows).expect("could not send message");
                }
            });

            let mut count = 0;
            while let Ok(res) = rx.recv() {
                count += 1;
                if count % CAPACITY_MULTIPLIER == 0 {
                    let mut builders_tmp = init_builders(
                        &projection,
                        self.batch_size * CAPACITY_MULTIPLIER,
                        &self.schema,
                    )?;
                    std::mem::swap(&mut builders_tmp, builders);
                    parsed_dfs.push(builders_to_df(builders_tmp));
                }
                let rows = res?;

                match self.add_to_builders(builders, &projection, &rows) {
                    Ok(_) => {}
                    // kill parsing thread in case of an error.
                    Err(e) => {
                        tx_end.send(true).map_err(anyhow::Error::from)?;
                        return Err(e);
                    }
                };
            }
            Ok(())
        })
        .expect("a thread has panicked")?;
        Ok(())
    }

    pub fn as_df(&mut self) -> Result<DataFrame> {
        let mut record_iter = self.record_iter.take().unwrap();
        if self.skip_rows > 0 {
            for _ in 0..self.skip_rows {
                self.line_number += 1;
                let _ = record_iter.next();
            }
        }

        // only take projections once
        let projection = take_projection(&mut self.projection, &self.schema);
        let mut capacity = self.batch_size * CAPACITY_MULTIPLIER;
        if let Some(n) = self.n_rows {
            self.batch_size = std::cmp::min(self.batch_size, n);
            capacity = std::cmp::min(n, capacity);
        }

        let mut builders = init_builders(&projection, capacity, &self.schema)?;
        // we reuse this container to amortize allocations
        let mut parsed_dfs = Vec::with_capacity(128);

        match self.one_thread {
            true => self.one_thread(&mut builders, &projection, &mut parsed_dfs, record_iter)?,
            false => self.two_threads(&mut builders, &projection, record_iter, &mut parsed_dfs)?,
        }

        parsed_dfs.push(builders_to_df(builders));
        utils::accumulate_dataframes_vertical(parsed_dfs)
    }
}

enum Builder {
    Boolean(PrimitiveChunkedBuilder<BooleanType>),
    Int16(PrimitiveChunkedBuilder<Int16Type>),
    Int32(PrimitiveChunkedBuilder<Int32Type>),
    Int64(PrimitiveChunkedBuilder<Int64Type>),
    UInt64(PrimitiveChunkedBuilder<UInt64Type>),
    UInt32(PrimitiveChunkedBuilder<UInt32Type>),
    UInt16(PrimitiveChunkedBuilder<UInt16Type>),
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
    fn i16(&mut self) -> &mut PrimitiveChunkedBuilder<Int16Type> {
        match self {
            Builder::Int16(builder) => builder,
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
    fn u16(&mut self) -> &mut PrimitiveChunkedBuilder<UInt16Type> {
        match self {
            Builder::UInt16(builder) => builder,
            _ => panic!("implementation error"),
        }
    }
    fn u32(&mut self) -> &mut PrimitiveChunkedBuilder<UInt32Type> {
        match self {
            Builder::UInt32(builder) => builder,
            _ => panic!("implementation error"),
        }
    }
    fn u64(&mut self) -> &mut PrimitiveChunkedBuilder<UInt64Type> {
        match self {
            Builder::UInt64(builder) => builder,
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
            Int16(b) => b.finish().into(),
            Int32(b) => b.finish().into(),
            Int64(b) => b.finish().into(),
            UInt16(b) => b.finish().into(),
            UInt32(b) => b.finish().into(),
            UInt64(b) => b.finish().into(),
            Float32(b) => b.finish().into(),
            Float64(b) => b.finish().into(),
            Boolean(b) => b.finish().into(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_csv_reader<R: 'static + Read + Seek + Sync + Send>(
    mut reader: R,
    n_rows: Option<usize>,
    skip_rows: usize,
    mut projection: Option<Vec<usize>>,
    batch_size: usize,
    max_records: Option<usize>,
    delimiter: Option<u8>,
    has_header: bool,
    ignore_parser_errors: bool,
    schema: Option<SchemaRef>,
    columns: Option<Vec<String>>,
    encoding: CsvEncoding,
    one_thread: bool,
) -> Result<SequentialReader<R>> {
    // check if schema should be inferred
    let delimiter = delimiter.unwrap_or(b',');
    let schema = match schema {
        Some(schema) => schema,
        None => {
            let (inferred_schema, _) =
                infer_file_schema(&mut reader, delimiter, max_records, has_header)?;

            Arc::new(inferred_schema)
        }
    };

    if let Some(cols) = columns {
        let mut prj = Vec::with_capacity(cols.len());
        for col in cols {
            let i = schema.index_of(&col)?;
            prj.push(i);
        }
        projection = Some(prj);
    }

    Ok(SequentialReader::from_reader(
        reader,
        schema,
        has_header,
        delimiter,
        batch_size,
        projection,
        ignore_parser_errors,
        n_rows,
        skip_rows,
        encoding,
        one_thread,
    ))
}
