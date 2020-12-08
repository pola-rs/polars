// Licensed to the Apache Software Foundation (ASF) under one_
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
#[cfg(feature = "lazy")]
use crate::lazy::prelude::PhysicalExpr;
use crate::prelude::*;
use crate::utils;
use ahash::RandomState;
use arrow::datatypes::SchemaRef;
use crossbeam::thread;
use crossbeam::thread::ScopedJoinHandle;
use csv::{ByteRecord, ByteRecordsIntoIter, Reader};
use num::traits::Pow;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

#[cfg(not(feature = "lazy"))]
pub trait PhysicalExpr {
    fn evaluate(&self, df: &DataFrame) -> Result<Series>;
}

/// Is multiplied with batch_size to determine capacity of builders
const CAPACITY_MULTIPLIER: usize = 512;

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn next_line_position(bytes: &[u8]) -> Option<usize> {
    let pos = find_subsequence(bytes, b"\n")?;
    bytes.get(pos + 1).and_then(|&b| {
        Option::from({
            if b == b'\r' {
                pos + 1
            } else {
                pos
            }
        })
    })
}

/// Get the mean and standard deviation of length of lines in bytes
fn get_line_stats(mut bytes: &[u8], n_lines: usize) -> Option<(f32, f32)> {
    let mut n_read = 0;
    let mut lengths = Vec::with_capacity(n_lines);

    for _ in 0..n_lines {
        if n_read >= bytes.len() {
            return None;
        }
        bytes = &bytes[n_read..];
        match bytes.iter().position(|&b| b == b'\n') {
            Some(position) => {
                n_read = position + 1;
                lengths.push(position + 1);
            }
            None => {
                return None;
            }
        }
    }
    let mean = (n_read as f32) / (n_lines as f32);
    let mut std = 0.0;
    for &len in lengths.iter().take(n_lines) {
        std += (len as f32 - mean).pow(2.0)
    }
    std = (std / n_lines as f32).pow(0.5);
    Some((mean, std))
}

fn get_file_chunks(bytes: &[u8], n_threads: usize) -> Vec<(usize, usize)> {
    let mut last_pos = 0;
    let total_len = bytes.len();
    let chunk_size = total_len / n_threads;
    let mut offsets = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        let search_pos = last_pos + chunk_size;

        if search_pos >= bytes.len() {
            break;
        }

        let end_pos = match next_line_position(&bytes[search_pos..]) {
            Some(pos) => search_pos + pos,
            None => {
                break;
            }
        };
        offsets.push((last_pos, end_pos + 1));
        last_pos = end_pos;
    }
    offsets.push((last_pos, total_len));
    offsets
}

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
pub(crate) fn infer_file_schema<R: Read + Seek>(
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

fn next_rows<R: Read + Send + Sync>(
    rows: &mut Vec<ByteRecord>,
    csv_reader: &mut Reader<R>,
    line_number: &mut usize,
    batch_size: usize,
    ignore_parser_errors: bool,
) -> Result<usize> {
    let mut count = 0;
    loop {
        *line_number += 1;
        debug_assert!(rows.get(count).is_some());
        let record = unsafe { rows.get_unchecked_mut(count) };

        match csv_reader.read_byte_record(record) {
            Ok(true) => count += 1,
            // end of file
            Ok(false) => {
                break;
            }
            Err(e) => {
                if ignore_parser_errors {
                    continue;
                } else {
                    return Err(PolarsError::Other(
                        format!("Error parsing line {}: {:?}", line_number, e).into(),
                    ));
                }
            }
        }
        if count == batch_size {
            break;
        }
    }
    Ok(count)
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
    n_threads: Option<usize>,
    path: Option<String>,
    has_header: bool,
    delimiter: u8,
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
        n_threads: Option<usize>,
        path: Option<String>,
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
            n_threads,
            path,
            has_header,
            delimiter,
        }
    }

    fn add_to_primitive<T>(
        &self,
        rows: &[ByteRecord],
        col_idx: usize,
        builder: &mut PrimitiveChunkedBuilder<T>,
    ) -> Result<()>
    where
        T: PolarsPrimitiveType + PrimitiveParser,
    {
        for (row_index, row) in rows.iter().enumerate() {
            match row.get(col_idx) {
                Some(bytes) => {
                    if bytes.is_empty() {
                        builder.append_null();
                        continue;
                    }
                    match T::parse(bytes) {
                        Ok(e) => builder.append_value(e),
                        Err(_) => {
                            if self.ignore_parser_errors {
                                builder.append_null();
                                continue;
                            }
                            return Err(PolarsError::Other(
                                format!(
                                    "Error while parsing value {} for column {} at line {}",
                                    String::from_utf8_lossy(bytes),
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
        bp: usize,
    ) -> Result<()> {
        let dispatch = |(i, builder): (&usize, &mut Builder)| {
            let field = self.schema.field(*i);
            match field.data_type() {
                ArrowDataType::Boolean => self.add_to_primitive(rows, *i, builder.bool()),
                ArrowDataType::Int8 => self.add_to_primitive(rows, *i, builder.i32()),
                ArrowDataType::Int16 => self.add_to_primitive(rows, *i, builder.i32()),
                ArrowDataType::Int32 => self.add_to_primitive(rows, *i, builder.i32()),
                ArrowDataType::Int64 => self.add_to_primitive(rows, *i, builder.i64()),
                ArrowDataType::UInt8 => self.add_to_primitive(rows, *i, builder.u32()),
                ArrowDataType::UInt16 => self.add_to_primitive(rows, *i, builder.u32()),
                ArrowDataType::UInt32 => self.add_to_primitive(rows, *i, builder.u32()),
                ArrowDataType::UInt64 => self.add_to_primitive(rows, *i, builder.u64()),
                ArrowDataType::Float32 => self.add_to_primitive(rows, *i, builder.f32()),
                ArrowDataType::Float64 => self.add_to_primitive(rows, *i, builder.f64()),
                ArrowDataType::Utf8 => add_to_utf8_builder(rows, *i, builder.utf8(), self.encoding),
                _ => todo!(),
            }
        };

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

    fn one_thread(
        &mut self,
        projection: &[usize],
        parsed_dfs: &mut Vec<DataFrame>,
        csv_reader: &mut Reader<R>,
        predicate: &Option<Arc<dyn PhysicalExpr>>,
        capacity: usize,
    ) -> Result<()> {
        self.batch_size = std::cmp::max(self.batch_size, 128);
        if self.skip_rows > 0 {
            let mut record = Default::default();
            for _ in 0..self.skip_rows {
                self.line_number += 1;
                let _ = csv_reader.read_byte_record(&mut record);
            }
        }

        let mut builders = init_builders(&projection, capacity, &self.schema)?;
        // this will be used to amortize allocations
        // Only correctly parsed lines will fill the start of the Vec.
        // The whole vec is initialized with default values.
        // Once a batch of rows is read we use the correctly parsed information to truncate the lenght
        // to all correct values.
        let mut rows = Vec::with_capacity(self.batch_size);
        rows.resize_with(self.batch_size, Default::default);
        let mut count = 0;
        let mut line_number = self.line_number;
        // TODO! benchmark this
        let bp = std::env::var("POLARS_PAR_COLUMN_BP")
            .unwrap_or_else(|_| "".to_string())
            .parse()
            .unwrap_or(15);
        loop {
            count += 1;
            let correctly_parsed = next_rows(
                &mut rows,
                csv_reader,
                &mut line_number,
                self.batch_size,
                self.ignore_parser_errors,
            )?;
            // stop when the whole file is processed
            if correctly_parsed == 0 {
                break;
            } else if correctly_parsed < self.batch_size {
                // this only happens at the last batch if it doesn't fit a whole batch.
                rows.truncate(correctly_parsed);
            }
            if count % CAPACITY_MULTIPLIER == 0 {
                let mut builders_tmp = init_builders(&projection, capacity, &self.schema)?;
                std::mem::swap(&mut builders_tmp, &mut builders);
                finish_builder(builders_tmp, parsed_dfs, predicate)?;
            }

            self.add_to_builders(&mut builders, &projection, &rows, bp)?;

            // stop after n_rows are processed
            if let Some(n_rows) = self.n_rows {
                if line_number > n_rows {
                    break;
                }
            }
        }
        finish_builder(builders, parsed_dfs, predicate)?;
        Ok(())
    }

    fn n_threads(
        &mut self,
        projection: &[usize],
        parsed_dfs: &mut Vec<DataFrame>,
        predicate: &Option<Arc<dyn PhysicalExpr>>,
        capacity: usize,
        n_threads: usize,
    ) -> Result<()> {
        let path = self.path.as_ref().unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };

        let mut bytes = mmap[..].as_ref();
        if self.has_header {
            let pos = next_line_position(bytes).expect("no newline characters found in file");
            bytes = mmap[pos..].as_ref();
        }
        if self.skip_rows > 0 {
            for _ in 0..self.skip_rows {
                let pos = next_line_position(bytes).expect("no newline characters found in file");
                bytes = bytes[pos..].as_ref();
            }
        }
        if let Some(n_rows) = self.n_rows {
            // if None, there are less then 128 rows in the file and skipping them isn't worth the effort
            let n_sample_lines = 128;
            if let Some((mean, std)) = get_line_stats(bytes, n_sample_lines) {
                // x % upper bound of byte length per line assuming normally distributed
                let line_bound = mean + 1.1 * std;

                let n_bytes = (line_bound * (n_rows as f32)) as usize;
                if n_bytes < bytes.len() {
                    if let Some(pos) = next_line_position(&bytes[n_bytes..]) {
                        bytes = &bytes[..n_bytes + pos]
                    }
                }
            }
        }

        let file_chunks = get_file_chunks(bytes, n_threads);

        let scopes: Result<_> = thread::scope(|s| {
            let mut handlers = Vec::with_capacity(n_threads);

            for (mut total_bytes_offset, stop_at_nbytes) in file_chunks {
                let delimiter = self.delimiter;
                let batch_size = self.batch_size;
                let schema = self.schema.clone();
                let ignore_parser_errors = self.ignore_parser_errors;
                let encoding = self.encoding;

                let h: ScopedJoinHandle<Result<_>> = s.spawn(move |_| {
                    // container to ammortize allocs
                    let mut rows = Vec::with_capacity(batch_size);
                    rows.resize_with(batch_size, Default::default);

                    let mut builders = init_builders(&projection, capacity, &schema).unwrap();
                    let mut local_parsed_dfs = Vec::with_capacity(16);

                    let mut local_bytes;
                    let mut core_reader =
                        csv_core::ReaderBuilder::new().delimiter(delimiter).build();

                    let mut count = 0;
                    loop {
                        count += 1;
                        local_bytes = &bytes[total_bytes_offset..stop_at_nbytes];
                        let (correctly_parsed, bytes_read) =
                            next_rows_core(&mut rows, local_bytes, &mut core_reader, batch_size);
                        total_bytes_offset += bytes_read;

                        if correctly_parsed < batch_size {
                            if correctly_parsed == 0 {
                                break;
                            }
                            // this only happens at the last batch if it doesn't fit a whole batch.
                            rows.truncate(correctly_parsed);
                        }
                        add_to_builders_core(
                            &mut builders,
                            &projection,
                            &rows,
                            &schema,
                            ignore_parser_errors,
                            encoding,
                        )?;

                        if total_bytes_offset >= stop_at_nbytes {
                            break;
                        }

                        if count % CAPACITY_MULTIPLIER == 0 {
                            let mut builders_tmp =
                                init_builders(&projection, capacity, &schema).unwrap();
                            std::mem::swap(&mut builders_tmp, &mut builders);
                            finish_builder(builders_tmp, &mut local_parsed_dfs, predicate).unwrap();
                        }
                    }
                    finish_builder(builders, &mut local_parsed_dfs, predicate)?;
                    Ok(local_parsed_dfs)
                });
                handlers.push(h)
            }
            for h in handlers {
                let local_parsed_dfs = h.join().expect("thread panicked")?;
                parsed_dfs.extend(local_parsed_dfs.into_iter())
            }

            Ok(())
        })
        .unwrap();
        let _ = scopes?;

        Ok(())
    }

    /// Read the csv into a DataFrame. The predicate can come from a lazy physical plan.
    pub fn as_df(&mut self, predicate: Option<Arc<dyn PhysicalExpr>>) -> Result<DataFrame> {
        let mut record_iter = self.record_iter.take().unwrap();

        // only take projections once
        let projection = take_projection(&mut self.projection, &self.schema);
        let mut capacity = self.batch_size * CAPACITY_MULTIPLIER;
        if let Some(n) = self.n_rows {
            self.batch_size = std::cmp::min(self.batch_size, n);
            capacity = std::cmp::min(n, capacity);
        }

        let physical_cpus = num_cpus::get_physical();
        let mut n_threads =
            std::cmp::min(physical_cpus * 4, self.n_threads.unwrap_or(physical_cpus));
        if self.path.is_none() {
            n_threads = 1;
        }

        // we reuse this container to amortize allocations
        let mut parsed_dfs = Vec::with_capacity(128);

        match n_threads {
            1 => self.one_thread(
                &projection,
                &mut parsed_dfs,
                record_iter.reader_mut(),
                &predicate,
                capacity,
            )?,
            _ => self.n_threads(
                &projection,
                &mut parsed_dfs,
                &predicate,
                capacity,
                n_threads,
            )?,
        }
        let df = utils::accumulate_dataframes_vertical(parsed_dfs);

        // if multi-threaded the n_rows was probabilistically determined.
        // Let's slice to correct number of rows if possible.
        if n_threads > 1 {
            if let Some(n_rows) = self.n_rows {
                return df.map(|df| {
                    if n_rows < df.height() {
                        df.slice(0, n_rows).unwrap()
                    } else {
                        df
                    }
                });
            }
        }
        df
    }
}

fn add_to_builders_core(
    builders: &mut [Builder],
    projection: &[usize],
    rows: &[PolarsCsvRecord],
    schema: &Schema,
    ignore_parser_error: bool,
    encoding: CsvEncoding,
) -> Result<()> {
    let dispatch = |(i, builder): (&usize, &mut Builder)| {
        let field = schema.field(*i);
        match field.data_type() {
            ArrowDataType::Boolean => {
                add_to_primitive_core(rows, *i, builder.bool(), ignore_parser_error)
            }
            ArrowDataType::Int8 => {
                add_to_primitive_core(rows, *i, builder.i32(), ignore_parser_error)
            }
            ArrowDataType::Int16 => {
                add_to_primitive_core(rows, *i, builder.i32(), ignore_parser_error)
            }
            ArrowDataType::Int32 => {
                add_to_primitive_core(rows, *i, builder.i32(), ignore_parser_error)
            }
            ArrowDataType::Int64 => {
                add_to_primitive_core(rows, *i, builder.i64(), ignore_parser_error)
            }
            ArrowDataType::UInt8 => {
                add_to_primitive_core(rows, *i, builder.u32(), ignore_parser_error)
            }
            ArrowDataType::UInt16 => {
                add_to_primitive_core(rows, *i, builder.u32(), ignore_parser_error)
            }
            ArrowDataType::UInt32 => {
                add_to_primitive_core(rows, *i, builder.u32(), ignore_parser_error)
            }
            ArrowDataType::UInt64 => {
                add_to_primitive_core(rows, *i, builder.u64(), ignore_parser_error)
            }
            ArrowDataType::Float32 => {
                add_to_primitive_core(rows, *i, builder.f32(), ignore_parser_error)
            }
            ArrowDataType::Float64 => {
                add_to_primitive_core(rows, *i, builder.f64(), ignore_parser_error)
            }
            ArrowDataType::Utf8 => add_to_utf8_builder_core(rows, *i, builder.utf8(), encoding),
            _ => todo!(),
        }
    };

    projection.iter().zip(builders).try_for_each(dispatch)?;

    Ok(())
}

fn add_to_utf8_builder_core(
    rows: &[PolarsCsvRecord],
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

fn add_to_primitive_core<T>(
    rows: &[PolarsCsvRecord],
    col_idx: usize,
    builder: &mut PrimitiveChunkedBuilder<T>,
    ignore_parser_errors: bool,
) -> Result<()>
where
    T: PolarsPrimitiveType + PrimitiveParser,
{
    // todo! keep track of line number for error reporting
    for (_row_index, row) in rows.iter().enumerate() {
        match row.get(col_idx) {
            Some(bytes) => {
                if bytes.is_empty() {
                    builder.append_null();
                    continue;
                }
                match T::parse(bytes) {
                    Ok(e) => builder.append_value(e),
                    Err(_) => {
                        if ignore_parser_errors {
                            builder.append_null();
                            continue;
                        }
                        return Err(PolarsError::Other(
                            format!(
                                "Error while parsing value {} for column {} as {:?}",
                                String::from_utf8_lossy(bytes),
                                col_idx,
                                T::get_data_type()
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

#[derive(Debug)]
struct PolarsCsvRecord {
    out: Vec<u8>,
    ends: Vec<usize>,
    n_out: usize,
}

impl PolarsCsvRecord {
    fn get(&self, index: usize) -> Option<&[u8]> {
        let start = match index.checked_sub(1).and_then(|idx| self.ends.get(idx)) {
            None => 0,
            Some(i) => *i,
        };
        let end = match self.ends.get(index) {
            Some(i) => *i,
            None => return None,
        };

        Some(&self.out[start..end])
    }

    fn resize_out_buffer(&mut self) {
        let size = std::cmp::max(self.out.len() * 2, 128);
        self.out.resize(size, 0);
    }

    fn reset(&mut self) {
        self.ends.truncate(0);
        self.n_out = 0;
    }
}

impl Default for PolarsCsvRecord {
    fn default() -> Self {
        PolarsCsvRecord {
            out: vec![],
            ends: vec![],
            n_out: 0,
        }
    }
}

fn next_rows_core(
    rows: &mut Vec<PolarsCsvRecord>,
    mut bytes: &[u8],
    reader: &mut csv_core::Reader,
    batch_size: usize,
) -> (usize, usize) {
    let mut line_count = 0;
    let mut bytes_read = 0;
    loop {
        debug_assert!(rows.get(line_count).is_some());
        let mut record = unsafe { rows.get_unchecked_mut(line_count) };
        record.reset();

        use csv_core::ReadFieldResult;
        loop {
            let (result, n_in, n_out) = reader.read_field(bytes, &mut record.out[record.n_out..]);
            match result {
                ReadFieldResult::Field { record_end } => {
                    bytes_read += n_in;
                    bytes = &bytes[n_in..];
                    record.n_out += n_out;
                    record.ends.push(record.n_out);

                    if record_end {
                        line_count += 1;
                        break;
                    }
                }
                ReadFieldResult::OutputFull => {
                    record.resize_out_buffer();
                }
                ReadFieldResult::End | ReadFieldResult::InputEmpty => {
                    return (line_count, bytes_read);
                }
            }
        }
        if line_count == batch_size {
            break;
        }
    }
    (line_count, bytes_read)
}

fn finish_builder(
    builders: Vec<Builder>,
    parsed_dfs: &mut Vec<DataFrame>,
    predicate: &Option<Arc<dyn PhysicalExpr>>,
) -> Result<()> {
    let mut df = builders_to_df(builders);
    if let Some(predicate) = &predicate {
        let s = predicate.evaluate(&df)?;
        let mask = s.bool().expect("filter predicates was not of type boolean");
        let local_df = df.filter(mask)?;
        if df.height() > 0 {
            df = local_df;
        }
    }
    parsed_dfs.push(df);
    Ok(())
}

impl From<lexical::Error> for PolarsError {
    fn from(_: lexical::Error) -> Self {
        PolarsError::Other("Could not parse primitive type during csv parsing".into())
    }
}

trait PrimitiveParser: ArrowPrimitiveType {
    fn parse(bytes: &[u8]) -> Result<Self::Native>;
}

impl PrimitiveParser for BooleanType {
    fn parse(bytes: &[u8]) -> Result<bool> {
        if bytes.eq_ignore_ascii_case(b"false") {
            Ok(false)
        } else if bytes.eq_ignore_ascii_case(b"true") {
            Ok(true)
        } else {
            Err(PolarsError::Other("Could not parse boolean".into()))
        }
    }
}

impl PrimitiveParser for Float32Type {
    fn parse(bytes: &[u8]) -> Result<f32> {
        let a = lexical::parse(bytes)?;
        Ok(a)
    }
}
impl PrimitiveParser for Float64Type {
    fn parse(bytes: &[u8]) -> Result<f64> {
        let a = lexical::parse(bytes)?;
        Ok(a)
    }
}

impl PrimitiveParser for UInt8Type {
    fn parse(bytes: &[u8]) -> Result<u8> {
        let a = lexical::parse(bytes)?;
        Ok(a)
    }
}
impl PrimitiveParser for UInt16Type {
    fn parse(bytes: &[u8]) -> Result<u16> {
        let a = lexical::parse(bytes)?;
        Ok(a)
    }
}
impl PrimitiveParser for UInt32Type {
    fn parse(bytes: &[u8]) -> Result<u32> {
        let a = lexical::parse(bytes)?;
        Ok(a)
    }
}
impl PrimitiveParser for UInt64Type {
    fn parse(bytes: &[u8]) -> Result<u64> {
        let a = lexical::parse(bytes)?;
        Ok(a)
    }
}
impl PrimitiveParser for Int8Type {
    fn parse(bytes: &[u8]) -> Result<i8> {
        let a = lexical::parse(bytes)?;
        Ok(a)
    }
}
impl PrimitiveParser for Int16Type {
    fn parse(bytes: &[u8]) -> Result<i16> {
        let a = lexical::parse(bytes)?;
        Ok(a)
    }
}
impl PrimitiveParser for Int32Type {
    fn parse(bytes: &[u8]) -> Result<i32> {
        let a = lexical::parse(bytes)?;
        Ok(a)
    }
}
impl PrimitiveParser for Int64Type {
    fn parse(bytes: &[u8]) -> Result<i64> {
        let a = lexical::parse(bytes)?;
        Ok(a)
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
    n_threads: Option<usize>,
    path: Option<String>,
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
        n_threads,
        path,
    ))
}
