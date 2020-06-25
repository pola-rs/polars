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

//! CSV Writer
//!
//! This CSV writer allows Arrow data (in record batches) to be written as CSV files.
//! The writer does not support writing `ListArray` and `StructArray`.
//!
//! Example:
//!
//! ```
//! use some::array::*;
//! use some::csv;
//! use some::datatypes::*;
//! use some::record_batch::RecordBatch;
//! use some::util::test_util::get_temp_file;
//! use std::fs::File;
//! use std::sync::Arc;
//!
//! let schema = Schema::new(vec![
//!     Field::new("c1", DataType::Utf8, false),
//!     Field::new("c2", DataType::Float64, true),
//!     Field::new("c3", DataType::UInt32, false),
//!     Field::new("c3", DataType::Boolean, true),
//! ]);
//! let c1 = StringArray::from(vec![
//!     "Lorem ipsum dolor sit amet",
//!     "consectetur adipiscing elit",
//!     "sed do eiusmod tempor",
//! ]);
//! let c2 = PrimitiveArray::<Float64Type>::from(vec![
//!     Some(123.564532),
//!     None,
//!     Some(-556132.25),
//! ]);
//! let c3 = PrimitiveArray::<UInt32Type>::from(vec![3, 2, 1]);
//! let c4 = PrimitiveArray::<BooleanType>::from(vec![Some(true), Some(false), None]);
//!
//! let batch = RecordBatch::try_new(
//!     Arc::new(schema),
//!     vec![Arc::new(c1), Arc::new(c2), Arc::new(c3), Arc::new(c4)],
//! )
//! .unwrap();
//!
//! let file = get_temp_file("out.csv", &[]);
//!
//! let mut writer = csv::Writer::new(file);
//! let batches = vec![&batch, &batch];
//! for batch in batches {
//!     writer.write(batch).unwrap();
//! }
//! ```

use csv as csv_crate;

use std::io::Write;

use crate::array::*;
use crate::datatypes::*;
use crate::error::{ArrowError, Result};
use crate::record_batch::RecordBatch;

const DEFAULT_DATE_FORMAT: &str = "%F";
const DEFAULT_TIME_FORMAT: &str = "%T";
const DEFAULT_TIMESTAMP_FORMAT: &str = "%FT%H:%M:%S.%9f";

fn write_primitive_value<T>(array: &ArrayRef, i: usize) -> String
where
    T: ArrowNumericType,
    T::Native: std::string::ToString,
{
    let c = array.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    c.value(i).to_string()
}

/// A CSV writer
pub struct Writer<W: Write> {
    /// The object to write to
    writer: csv_crate::Writer<W>,
    /// Column delimiter. Defaults to `b','`
    delimiter: u8,
    /// Whether file should be written with headers. Defaults to `true`
    has_headers: bool,
    /// The date format for date arrays
    date_format: String,
    /// The timestamp format for timestamp arrays
    timestamp_format: String,
    /// The time format for time arrays
    time_format: String,
    /// Is the beginning-of-writer
    beginning: bool,
}

impl<W: Write> Writer<W> {
    /// Create a new CsvWriter from a writable object, with default options
    pub fn new(writer: W) -> Self {
        let delimiter = b',';
        let mut builder = csv_crate::WriterBuilder::new();
        let writer = builder.delimiter(delimiter).from_writer(writer);
        Writer {
            writer,
            delimiter,
            has_headers: true,
            date_format: DEFAULT_DATE_FORMAT.to_string(),
            time_format: DEFAULT_TIME_FORMAT.to_string(),
            timestamp_format: DEFAULT_TIMESTAMP_FORMAT.to_string(),
            beginning: true,
        }
    }

    /// Convert a record to a string vector
    fn convert(&self, batch: &RecordBatch, row_index: usize) -> Result<Vec<String>> {
        // TODO: it'd be more efficient if we could create `record: Vec<&[u8]>
        let mut record: Vec<String> = Vec::with_capacity(batch.num_columns());
        for col_index in 0..batch.num_columns() {
            let col = batch.column(col_index);
            if col.is_null(row_index) {
                // write an empty value
                record.push(String::from(""));
                continue;
            }
            let string = match col.data_type() {
                DataType::Float64 => write_primitive_value::<Float64Type>(col, row_index),
                DataType::Float32 => write_primitive_value::<Float32Type>(col, row_index),
                DataType::Int8 => write_primitive_value::<Int8Type>(col, row_index),
                DataType::Int16 => write_primitive_value::<Int16Type>(col, row_index),
                DataType::Int32 => write_primitive_value::<Int32Type>(col, row_index),
                DataType::Int64 => write_primitive_value::<Int64Type>(col, row_index),
                DataType::UInt8 => write_primitive_value::<UInt8Type>(col, row_index),
                DataType::UInt16 => write_primitive_value::<UInt16Type>(col, row_index),
                DataType::UInt32 => write_primitive_value::<UInt32Type>(col, row_index),
                DataType::UInt64 => write_primitive_value::<UInt64Type>(col, row_index),
                DataType::Boolean => {
                    let c = col.as_any().downcast_ref::<BooleanArray>().unwrap();
                    c.value(row_index).to_string()
                }
                DataType::Utf8 => {
                    let c = col.as_any().downcast_ref::<StringArray>().unwrap();
                    c.value(row_index).to_owned()
                }
                DataType::Date32(DateUnit::Day) => {
                    let c = col.as_any().downcast_ref::<Date32Array>().unwrap();
                    c.value_as_date(row_index)
                        .unwrap()
                        .format(&self.date_format)
                        .to_string()
                }
                DataType::Date64(DateUnit::Millisecond) => {
                    let c = col.as_any().downcast_ref::<Date64Array>().unwrap();
                    c.value_as_date(row_index)
                        .unwrap()
                        .format(&self.date_format)
                        .to_string()
                }
                DataType::Time32(TimeUnit::Second) => {
                    let c = col.as_any().downcast_ref::<Time32SecondArray>().unwrap();
                    c.value_as_time(row_index)
                        .unwrap()
                        .format(&self.time_format)
                        .to_string()
                }
                DataType::Time32(TimeUnit::Millisecond) => {
                    let c = col
                        .as_any()
                        .downcast_ref::<Time32MillisecondArray>()
                        .unwrap();
                    c.value_as_time(row_index)
                        .unwrap()
                        .format(&self.time_format)
                        .to_string()
                }
                DataType::Time64(TimeUnit::Microsecond) => {
                    let c = col
                        .as_any()
                        .downcast_ref::<Time64MicrosecondArray>()
                        .unwrap();
                    c.value_as_time(row_index)
                        .unwrap()
                        .format(&self.time_format)
                        .to_string()
                }
                DataType::Time64(TimeUnit::Nanosecond) => {
                    let c = col
                        .as_any()
                        .downcast_ref::<Time64NanosecondArray>()
                        .unwrap();
                    c.value_as_time(row_index)
                        .unwrap()
                        .format(&self.time_format)
                        .to_string()
                }
                DataType::Timestamp(time_unit, _) => {
                    use TimeUnit::*;
                    let datetime = match time_unit {
                        Second => col
                            .as_any()
                            .downcast_ref::<TimestampSecondArray>()
                            .unwrap()
                            .value_as_datetime(row_index)
                            .unwrap(),
                        Millisecond => col
                            .as_any()
                            .downcast_ref::<TimestampMillisecondArray>()
                            .unwrap()
                            .value_as_datetime(row_index)
                            .unwrap(),
                        Microsecond => col
                            .as_any()
                            .downcast_ref::<TimestampMicrosecondArray>()
                            .unwrap()
                            .value_as_datetime(row_index)
                            .unwrap(),
                        Nanosecond => col
                            .as_any()
                            .downcast_ref::<TimestampNanosecondArray>()
                            .unwrap()
                            .value_as_datetime(row_index)
                            .unwrap(),
                    };
                    format!("{}", datetime.format(&self.timestamp_format))
                }
                t => {
                    // List and Struct arrays not supported by the writer, any
                    // other type needs to be implemented
                    return Err(ArrowError::CsvError(format!(
                        "CSV Writer does not support {:?} data type",
                        t
                    )));
                }
            };

            record.push(string);
        }
        Ok(record)
    }

    /// Write a vector of record batches to a writable object
    pub fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        let num_columns = batch.num_columns();
        if self.beginning {
            if self.has_headers {
                let mut headers: Vec<String> = Vec::with_capacity(num_columns);
                batch
                    .schema()
                    .fields()
                    .iter()
                    .for_each(|field| headers.push(field.name().to_string()));
                self.writer.write_record(&headers[..])?;
            }
            self.beginning = false;
        }

        for row_index in 0..batch.num_rows() {
            let record = self.convert(batch, row_index)?;
            self.writer.write_record(&record[..])?;
        }
        self.writer.flush()?;

        Ok(())
    }
}

/// A CSV writer builder
pub struct WriterBuilder {
    /// Optional column delimiter. Defaults to `b','`
    delimiter: Option<u8>,
    /// Whether to write column names as file headers. Defaults to `true`
    has_headers: bool,
    /// Optional date format for date arrays
    date_format: Option<String>,
    /// Optional timestamp format for timestamp arrays
    timestamp_format: Option<String>,
    /// Optional time format for time arrays
    time_format: Option<String>,
}

impl Default for WriterBuilder {
    fn default() -> Self {
        Self {
            has_headers: true,
            delimiter: None,
            date_format: Some(DEFAULT_DATE_FORMAT.to_string()),
            time_format: Some(DEFAULT_TIME_FORMAT.to_string()),
            timestamp_format: Some(DEFAULT_TIMESTAMP_FORMAT.to_string()),
        }
    }
}

impl WriterBuilder {
    /// Create a new builder for configuring CSV writing options.
    ///
    /// To convert a builder into a writer, call `WriterBuilder::build`
    ///
    /// # Example
    ///
    /// ```
    /// extern crate some;
    ///
    /// use some::csv;
    /// use std::fs::File;
    ///
    /// fn example() -> csv::Writer<File> {
    ///     let file = File::create("target/out.csv").unwrap();
    ///
    ///     // create a builder that doesn't write headers
    ///     let builder = csv::WriterBuilder::new().has_headers(false);
    ///     let writer = builder.build(file);
    ///
    ///     writer
    /// }
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to write headers
    pub fn has_headers(mut self, has_headers: bool) -> Self {
        self.has_headers = has_headers;
        self
    }

    /// Set the CSV file's column delimiter as a byte character
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = Some(delimiter);
        self
    }

    /// Set the CSV file's date format
    pub fn with_date_format(mut self, format: String) -> Self {
        self.date_format = Some(format);
        self
    }

    /// Set the CSV file's time format
    pub fn with_time_format(mut self, format: String) -> Self {
        self.time_format = Some(format);
        self
    }

    /// Set the CSV file's timestamp format
    pub fn with_timestamp_format(mut self, format: String) -> Self {
        self.timestamp_format = Some(format);
        self
    }

    /// Create a new `Writer`
    pub fn build<W: Write>(self, writer: W) -> Writer<W> {
        let delimiter = self.delimiter.unwrap_or(b',');
        let mut builder = csv_crate::WriterBuilder::new();
        let writer = builder.delimiter(delimiter).from_writer(writer);
        Writer {
            writer,
            delimiter,
            has_headers: self.has_headers,
            date_format: self
                .date_format
                .unwrap_or_else(|| DEFAULT_DATE_FORMAT.to_string()),
            time_format: self
                .time_format
                .unwrap_or_else(|| DEFAULT_TIME_FORMAT.to_string()),
            timestamp_format: self
                .timestamp_format
                .unwrap_or_else(|| DEFAULT_TIMESTAMP_FORMAT.to_string()),
            beginning: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::datatypes::{Field, Schema};
    use crate::util::string_writer::StringWriter;
    use crate::util::test_util::get_temp_file;
    use std::fs::File;
    use std::io::Read;
    use std::sync::Arc;

    #[test]
    fn test_write_csv() {
        let schema = Schema::new(vec![
            Field::new("c1", DataType::Utf8, false),
            Field::new("c2", DataType::Float64, true),
            Field::new("c3", DataType::UInt32, false),
            Field::new("c4", DataType::Boolean, true),
            Field::new("c5", DataType::Timestamp(TimeUnit::Millisecond, None), true),
            Field::new("c6", DataType::Time32(TimeUnit::Second), false),
        ]);

        let c1 = StringArray::from(vec![
            "Lorem ipsum dolor sit amet",
            "consectetur adipiscing elit",
            "sed do eiusmod tempor",
        ]);
        let c2 = PrimitiveArray::<Float64Type>::from(vec![
            Some(123.564532),
            None,
            Some(-556132.25),
        ]);
        let c3 = PrimitiveArray::<UInt32Type>::from(vec![3, 2, 1]);
        let c4 = PrimitiveArray::<BooleanType>::from(vec![Some(true), Some(false), None]);
        let c5 = TimestampMillisecondArray::from_opt_vec(
            vec![None, Some(1555584887378), Some(1555555555555)],
            None,
        );
        let c6 = Time32SecondArray::from(vec![1234, 24680, 85563]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(c1),
                Arc::new(c2),
                Arc::new(c3),
                Arc::new(c4),
                Arc::new(c5),
                Arc::new(c6),
            ],
        )
        .unwrap();

        let file = get_temp_file("columns.csv", &[]);

        let mut writer = Writer::new(file);
        let batches = vec![&batch, &batch];
        for batch in batches {
            writer.write(batch).unwrap();
        }
        // check that file was written successfully
        let mut file = File::open("target/debug/testdata/columns.csv").unwrap();
        let mut buffer: Vec<u8> = vec![];
        file.read_to_end(&mut buffer).unwrap();

        assert_eq!(
            r#"c1,c2,c3,c4,c5,c6
Lorem ipsum dolor sit amet,123.564532,3,true,,00:20:34
consectetur adipiscing elit,,2,false,2019-04-18T10:54:47.378000000,06:51:20
sed do eiusmod tempor,-556132.25,1,,2019-04-18T02:45:55.555000000,23:46:03
Lorem ipsum dolor sit amet,123.564532,3,true,,00:20:34
consectetur adipiscing elit,,2,false,2019-04-18T10:54:47.378000000,06:51:20
sed do eiusmod tempor,-556132.25,1,,2019-04-18T02:45:55.555000000,23:46:03
"#
            .to_string(),
            String::from_utf8(buffer).unwrap()
        );
    }

    #[test]
    fn test_write_csv_custom_options() {
        let schema = Schema::new(vec![
            Field::new("c1", DataType::Utf8, false),
            Field::new("c2", DataType::Float64, true),
            Field::new("c3", DataType::UInt32, false),
            Field::new("c4", DataType::Boolean, true),
            Field::new("c6", DataType::Time32(TimeUnit::Second), false),
        ]);

        let c1 = StringArray::from(vec![
            "Lorem ipsum dolor sit amet",
            "consectetur adipiscing elit",
            "sed do eiusmod tempor",
        ]);
        let c2 = PrimitiveArray::<Float64Type>::from(vec![
            Some(123.564532),
            None,
            Some(-556132.25),
        ]);
        let c3 = PrimitiveArray::<UInt32Type>::from(vec![3, 2, 1]);
        let c4 = PrimitiveArray::<BooleanType>::from(vec![Some(true), Some(false), None]);
        let c6 = Time32SecondArray::from(vec![1234, 24680, 85563]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(c1),
                Arc::new(c2),
                Arc::new(c3),
                Arc::new(c4),
                Arc::new(c6),
            ],
        )
        .unwrap();

        let file = get_temp_file("custom_options.csv", &[]);

        let builder = WriterBuilder::new()
            .has_headers(false)
            .with_delimiter(b'|')
            .with_time_format("%r".to_string());
        let mut writer = builder.build(file);
        let batches = vec![&batch];
        for batch in batches {
            writer.write(batch).unwrap();
        }

        // check that file was written successfully
        let mut file = File::open("target/debug/testdata/custom_options.csv").unwrap();
        let mut buffer: Vec<u8> = vec![];
        file.read_to_end(&mut buffer).unwrap();

        assert_eq!(
            "Lorem ipsum dolor sit amet|123.564532|3|true|12:20:34 AM\nconsectetur adipiscing elit||2|false|06:51:20 AM\nsed do eiusmod tempor|-556132.25|1||11:46:03 PM\n"
            .to_string(),
            String::from_utf8(buffer).unwrap()
        );
    }

    #[test]
    fn test_export_csv_string() {
        let schema = Schema::new(vec![
            Field::new("c1", DataType::Utf8, false),
            Field::new("c2", DataType::Float64, true),
            Field::new("c3", DataType::UInt32, false),
            Field::new("c4", DataType::Boolean, true),
            Field::new("c5", DataType::Timestamp(TimeUnit::Millisecond, None), true),
            Field::new("c6", DataType::Time32(TimeUnit::Second), false),
        ]);

        let c1 = StringArray::from(vec![
            "Lorem ipsum dolor sit amet",
            "consectetur adipiscing elit",
            "sed do eiusmod tempor",
        ]);
        let c2 = PrimitiveArray::<Float64Type>::from(vec![
            Some(123.564532),
            None,
            Some(-556132.25),
        ]);
        let c3 = PrimitiveArray::<UInt32Type>::from(vec![3, 2, 1]);
        let c4 = PrimitiveArray::<BooleanType>::from(vec![Some(true), Some(false), None]);
        let c5 = TimestampMillisecondArray::from_opt_vec(
            vec![None, Some(1555584887378), Some(1555555555555)],
            None,
        );
        let c6 = Time32SecondArray::from(vec![1234, 24680, 85563]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(c1),
                Arc::new(c2),
                Arc::new(c3),
                Arc::new(c4),
                Arc::new(c5),
                Arc::new(c6),
            ],
        )
        .unwrap();

        let sw = StringWriter::new();
        let mut writer = Writer::new(sw);
        let batches = vec![&batch, &batch];
        for batch in batches {
            writer.write(batch).unwrap();
        }

        let left = "c1,c2,c3,c4,c5,c6
Lorem ipsum dolor sit amet,123.564532,3,true,,00:20:34
consectetur adipiscing elit,,2,false,2019-04-18T10:54:47.378000000,06:51:20
sed do eiusmod tempor,-556132.25,1,,2019-04-18T02:45:55.555000000,23:46:03
Lorem ipsum dolor sit amet,123.564532,3,true,,00:20:34
consectetur adipiscing elit,,2,false,2019-04-18T10:54:47.378000000,06:51:20
sed do eiusmod tempor,-556132.25,1,,2019-04-18T02:45:55.555000000,23:46:03\n";
        let right = writer.writer.into_inner().map(|s| s.to_string());
        assert_eq!(Some(left.to_string()), right.ok());
    }
}
