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

//! Main entrypoint for working with Parquet API.
//!
//! Provides access to file and row group readers and writers, record API, metadata, etc.
//!
//! See [`reader::SerializedFileReader`](reader/struct.SerializedFileReader.html) or
//! [`writer::SerializedFileWriter`](writer/struct.SerializedFileWriter.html) for a
//! starting reference, [`metadata::ParquetMetaData`](metadata/index.html) for file
//! metadata, and [`statistics`](statistics/index.html) for working with statistics.
//!
//! # Example of writing a new file
//!
//! ```rust,no_run
//! use std::{fs, path::Path, rc::Rc};
//!
//! use parquet::{
//!     file::{
//!         properties::WriterProperties,
//!         writer::{FileWriter, SerializedFileWriter},
//!     },
//!     schema::parser::parse_message_type,
//! };
//!
//! let path = Path::new("/path/to/sample.parquet");
//!
//! let message_type = "
//!   message schema {
//!     REQUIRED INT32 b;
//!   }
//! ";
//! let schema = Rc::new(parse_message_type(message_type).unwrap());
//! let props = Rc::new(WriterProperties::builder().build());
//! let file = fs::File::create(&path).unwrap();
//! let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
//! let mut row_group_writer = writer.next_row_group().unwrap();
//! while let Some(mut col_writer) = row_group_writer.next_column().unwrap() {
//!     // ... write values to a column writer
//!     row_group_writer.close_column(col_writer).unwrap();
//! }
//! writer.close_row_group(row_group_writer).unwrap();
//! writer.close().unwrap();
//!
//! let bytes = fs::read(&path).unwrap();
//! assert_eq!(&bytes[0..4], &[b'P', b'A', b'R', b'1']);
//! ```
//! # Example of reading an existing file
//!
//! ```rust,no_run
//! use parquet::file::reader::{FileReader, SerializedFileReader};
//! use std::{fs::File, path::Path};
//!
//! let path = Path::new("/path/to/sample.parquet");
//! if let Ok(file) = File::open(&path) {
//!     let file = File::open(&path).unwrap();
//!     let reader = SerializedFileReader::new(file).unwrap();
//!
//!     let parquet_metadata = reader.metadata();
//!     assert_eq!(parquet_metadata.num_row_groups(), 1);
//!
//!     let row_group_reader = reader.get_row_group(0).unwrap();
//!     assert_eq!(row_group_reader.num_columns(), 1);
//! }
//! ```
//! # Example of reading multiple files
//!
//! ```rust,no_run
//! use parquet::file::reader::SerializedFileReader;
//! use std::convert::TryFrom;
//!
//! let paths = vec![
//!     "/path/to/sample.parquet/part-1.snappy.parquet",
//!     "/path/to/sample.parquet/part-2.snappy.parquet"
//! ];
//! // Create a reader for each file and flat map rows
//! let rows = paths.iter()
//!     .map(|p| SerializedFileReader::try_from(*p).unwrap())
//!     .flat_map(|r| r.into_iter());
//!
//! for row in rows {
//!     println!("{}", row);
//! }
//! ```
pub mod metadata;
pub mod properties;
pub mod reader;
pub mod statistics;
pub mod writer;

const FOOTER_SIZE: usize = 8;
const PARQUET_MAGIC: [u8; 4] = [b'P', b'A', b'R', b'1'];
