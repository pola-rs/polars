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

//! [Apache Arrow](http://arrow.apache.org/) is a cross-language development platform for
//! in-memory data.
//!
//! This mod provides API for converting between some and parquet.
//!
//! # Example of reading parquet file into some record batch
//!
//! ```rust, no_run
//! use some::record_batch::RecordBatchReader;
//! use parquet::file::reader::SerializedFileReader;
//! use parquet::some::{ParquetFileArrowReader, ArrowReader};
//! use std::rc::Rc;
//! use std::fs::File;
//!
//! let file = File::open("parquet.file").unwrap();
//! let file_reader = SerializedFileReader::new(file).unwrap();
//! let mut arrow_reader = ParquetFileArrowReader::new(Rc::new(file_reader));
//!
//! println!("Converted some schema is: {}", arrow_reader.get_schema().unwrap());
//! println!("Arrow schema after projection is: {}",
//!    arrow_reader.get_schema_by_columns(vec![2, 4, 6]).unwrap());
//!
//! let mut record_batch_reader = arrow_reader.get_record_reader(2048).unwrap();
//!
//! loop {
//!    let record_batch = record_batch_reader.next_batch().unwrap().unwrap();
//!    if record_batch.num_rows() > 0 {
//!        println!("Read {} records.", record_batch.num_rows());
//!    } else {
//!        println!("End of file!");
//!    }
//!}
//! ```

pub(in crate::arrow) mod array_reader;
pub mod arrow_reader;
pub(in crate::arrow) mod converter;
pub(in crate::arrow) mod record_reader;
pub mod schema;

pub use self::arrow_reader::ArrowReader;
pub use self::arrow_reader::ParquetFileArrowReader;
pub use self::schema::{parquet_to_arrow_schema, parquet_to_arrow_schema_by_columns};
