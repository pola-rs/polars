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

//! Binary file to read data from a Parquet file.
//!
//! # Install
//!
//! `parquet-read` can be installed using `cargo`:
//! ```
//! cargo install parquet
//! ```
//! After this `parquet-read` should be globally available:
//! ```
//! parquet-read XYZ.parquet
//! ```
//!
//! The binary can also be built from the source code and run as follows:
//! ```
//! cargo run --bin parquet-read XYZ.parquet
//! ```
//!
//! # Usage
//!
//! ```
//! parquet-read <file-path> [num-records]
//! ```
//! where `file-path` is the path to a Parquet file and `num-records` is the optional
//! numeric option that allows to specify number of records to read from a file.
//! When not provided, all records are read.
//!
//! Note that `parquet-read` reads full file schema, no projection or filtering is
//! applied.

extern crate parquet;

use std::{env, fs::File, path::Path, process};

use parquet::file::reader::{FileReader, SerializedFileReader};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 && args.len() != 3 {
        println!("Usage: parquet-read <file-path> [num-records]");
        process::exit(1);
    }

    let mut num_records: Option<usize> = None;
    if args.len() == 3 {
        match args[2].parse() {
            Ok(value) => num_records = Some(value),
            Err(e) => panic!("Error when reading value for [num-records], {}", e),
        }
    }

    let path = Path::new(&args[1]);
    let file = File::open(&path).unwrap();
    let parquet_reader = SerializedFileReader::new(file).unwrap();

    // Use full schema as projected schema
    let mut iter = parquet_reader.get_row_iter(None).unwrap();

    let mut start = 0;
    let end = num_records.unwrap_or(0);
    let all_records = num_records.is_none();

    while all_records || start < end {
        match iter.next() {
            Some(row) => println!("{}", row),
            None => break,
        }
        start += 1;
    }
}
