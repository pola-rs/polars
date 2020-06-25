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

//! Binary file to return the number of rows found from Parquet file(s).
//!
//! # Install
//!
//! `parquet-rowcount` can be installed using `cargo`:
//! ```
//! cargo install parquet
//! ```
//! After this `parquet-rowcount` should be globally available:
//! ```
//! parquet-rowcount XYZ.parquet
//! ```
//!
//! The binary can also be built from the source code and run as follows:
//! ```
//! cargo run --bin parquet-rowcount XYZ.parquet ABC.parquet ZXC.parquet
//! ```
//!
//! # Usage
//!
//! ```
//! parquet-rowcount <file-path> ...
//! ```
//! where `file-path` is the path to a Parquet file and `...` is any additional number of
//! parquet files to count the number of rows from.
//!
//! Note that `parquet-rowcount` reads full file schema, no projection or filtering is
//! applied.

extern crate parquet;

use std::{env, fs::File, path::Path, process};

use parquet::file::reader::{FileReader, SerializedFileReader};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: parquet-rowcount <file-path> ...");
        process::exit(1);
    }

    for i in 1..args.len() {
        let filename = args[i].clone();
        let path = Path::new(&filename);
        let file = File::open(&path).unwrap();
        let parquet_reader = SerializedFileReader::new(file).unwrap();
        let row_group_metadata = parquet_reader.metadata().row_groups();
        let mut total_num_rows = 0;

        for group_metadata in row_group_metadata {
            total_num_rows += group_metadata.num_rows();
        }

        eprintln!("File {}: rowcount={}", filename, total_num_rows);
    }
}
