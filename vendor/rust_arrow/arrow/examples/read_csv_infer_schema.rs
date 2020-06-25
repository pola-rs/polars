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

extern crate arrow;

use arrow::csv;
use arrow::error::Result;
#[cfg(feature = "prettyprint")]
use arrow::util::pretty::print_batches;
use std::fs::File;

fn main() -> Result<()> {
    let file = File::open("test/data/uk_cities_with_headers.csv").unwrap();
    let builder = csv::ReaderBuilder::new()
        .has_header(true)
        .infer_schema(Some(100));
    let mut csv = builder.build(file).unwrap();
    let _batch = csv.next().unwrap().unwrap();
    #[cfg(feature = "prettyprint")]
    {
        print_batches(&vec![_batch]).unwrap();
    }
    Ok(())
}
