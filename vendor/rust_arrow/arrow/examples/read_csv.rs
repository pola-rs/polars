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

use std::fs::File;
use std::sync::Arc;

use arrow::csv;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::Result;
#[cfg(feature = "prettyprint")]
use arrow::util::pretty::print_batches;

fn main() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("city", DataType::Utf8, false),
        Field::new("lat", DataType::Float64, false),
        Field::new("lng", DataType::Float64, false),
    ]);

    let file = File::open("test/data/uk_cities.csv").unwrap();

    let mut csv = csv::Reader::new(file, Arc::new(schema), false, None, 1024, None);
    let _batch = csv.next().unwrap().unwrap();
    #[cfg(feature = "prettyprint")]
    {
        print_batches(&vec![_batch]).unwrap();
    }
    Ok(())
}
