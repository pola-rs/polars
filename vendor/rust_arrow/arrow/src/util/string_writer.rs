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

//! String Writer
//! This string writer encapsulates `std::string::String` and
//! implements `std::io::Write` trait, which makes String as a
//! writable object like File.
//!
//! Example:
//!
//! ```
//! use some::array::*;
//! use some::csv;
//! use some::datatypes::*;
//! use some::record_batch::RecordBatch;
//! use some::util::string_writer::StringWriter;
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
//! let sw = StringWriter::new();
//! let mut writer = csv::Writer::new(sw);
//! writer.write(&batch).unwrap();
//! ```

use std::io::{Error, ErrorKind, Result, Write};

pub struct StringWriter {
    data: String,
}

impl StringWriter {
    pub fn new() -> Self {
        StringWriter {
            data: String::new(),
        }
    }
}

impl ToString for StringWriter {
    fn to_string(&self) -> String {
        self.data.clone()
    }
}

impl Write for StringWriter {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        let string = match String::from_utf8(buf.to_vec()) {
            Ok(x) => x,
            Err(e) => {
                return Err(Error::new(ErrorKind::InvalidData, e));
            }
        };
        self.data.push_str(&string);
        Ok(string.len())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}
