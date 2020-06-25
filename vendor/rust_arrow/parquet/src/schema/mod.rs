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

//! Parquet schema definitions and methods to print and parse schema.
//!
//! # Example
//!
//! ```rust
//! use parquet::{
//!     basic::{LogicalType, Repetition, Type as PhysicalType},
//!     schema::{parser, printer, types::Type},
//! };
//! use std::rc::Rc;
//!
//! // Create the following schema:
//! //
//! // message schema {
//! //   OPTIONAL BYTE_ARRAY a (UTF8);
//! //   REQUIRED INT32 b;
//! // }
//!
//! let field_a = Type::primitive_type_builder("a", PhysicalType::BYTE_ARRAY)
//!     .with_logical_type(LogicalType::UTF8)
//!     .with_repetition(Repetition::OPTIONAL)
//!     .build()
//!     .unwrap();
//!
//! let field_b = Type::primitive_type_builder("b", PhysicalType::INT32)
//!     .with_repetition(Repetition::REQUIRED)
//!     .build()
//!     .unwrap();
//!
//! let schema = Type::group_type_builder("schema")
//!     .with_fields(&mut vec![Rc::new(field_a), Rc::new(field_b)])
//!     .build()
//!     .unwrap();
//!
//! let mut buf = Vec::new();
//!
//! // Print schema into buffer
//! printer::print_schema(&mut buf, &schema);
//!
//! // Parse schema from the string
//! let string_schema = String::from_utf8(buf).unwrap();
//! let parsed_schema = parser::parse_message_type(&string_schema).unwrap();
//!
//! assert_eq!(schema, parsed_schema);
//! ```

pub mod parser;
pub mod printer;
pub mod types;
pub mod visitor;
