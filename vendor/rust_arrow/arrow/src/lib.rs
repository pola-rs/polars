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

//! A native Rust implementation of [Apache Arrow](https://arrow.apache.org), a cross-language
//! development platform for in-memory data.
//!
//! Currently the project is developed and tested against nightly Rust. To learn more
//! about the status of Arrow in Rust, see `README.md`.

#![feature(specialization)]
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(bare_trait_objects)]

pub mod array;
pub mod bitmap;
pub mod buffer;
pub mod compute;
pub mod csv;
pub mod datatypes;
pub mod error;
// #[cfg(feature = "flight")]
// pub mod flight;
#[allow(clippy::redundant_closure)]
#[allow(clippy::needless_lifetimes)]
#[allow(clippy::extra_unused_lifetimes)]
#[allow(clippy::redundant_static_lifetimes)]
#[allow(clippy::redundant_field_names)]
pub mod ipc;
pub mod json;
pub mod memory;
pub mod record_batch;
pub mod tensor;
pub mod util;
