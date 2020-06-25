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

//! Common Parquet errors and macros.

use std::{cell, convert, io, result, str};

use arrow::error::ArrowError;
use quick_error::quick_error;
use snap;
use thrift;

quick_error! {
  /// Set of errors that can be produced during different operations in Parquet.
  #[derive(Debug, PartialEq)]
  pub enum ParquetError {
      /// General Parquet error.
      /// Returned when code violates normal workflow of working with Parquet files.
      General(message: String) {
          display("Parquet error: {}", message)
              from(e: io::Error) -> (format!("underlying IO error: {}", e))
              from(e: snap::Error) -> (format!("underlying snap error: {}", e))
              from(e: thrift::Error) -> (format!("underlying Thrift error: {}", e))
              from(e: cell::BorrowMutError) -> (format!("underlying borrow error: {}", e))
              from(e: str::Utf8Error) -> (format!("underlying utf8 error: {}", e))
      }
      /// "Not yet implemented" Parquet error.
      /// Returned when functionality is not yet available.
      NYI(message: String) {
          display("NYI: {}", message)
      }
      /// "End of file" Parquet error.
      /// Returned when IO related failures occur, e.g. when there are not enough bytes to
      /// decode.
      EOF(message: String) {
          display("EOF: {}", message)
      }
      /// Arrow error.
      /// Returned when reading into some or writing from some.
      ArrowError(message:  String) {
          display("Arrow: {}", message)
              from(e: ArrowError) -> (format!("underlying Arrow error: {:?}", e))
      }
      IndexOutOfBound(index: usize, bound: usize) {
          display("Index {} out of bound: {}", index, bound)
      }
  }
}

/// A specialized `Result` for Parquet errors.
pub type Result<T> = result::Result<T, ParquetError>;

// ----------------------------------------------------------------------
// Conversion from `ParquetError` to other types of `Error`s

impl convert::From<ParquetError> for io::Error {
    fn from(e: ParquetError) -> Self {
        io::Error::new(io::ErrorKind::Other, e)
    }
}

// ----------------------------------------------------------------------
// Convenient macros for different errors

macro_rules! general_err {
    ($fmt:expr) => (ParquetError::General($fmt.to_owned()));
    ($fmt:expr, $($args:expr),*) => (ParquetError::General(format!($fmt, $($args),*)));
    ($e:expr, $fmt:expr) => (ParquetError::General($fmt.to_owned(), $e));
    ($e:ident, $fmt:expr, $($args:tt),*) => (
        ParquetError::General(&format!($fmt, $($args),*), $e));
}

macro_rules! nyi_err {
    ($fmt:expr) => (ParquetError::NYI($fmt.to_owned()));
    ($fmt:expr, $($args:expr),*) => (ParquetError::NYI(format!($fmt, $($args),*)));
}

macro_rules! eof_err {
    ($fmt:expr) => (ParquetError::EOF($fmt.to_owned()));
    ($fmt:expr, $($args:expr),*) => (ParquetError::EOF(format!($fmt, $($args),*)));
}

// ----------------------------------------------------------------------
// Convert parquet error into other errors

impl Into<ArrowError> for ParquetError {
    fn into(self) -> ArrowError {
        ArrowError::ParquetError(format!("{}", self))
    }
}
