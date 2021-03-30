//! # DateTime related functionality
//!
//! Polars supports most data types in Arrow related to time and dates.
//! The datatypes that have most utility support are:
//!
//! * Date32
//!     - A *Date* object representing the time in days since the unix epoch.
//!     - Chrono support type: [NaiveDate](https://docs.rs/chrono/0.4.13/chrono/naive/struct.NaiveDate.html)
//!     - Underlying data type: `i32`
//! * Date64
//!     - A *DateTime* object representing the time in milliseconds since the unix epoch.
//!     - Chrono support type: [NaiveDateTime](https://docs.rs/chrono/0.4.13/chrono/naive/struct.NaiveDateTime.html)
//!     - Underlying data type: `i64`
//!
//! ## Utility methods
//! Given a `Date32` or `Date64` `ChunkedArray` one can extract various temporal information
//!
//! ## Chrono
//!
//! Polars has interopatibilty with the `chrono` library.
//!
//! ### Example
//!
//! ```rust
//! use chrono::NaiveTime;
//! use polars_core::prelude::*;;
//!
//! // We can create a ChunkedArray from NaiveTime objects
//! fn from_naive_time_to_time64(time_values: &[NaiveTime]) -> Time64NanosecondChunked {
//!     Time64NanosecondChunked::new_from_naive_time("name", time_values)
//! }
//!
//! // Or from a ChunkedArray to NaiveTime objects
//! fn from_time64_to_naive_time(ca: &Time64NanosecondChunked) -> Vec<Option<NaiveTime>> {
//!     ca.as_naive_time()
//! }
//! ```
//!
//! ## String formatting
//!
//! We can also directly parse strings given a predefined `fmt: &str`. This uses **chrono's**
//! [NaiveTime::parse_from_str](https://docs.rs/chrono/0.4.15/chrono/naive/struct.NaiveTime.html#method.parse_from_str)
//! under the hood. So look there for the different formatting options. If the string parsing is not
//! successful, the value will be `None`.
//!
//! ### Examples
//!
//! #### Parsing from Utf8Chunked
//!
//! ```rust
//! use polars_core::prelude::*;
//! use chrono::NaiveTime;
//!
//! // String values to parse, Note that the 2nd value is not a correct time value.
//! let time_values = &["23:56:04", "26:00:99", "12:39:08"];
//! // Parsing fmt
//! let fmt = "%H:%M:%S";
//! // Create the ChunkedArray
//! let ca = Utf8Chunked::new_from_slice("time", time_values);
//! // Parse strings as DateTime objects
//! let date_ca = ca.as_date64(Some(fmt));
//! ```
//! #### Parsing directly from slice
//!
//! ```rust
//! use polars_core::prelude::*;
//! use chrono::NaiveDateTime;
//!
//! // String values to parse, Note that the 2nd value is not a correct time value.
//! let datetime_values = &[
//!             "1988-08-25 00:00:16",
//!             "2015-09-05 23:56:04",
//!             "2012-12-21 00:00:00",
//!         ];
//! // Parsing fmt
//! let fmt = "%Y-%m-%d %H:%M:%S";
//!
//! // Create the ChunkedArray
//! let ca = Date64Chunked::parse_from_str_slice("datetime as ms since Epoch", datetime_values, fmt);
//!
//! // or dates in different precision (days)
//! let ca = Date32Chunked::parse_from_str_slice("date as days since Epoch", datetime_values, fmt);
//! ```
//!
