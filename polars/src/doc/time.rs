//! # DateTime related functionality
//!
//! Polars supports all data types in Arrow related to time and dates in any kind.
//! In Arrow times and dates are stored as i32 or i64 integers. This can represent for instance
//! a duration in seconds since the *Unix Epoch: 00:00:00 1 january 1970*.
//!
//! ## Chrono
//! I can imagine that integer values aren't the most intuitive when dealing with dates and times.
//! For this reason, Polars supports conversion from **chrono's** [NaiveTime](https://docs.rs/chrono/0.4.13/chrono/naive/struct.NaiveTime.html)
//! and [NaiveDate](https://docs.rs/chrono/0.4.13/chrono/naive/struct.NaiveDate.html) structs to
//! Polars and vice versa.
//!
//! `ChunkedArray<T>`'s initialization is represented by the the following traits:
//! * [FromNaiveTime](../../chunked_array/temporal/trait.FromNaiveTime.html)
//! * [FromNaiveDateTime](../../chunked_array/temporal/trait.FromNaiveDateTime.html)
//!
//! To cast a `ChunkedArray<T>` to a `Vec<NaiveTime>` or a `Vec<NaiveDateTime>` see:
//! * [AsNaiveTime](../../chunked_array/temporal/trait.AsNaiveTime.html)
//!
//! ### Example
//!
//! ```rust
//! use chrono::NaiveTime;
//! use polars::prelude::*;;
//!
//! // We can create a ChunkedArray from NaiveTime objects
//! fn from_naive_time_to_time32(time_values: &[NaiveTime]) -> Time32SecondChunked {
//!     Time32SecondChunked::new_from_naive_time("name", time_values)
//! }
//!
//! // Or from a ChunkedArray to NaiveTime objects
//! fn from_time32_to_naive_time(ca: &Time32SecondChunked) -> Vec<Option<NaiveTime>> {
//!     ca.as_naive_time()
//! }
//! ```
//!
//! ## String formatting
//!
//! We can also directly parse strings given a predifined `fmt: &str`. This uses **chrono's**
//! [NaiveTime::parse_from_str](https://docs.rs/chrono/0.4.15/chrono/naive/struct.NaiveTime.html#method.parse_from_str)
//! under the hood. So look there for the different formatting options. If the string parsing is not
//! succesful, the value will be `None`.
//!
//! ### Examples
//!
//! #### NaiveTime
//!
//! ```rust
//! use polars::prelude::*;
//! use chrono::NaiveTime;
//!
//! // String values to parse, Note that the 2nd value is not a correct time value.
//! let time_values = &["23:56:04", "26:00:99", "12:39:08"];
//! // Parsing fmt
//! let fmt = "%H:%M:%S";
//! // Create the ChunkedArray
//! let ca = Time64NanosecondChunked::parse_from_str_slice("name", time_values, fmt);
//!
//! // Assert that we've got a ChunkedArray with a single None value.
//! assert_eq!(ca.as_naive_time(),
//!     &[NaiveTime::parse_from_str(time_values[0], fmt).ok(),
//!     None,
//!     NaiveTime::parse_from_str(time_values[2], fmt).ok()]);
//! ```
//! #### NaiveDateTime
//!
//! ```rust
//! use polars::prelude::*;
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
//! let ca = Date32Chunked::parse_from_str_slice("name", datetime_values, fmt);
//!
//! // Or collect into a Vec<NaiveDateTime>
//! let vec = ca.as_naive_datetime();
//! ```
//!
//! ## Temporal Data Types
//! Polars supports all the time datatypes supported by Apache Arrow. These store the time values in
//! different precisions and bytes:
//!
//! ### Time
//! * **Time64NanosecondChunked**
//!     - A ChunkedArray which stores time with nanosecond precision as 64 bit signed integer.
//! * **Time64MicrosecondChunked**
//!     - A ChunkedArray which stores time with microsecond precision as 64 bit signed integer.
//! * **Time32MillisecondChunked**
//!     - A ChunkedArray which stores time with millisecond precision as 32 bit signed integer.
//! * **Time32SecondChunked**
//!     - A ChunkedArray which stores time with second precision as 32 bit signed integer.
//!
//! ### Date
//! * **Date32Chunked**
//!     - A ChunkedArray storing the elapsed days since the Unix Epoch as 32 bit signed integer.
//! * **Date64Chunked**
//!     - A ChunkedArray storing the elapsed milliseconds since the Unix Epoch as 64 bit signed integer.
