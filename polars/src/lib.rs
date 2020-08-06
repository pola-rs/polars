//! # Polars DataFrames in Rust
//!
//! Read more in the [DataFrame](frame/struct.DataFrame.html), [Series](series/enum.Series.html), and
//! [ChunkedArray](chunked_array/struct.ChunkedArray.html) data structures.
//!
//! ## Read and write CSV/ JSON
//!
//! ```
//! use polars::prelude::*;
//! use std::fs::File;
//!
//! fn example() -> Result<DataFrame> {
//!     let file = File::open("iris.csv").expect("could not open file");
//!
//!     CsvReader::new(file)
//!             .infer_schema(None)
//!             .has_header(true)
//!             .finish()
//! }
//! ```
//!
//! For more IO examples see:
//!
//! * [the csv module](frame/ser/csv/index.html)
//! * [the json module](frame/ser/json/index.html)
//!
//!
//! ## Joins
//!
//! ```
//! use polars::prelude::*;
//!
//! // Create first df.
//! let s0 = Series::new("days", &[0, 1, 2, 3, 4]);
//! let s1 = Series::new("temp", &[22.1, 19.9, 7., 2., 3.]);
//! let temp = DataFrame::new(vec![s0, s1]).unwrap();
//!
//! // Create second df.
//! let s0 = Series::new("days", &[1, 2]);
//! let s1 = Series::new("rain", &[0.1, 0.2]);
//! let rain = DataFrame::new(vec![s0, s1]).unwrap();
//!
//! // Left join on days column.
//! let joined = temp.left_join(&rain, "days", "days");
//! println!("{}", joined.unwrap())
//! ```
//!
//! ```text
//! +------+------+------+
//! | days | temp | rain |
//! | ---  | ---  | ---  |
//! | i32  | f64  | f64  |
//! +======+======+======+
//! | 0    | 22.1 | null |
//! +------+------+------+
//! | 1    | 19.9 | 0.1  |
//! +------+------+------+
//! | 2    | 7    | 0.2  |
//! +------+------+------+
//! | 3    | 2    | null |
//! +------+------+------+
//! | 4    | 3    | null |
//! +------+------+------+
//! ```
//!
//! ## GroupBys
//!
//! ```
//! use polars::prelude::*;
//! fn groupby_sum(df: &DataFrame) -> Result<DataFrame> {
//!     df.groupby("column_name")?
//!     .select("agg_column_name")
//!     .sum()
//! }
//! ```
//!
//! ## Arithmetic
//! ```
//! use polars::prelude::*;
//! let s: Series = [1, 2, 3].iter().collect();
//! let s_squared = &s * &s;
//! ```
//!
//! ## Rust iterators
//!
//! ```
//! use polars::prelude::*;
//!
//! let s: Series = [1, 2, 3].iter().collect();
//! let s_squared: Series = s.i32()
//!      .expect("datatype mismatch")
//!      .into_iter()
//!      .map(|optional_v| {
//!          match optional_v {
//!              Some(v) => Some(v * v),
//!              None => None, // null value
//!          }
//!  }).collect();
//! ```
//!
//! ## Apply custom closures
//!
//! Besides running custom iterators, custom closures can be applied on the values of [ChunkedArray](chunked_array/struct.ChunkedArray.html)
//! by using the [apply](chunked_array/apply/trait.Apply.html) method. This method accepts
//! a closure that will be applied on all values of `Option<T>` that are non null. Note that this is the
//! **fastest** way to apply a custom closure on `ChunkedArray`'s.
//! ```
//! # use polars::prelude::*;
//! let s: Series = Series::new("values", [Some(1.0), None, Some(3.0)]);
//! // null values are ignored automatically
//! let squared = s.f64()
//!     .unwrap()
//!     .apply(|value| value.powf(2.0))
//!     .into_series();
//!
//! assert_eq!(Vec::from(squared.f64().unwrap()), &[Some(1.0), None, Some(9.0)])
//! ```
//!
//! ## Comparisons
//!
//! ```
//! use polars::prelude::*;
//! use itertools::Itertools;
//! let s = Series::new("dollars", &[1, 2, 3]);
//! let mask = s.eq(1);
//! let valid = [true, false, false].iter();
//!
//! assert_eq!(Vec::from(mask.bool().unwrap()), &[Some(true), Some(false), Some(false)]);
//! ```
//!
//! ## And more...
//!
//! * [DataFrame](frame/struct.DataFrame.html)
//! * [Series](series/enum.Series.html)
//! * [ChunkedArray](chunked_array/struct.ChunkedArray.html)
//!
//! ## Features
//!
//! Additional cargo features:
//!
//! * `pretty` (default)
//!     - pretty printing of DataFrames
//! * `simd`
//!     - SIMD operations
#![allow(dead_code)]
#![feature(iterator_fold_self)]
#[macro_use]
pub mod series;
#[macro_use]
pub(crate) mod utils;
pub mod chunked_array;
pub mod datatypes;
pub mod error;
mod fmt;
pub mod frame;
pub mod prelude;
pub mod testing;
