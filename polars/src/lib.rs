//! # Polars DataFrames in Rust
//!
//! # WIP
//!
//! Read more in the [DataFrame](frame/struct.DataFrame.html) and [Series](series/index.html)
//! modules.
//!
//! ## Read and write csv
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
//! For more examples see: [the csv module](frame/csv/index.html).
//!
//! ## Join
//!
//! ```
//! use polars::prelude::*;
//!
//! // Create first df.
//! let s0 = Series::new("days", [0, 1, 2, 3, 4].as_ref());
//! let s1 = Series::new("temp", [22.1, 19.9, 7., 2., 3.].as_ref());
//! let temp = DataFrame::new(vec![s0, s1]).unwrap();
//!
//! // Create second df.
//! let s0 = Series::new("days", [1, 2].as_ref());
//! let s1 = Series::new("rain", [0.1, 0.2].as_ref());
//! let rain = DataFrame::new(vec![s0, s1]).unwrap();
//!
//! // Left join on days column.
//! let joined = temp.left_join(&rain, "days", "days");
//! println!("{}", joined.unwrap())
//! ```
//!
//! ## GroupBy
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
//! ## Comparison
//!
//! ```
//! use polars::prelude::*;
//! use itertools::Itertools;
//! let s = Series::new("dollars", [1, 2, 3].as_ref());
//! let mask = s.eq(1);
//! let valid = [true, false, false].iter();
//! assert!(mask
//!     .into_iter()
//!     .map(|opt_bool| opt_bool.unwrap()) // option, because series can be null
//!     .zip(valid)
//!     .all(|(a, b)| a == *b))
//! ```
//!
#![allow(dead_code)]
#![feature(iterator_fold_self)]
pub mod chunked_array;
pub mod datatypes;
pub mod error;
mod fmt;
pub mod frame;
pub mod prelude;
pub mod series;
pub mod testing;
