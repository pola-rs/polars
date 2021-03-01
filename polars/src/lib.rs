//! # Polars: *<small>DataFrames in Rust</small>*
//!
//! Polars is a DataFrame library for Rust. It is based on [Apache Arrows](https://arrow.apache.org/) memory model.
//! This means that operations on Polars array's *(called `Series` or `ChunkedArray<T>` {if the type `T` is known})* are
//! optimally aligned cache friendly operations and SIMD.
//!
//! Polars supports an eager and a lazy api. The eager api is similar to [pandas](https://pandas.pydata.org/),
//! the lazy api is similar to [Spark](https://spark.apache.org/).
//!
//! ### Eager
//! Read more in the pages of the following data structures /traits.
//!
//! * [DataFrame struct](crate::frame::DataFrame)
//! * [Series struct](crate::series::Series)
//! * [Series trait](crate::series::SeriesTrait)
//! * [ChunkedArray struct](crate::chunked_array::ChunkedArray)
//!
//! ### Lazy
//! Read more in the [lazy](polars_lazy) module
//!
//! ## Read and write CSV/ JSON
//!
//! ```
//! use polars::prelude::*;
//! use std::fs::File;
//!
//! fn example() -> Result<DataFrame> {
//!     let file = File::open("iris.csv")
//!                     .expect("could not read file");
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
//! * [the csv module](polars_io::csv)
//! * [the json module](polars_io::json)
//! * [the IPC module](polars_io::ipc)
//! * [the parquet module](polars_io::parquet)
//!
//! ## Joins
//!
//! ```
//! # #[macro_use] extern crate polars;
//! # fn main() {
//! use polars::prelude::*;
//!
//! fn join() -> Result<DataFrame> {
//!     // Create first df.
//!     let temp = df!("days" => &[0, 1, 2, 3, 4],
//!                    "temp" => &[22.1, 19.9, 7., 2., 3.])?;
//!
//!     // Create second df.
//!     let rain = df!("days" => &[1, 2],
//!                    "rain" => &[0.1, 0.2])?;
//!
//!     // Left join on days column.
//!     temp.left_join(&rain, "days", "days")
//! }
//!
//! println!("{}", join().unwrap())
//! # }
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
//! ## Groupby's | aggregations | pivots | melts
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
//! let s = Series::new("foo", [1, 2, 3]);
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
//! let s = Series::new("dollars", &[1, 2, 3]);
//! let mask = s.eq(1);
//!
//! assert_eq!(Vec::from(mask), &[Some(true), Some(false), Some(false)]);
//! ```
//!
//! ## Temporal data types
//!
//! ```rust
//! # use polars::prelude::*;
//! let dates = &[
//! "2020-08-21",
//! "2020-08-21",
//! "2020-08-22",
//! "2020-08-23",
//! "2020-08-22",
//! ];
//! // date format
//! let fmt = "%Y-%m-%d";
//! // create date series
//! let s0 = Date32Chunked::parse_from_str_slice("date", dates, fmt)
//!         .into_series();
//! ```
//!
//!
//! ## And more...
//!
//! * [DataFrame](crate::frame::DataFrame)
//! * [Series](crate::series::Series)
//! * [ChunkedArray](crate::chunked_array::ChunkedArray)
//!     - [Operations implemented by Traits](crate::chunked_array::ops)
//! * [Time/ DateTime utilities](crate::doc::time)
//! * [Groupby, aggregations, pivots and melts](crate::frame::group_by::GroupBy)
//!
//! ## Features
//!
//! Additional cargo features:
//!
//! * `temporal (default)`
//!     - Conversions between Chrono and Polars for temporal data
//! * `simd (nightly only)`
//!     - SIMD operations
//! * `parquet`
//!     - Read Apache Parquet format
//! * `json`
//!     - Json serialization
//! * `ipc`
//!     - Arrow's IPC format serialization
//! * `random`
//!     - Generate array's with randomly sampled values
//! * `ndarray`
//!     - Convert from `DataFrame` to `ndarray`
//! * `parallel`
//!     - ChunkedArrays can be used by rayon::par_iter()
//! * `lazy`
//!     - Lazy api
//! * `strings`
//!     - String utilities for `Utf8Chunked`
//! * `object`
//!     - Support for generic ChunkedArray's called `ObjectChunked<T>` (generic over `T`).
//!       These will downcastable from Series through the [Any](https://doc.rust-lang.org/std/any/index.html) trait.
pub mod prelude;
pub use polars_core::{
    chunked_array, datatypes, doc, error, frame, functions, series, testing, toggle_string_cache,
};

pub use polars_core::apply_method_all_arrow_series;
pub use polars_core::df;

pub use polars_io as io;
#[cfg(feature = "lazy")]
pub use polars_lazy as lazy;

#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
