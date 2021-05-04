//! # Polars: *<small>DataFrames in Rust</small>*
//!
//! Polars is a DataFrame library for Rust. It is based on [Apache Arrows](https://arrow.apache.org/) memory model.
//! Apache arrow provides very cache efficient columnar data structures, and is becoming the defacto
//! standard for columnar data.
//!
//! This means that Polars data structures can be shared zero copy with processes in many different
//! languages.
//!
//! ## 1. Data Structures
//! The base data structures provided by polars are `DataFrame`, `Series`, and `ChunkedArray<T>`.
//! We will provide a short top down view of these data structures.
//!
//! ### 1.1 DataFrame
//! A `DataFrame` is a 2 dimensional data structure that is backed by a `Series`, and it could be
//! seen as an abstraction on `Vec<Series>`. Operations that can be executed on `DataFrame`s are very
//! similar to what is done in a `SQL` like query. You can `GROUP`, `JOIN`, `PIVOT` etc. The closes
//! arrow equivalent to a `DataFrame` is a [RecordBatch](https://docs.rs/arrow/4.0.0/arrow/record_batch/struct.RecordBatch.html),
//! and Polars provides zero copy coercion.
//!
//! ### 1.2 Series
//! `Series` are the type agnostic columnar data representation of Polars. They provide many
//! operations out of the box, many via the [Series struct](crate::prelude::Series) and
//! [SeriesTrait trait](crate::series/trait.SeriesTrait.html). Whether or not an operation is provided
//! by a `Series` is determined by the operation. If the operation can be done without knowing the
//! underlying columnar type, this operation probably is provided by the `Series`. If not, you must
//! downcast to the typed data structure that is wrapped by the `Series`. That is the `ChunkedArray<T>`.
//!
//! ### 1.3 ChunkedArray
//! `ChunkedArray<T>` are wrappers around an arrow array, that can contain multiples chunks, e.g.
//! `Vec<dyn ArrowArray>`. These are the root data structures of Polars, and implement many operations.
//! Most operations are implemented by traits defined in [chunked_array::ops](crate::chunked_array::ops),
//! or on the [ChunkedArray struct](crate::chunked_array::ops).
//!
//! ## 2. SIMD
//! Polars / Arrow uses packed_simd to speed up kernels with SIMD operations. SIMD is an optional
//! `feature = "simd"`, and requires a nightly compiler. If you don't need SIMD, **Polars runs on stable!**
//!
//! ## 3. API
//! Polars supports an eager and a lazy API, and strives to make them both equally capable.
//! The eager API is similar to [pandas](https://pandas.pydata.org/), and is easy to get started.
//! The lazy API is similar to [Spark](https://spark.apache.org/), and builds a query plan that will
//! be optimized. This may be less intuitive, but you may gain of additional performance.
//!
//! ### 3.1 Eager
//! Read more in the pages of the following data structures /traits.
//!
//! * [DataFrame struct](crate::frame::DataFrame)
//! * [Series struct](crate::series::Series)
//! * [Series trait](crate::series::SeriesTrait)
//! * [ChunkedArray struct](crate::chunked_array::ChunkedArray)
//! * [ChunkedArray operations traits](crate::chunked_array::ops)
//!
//! ### 3.2 Lazy
//! Unlock full potential with lazy computation. This allows query optimizations and provides Polars
//! the full query context so that the fastest algorithm can be chosen.
//!
//! **[Read more in the lazy module.](polars_lazy)**
//!
//! ## 4. Compile times
//! A DataFrame library typically consists of
//!
//! * Tons of features
//! * A lot of datatypes
//!
//! Both of these really put large strains on compile times. To keep Polars lean, we make both **opt-in**,
//! meaning that you only pay the compilation cost, if you need it.
//!
//! ## 4.2 Compile times and opt-in featurs
//! The opt-in features are:
//!
//! * `pivot` - [pivot operation](crate::frame::groupby::GroupBy::pivot) on `DataFrame`s
//! * `random` - Generate array's with randomly sampled values
//! * `ndarray`- Convert from `DataFrame` to `ndarray`
//! * `downsample` - [downsample operation](crate::frame::DataFrame::downsample) on `DataFrame`s
//!
//! ## 4.3 Compile times and opt-in data types
//! As mentioned above, Polars `Series` are wrappers around
//! `ChunkedArray<T>` without the generic parameter `T`.
//! To get rid of the generic parameter, all the possible value of `T` are compiled
//! for `Series`. This gets more expensive the more types you want for a `Series`. In order to reduce
//! the compile times, we have decided to default to a minimal set of types and make more `Series` types
//! opt-in.
//!
//! Note that if you get strange compile time errors, you probably need to opt-in for that `Series` dtype.
//! The opt-in dtypes are:
//!
//! | data type               | feature flag      |
//! |-------------------------|-------------------|
//! | Time64NanoSecondType    | dtype-time64-ns   |
//! | DurationNanosecondType  | dtype-duration-ns |
//! | DurationMillisecondType | dtype-duration-ms |
//! | Date32Type              | dtype-date32      |
//! | Date64Type              | dtype-date64      |
//! | Int8Type                | dtype-i8          |
//! | Int16Type               | dtype-i16         |
//! | UInt8Type               | dtype-u8          |
//! | UInt16Type              | dtype-u16         |
//! | UInt64Type              | dtype-u64         |
//!
//!
//! Or you can choose on of the preconfigured pre-sets.
//!
//! * `dtype-full` - all opt-in dtypes.
//! * `dtype-slim` - slim preset of opt-in dtypes.
//!
//! ## 5. Performance and string data
//! Large string data can really slow down your queries.
//! Read more in the [performance section](crate::docs::performance)
//!
//! ## 6. Custom allocator
//! A DataFrame library naturally does a lot of heap allocations. It is recommended to use a custom
//! allocator. [Mimalloc](https://docs.rs/mimalloc/0.1.25/mimalloc/) for instance, shows a significant
//! performance gain in runtime as well as memory usage.
//!
//! ### Usage
//! ```ignore
//! use mimalloc::MiMalloc;
//!
//! #[global_allocator]
//! static GLOBAL: MiMalloc = MiMalloc;
//! ```
//!
//! ### Cargo.toml
//! ```ignore
//! [dependencies]
//! mimalloc = { version = "*", default-features = false }
//! ```
//!
//! ## 7. Examples
//! Below we show some minimal examples, most can be found in the provided `traits` and `structs`
//! documentation.
//!
//! ### Read and write CSV/ JSON
//!
//! ```
//! use polars::prelude::*;
//!
//! fn example() -> Result<DataFrame> {
//!     CsvReader::from_path("iris.csv")?
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
//! ### Joins
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
//! ### Groupby's | aggregations | pivots | melts
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
//! ### Arithmetic
//! The syntax required for arithmetic require understanding a few **gotcha's**. Due to the ownership
//! rules and because we don't want an operation such as a **multiply** or an **addition** takes
//! ownership of a Series, these need to be referenced when doing an arithmetic operation.
//! ```
//! use polars::prelude::*;
//! let s = Series::new("foo", [1, 2, 3]);
//! let s_squared = &s * &s;
//!
//! let s_twice = &s * 100;
//! ```
//!
//! Because Rusts Orphan Rule doesn't allow use to implement left side operations, we need to call
//! such operation directly.
//!
//! ```rust
//! # use polars::prelude::*;
//! let series = Series::new("foo", [1, 2, 3]);
//!
//! // 1 / s
//! let divide_one_by_s = 1.div(&series);
//!
//! // 1 - s
//! let subtract_one_by_s = 1.sub(&series);
//! ```
//!
//! For `ChunkedArray`s this left hand side operations can be done with the `apply` method.
//!
//! ```rust
//! # use polars::prelude::*;
//! let ca = UInt32Chunked::new_from_slice("foo", &[1, 2, 3]);
//!
//! // 1 / ca
//! let divide_one_by_ca = ca.apply(|rhs| 1 / rhs);
//! ```
//!
//! ### Rust iterators
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
//! ### Apply custom closures
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
//! ### Comparisons
//!
//! ```
//! use polars::prelude::*;
//! let s = Series::new("dollars", &[1, 2, 3]);
//! let mask = s.eq(1);
//!
//! assert_eq!(Vec::from(mask), &[Some(true), Some(false), Some(false)]);
//! ```
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
//! * `lazy`
//!     - Lazy api
//! * `strings`
//!     - String utilities for `Utf8Chunked`
//! * `object`
//!     - Support for generic ChunkedArray's called `ObjectChunked<T>` (generic over `T`).
//!       These will downcastable from Series through the [Any](https://doc.rust-lang.org/std/any/index.html) trait.
//!
//! ## User Guide
//!
//! If you want to read more, [check the User Guide](https://ritchie46.github.io/polars-book/).
pub mod docs;
pub mod prelude;

pub use polars_core::{
    chunked_array, datatypes, doc, error, frame, functions, series, testing, toggle_string_cache,
};

pub use polars_core::apply_method_all_arrow_series;
pub use polars_core::df;

#[cfg(feature = "polars-io")]
pub use polars_io as io;
#[cfg(feature = "lazy")]
pub use polars_lazy as lazy;
