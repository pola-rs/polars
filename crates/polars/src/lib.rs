//! # Polars: *<small>DataFrames in Rust</small>*
//!
//! Polars is a DataFrame library for Rust. It is based on [Apache Arrow](https://arrow.apache.org/)'s memory model.
//! Apache arrow provides very cache efficient columnar data structures and is becoming the defacto
//! standard for columnar data.
//!
//! ## Quickstart
//! We recommend to build your queries directly with [polars-lazy]. This allows you to combine
//! expression into powerful aggregations and column selections. All expressions are evaluated
//! in parallel and your queries are optimized just in time.
//!
//! [polars-lazy]: polars_lazy
//!
//! ```no_run
//! use polars::prelude::*;
//! # fn example() -> PolarsResult<()> {
//!
//! let lf1 = LazyFrame::scan_parquet("myfile_1.parquet", Default::default())?
//!     .group_by([col("ham")])
//!     .agg([
//!         // expressions can be combined into powerful aggregations
//!         col("foo")
//!             .sort_by([col("ham").rank(Default::default(), None)], [false])
//!             .last()
//!             .alias("last_foo_ranked_by_ham"),
//!         // every expression runs in parallel
//!         col("foo").cum_min(false).alias("cumulative_min_per_group"),
//!         // every expression runs in parallel
//!         col("foo").reverse().implode().alias("reverse_group"),
//!     ]);
//!
//! let lf2 = LazyFrame::scan_parquet("myfile_2.parquet", Default::default())?
//!     .select([col("ham"), col("spam")]);
//!
//! let df = lf1
//!     .join(lf2, [col("reverse")], [col("foo")], JoinArgs::new(JoinType::Left))
//!     // now we finally materialize the result.
//!     .collect()?;
//! # Ok(())
//! # }
//! ```
//!
//! This means that Polars data structures can be shared zero copy with processes in many different
//! languages.
//!
//! ## Tree Of Contents
//!
//! * [Cookbooks](#cookbooks)
//! * [Data structures](#data-structures)
//!     - [DataFrame](#dataframe)
//!     - [Series](#series)
//!     - [ChunkedArray](#chunkedarray)
//! * [SIMD](#simd)
//! * [API](#api)
//! * [Expressions](#expressions)
//! * [Compile times](#compile-times)
//! * [Performance](#performance-and-string-data)
//!     - [Custom allocator](#custom-allocator)
//! * [Config](#config-with-env-vars)
//! * [User guide](#user-guide)
//!
//! ## Cookbooks
//! See examples in the cookbooks:
//!
//! * [Eager](crate::docs::eager)
//! * [Lazy](crate::docs::lazy)
//!
//! ## Data Structures
//! The base data structures provided by polars are [`DataFrame`], [`Series`], and [`ChunkedArray<T>`].
//! We will provide a short, top-down view of these data structures.
//!
//! [`DataFrame`]: crate::frame::DataFrame
//! [`Series`]: crate::series::Series
//! [`ChunkedArray<T>`]: crate::chunked_array::ChunkedArray
//!
//! ### DataFrame
//! A [`DataFrame`] is a 2 dimensional data structure that is backed by a [`Series`], and it could be
//! seen as an abstraction on [`Vec<Series>`]. Operations that can be executed on [`DataFrame`] are very
//! similar to what is done in a `SQL` like query. You can `GROUP`, `JOIN`, `PIVOT` etc.
//!
//! [`Vec<Series>`]: std::vec::Vec
//!
//! ### Series
//! [`Series`] are the type agnostic columnar data representation of Polars. They provide many
//! operations out of the box, many via the [`Series`] series and
//! [`SeriesTrait`] trait. Whether or not an operation is provided
//! by a [`Series`] is determined by the operation. If the operation can be done without knowing the
//! underlying columnar type, this operation probably is provided by the [`Series`]. If not, you must
//! downcast to the typed data structure that is wrapped by the [`Series`]. That is the [`ChunkedArray<T>`].
//!
//! [`SeriesTrait`]: crate::series::SeriesTrait
//!
//! ### ChunkedArray
//! [`ChunkedArray<T>`] are wrappers around an arrow array, that can contain multiples chunks, e.g.
//! [`Vec<dyn ArrowArray>`]. These are the root data structures of Polars, and implement many operations.
//! Most operations are implemented by traits defined in [chunked_array::ops],
//! or on the [`ChunkedArray`] struct.
//!
//! [`ChunkedArray`]: crate::chunked_array::ChunkedArray
//!
//! ## SIMD
//! Polars / Arrow uses packed_simd to speed up kernels with SIMD operations. SIMD is an optional
//! `feature = "nightly"`, and requires a nightly compiler. If you don't need SIMD, **Polars runs on stable!**
//!
//! ## API
//! Polars supports an eager and a lazy API. The eager API directly yields results, but is overall
//! more verbose and less capable of building elegant composite queries. We recommend to use the Lazy API
//! whenever you can.
//!
//! As neither API is async they should be wrapped in _spawn_blocking_ when used in an async context
//! to avoid blocking the async thread pool of the runtime.
//!
//! ## Expressions
//! Polars has a powerful concept called expressions.
//! Polars expressions can be used in various contexts and are a functional mapping of
//! `Fn(Series) -> Series`, meaning that they have [`Series`] as input and [`Series`] as output.
//! By looking at this functional definition, we can see that the output of an [`Expr`] also can serve
//! as the input of an [`Expr`].
//!
//! [`Expr`]: polars_lazy::dsl::Expr
//!
//! That may sound a bit strange, so lets give an example. The following is an expression:
//!
//! `col("foo").sort().head(2)`
//!
//! The snippet above says select column `"foo"` then sort this column and then take first 2 values
//! of the sorted output.
//! The power of expressions is that every expression produces a new expression and that they can
//! be piped together.
//! You can run an expression by passing them on one of polars execution contexts.
//! Here we run two expressions in the **select** context:
//!
//! ```no_run
//! # use polars::prelude::*;
//! # fn example() -> PolarsResult<()> {
//! # let df = DataFrame::default();
//!   df.lazy()
//!    .select([
//!        col("foo").sort(Default::default()).head(None),
//!        col("bar").filter(col("foo").eq(lit(1))).sum(),
//!    ])
//!    .collect()?;
//! # Ok(())
//! # }
//! ```
//! All expressions are ran in parallel, meaning that separate polars expressions are embarrassingly parallel.
//! (Note that within an expression there may be more parallelization going on).
//!
//! Understanding polars expressions is most important when starting with the polars library. Read more
//! about them in the [user guide](https://docs.pola.rs/user-guide/concepts/expressions).
//!
//! ### Eager
//! Read more in the pages of the following data structures /traits.
//!
//! * [DataFrame struct](crate::frame::DataFrame)
//! * [Series struct](crate::series::Series)
//! * [Series trait](crate::series::SeriesTrait)
//! * [ChunkedArray struct](crate::chunked_array::ChunkedArray)
//! * [ChunkedArray operations traits](crate::chunked_array::ops)
//!
//! ### Lazy
//! Unlock full potential with lazy computation. This allows query optimizations and provides Polars
//! the full query context so that the fastest algorithm can be chosen.
//!
//! **[Read more in the lazy module.](polars_lazy)**
//!
//! ## Compile times
//! A DataFrame library typically consists of
//!
//! * Tons of features
//! * A lot of datatypes
//!
//! Both of these really put strain on compile times. To keep Polars lean, we make both **opt-in**,
//! meaning that you only pay the compilation cost, if you need it.
//!
//! ## Compile times and opt-in features
//! The opt-in features are (not including dtype features):
//!
//! * `performant` - Longer compile times more fast paths.
//! * `lazy` - Lazy API
//!     - `regex` - Use regexes in [column selection]
//!     - `dot_diagram` - Create dot diagrams from lazy logical plans.
//! * `sql` - Pass SQL queries to polars.
//! * `streaming` - Be able to process datasets that are larger than RAM.
//! * `random` - Generate arrays with randomly sampled values
//! * `ndarray`- Convert from [`DataFrame`] to [ndarray](https://docs.rs/ndarray/)
//! * `temporal` - Conversions between [Chrono](https://docs.rs/chrono/) and Polars for temporal data types
//! * `timezones` - Activate timezone support.
//! * `strings` - Extra string utilities for [`StringChunked`] //!     - `string_pad` - `zfill`, `ljust`, `rjust`
//!     - `string_to_integer` - `parse_int`
//! * `object` - Support for generic ChunkedArrays called [`ObjectChunked<T>`] (generic over `T`).
//!              These are downcastable from Series through the [Any](https://doc.rust-lang.org/std/any/index.html) trait.
//! * Performance related:
//!     - `nightly` - Several nightly only features such as SIMD and specialization.
//!     - `performant` - more fast paths, slower compile times.
//!     - `bigidx` - Activate this feature if you expect >> 2^32 rows. This has not been needed by anyone.
//!                  This allows polars to scale up way beyond that by using `u64` as an index.
//!                  Polars will be a bit slower with this feature activated as many data structures
//!                  are less cache efficient.
//!     - `cse` - Activate common subplan elimination optimization
//! * IO related:
//!     - `serde` - Support for [serde](https://crates.io/crates/serde) serialization and deserialization.
//!                 Can be used for JSON and more serde supported serialization formats.
//!     - `serde-lazy` - Support for [serde](https://crates.io/crates/serde) serialization and deserialization.
//!                 Can be used for JSON and more serde supported serialization formats.
//!     - `parquet` - Read Apache Parquet format
//!     - `json` - JSON serialization
//!     - `ipc` - Arrow's IPC format serialization
//!     - `decompress` - Automatically infer compression of csvs and decompress them.
//!                      Supported compressions:
//!                         * zip
//!                         * gzip
//!
//! [`StringChunked`]: crate::datatypes::StringChunked
//! [column selection]: polars_lazy::dsl::col
//! [`ObjectChunked<T>`]: polars_core::datatypes::ObjectChunked
//!
//!
//! * [`DataFrame`] operations:
//!     - `dynamic_group_by` - Groupby based on a time window instead of predefined keys.
//!                           Also activates rolling window group by operations.
//!     - `sort_multiple` - Allow sorting a [`DataFrame`] on multiple columns
//!     - `rows` - Create [`DataFrame`] from rows and extract rows from [`DataFrame`]s.
//!                And activates `pivot` and `transpose` operations
//!     - `asof_join` - Join ASOF, to join on nearest keys instead of exact equality match.
//!     - `cross_join` - Create the Cartesian product of two [`DataFrame`]s.
//!     - `semi_anti_join` - SEMI and ANTI joins.
//!     - `row_hash` - Utility to hash [`DataFrame`] rows to [`UInt64Chunked`]
//!     - `diagonal_concat` - Concat diagonally thereby combining different schemas.
//!     - `dataframe_arithmetic` - Arithmetic on ([`Dataframe`] and [`DataFrame`]s) and ([`DataFrame`] on [`Series`])
//!     - `partition_by` - Split into multiple [`DataFrame`]s partitioned by groups.
//! * [`Series`]/[`Expr`] operations:
//!     - `is_in` - Check for membership in [`Series`].
//!     - `zip_with` - [Zip two Series/ ChunkedArrays](crate::chunked_array::ops::ChunkZip).
//!     - `round_series` - round underlying float types of [`Series`].
//!     - `repeat_by` - [Repeat element in an Array N times, where N is given by another array.
//!     - `is_first_distinct` - Check if element is first unique value.
//!     - `is_last_distinct` - Check if element is last unique value.
//!     - `is_between` - Check if this expression is between the given lower and upper bounds.
//!     - `checked_arithmetic` - checked arithmetic/ returning [`None`] on invalid operations.
//!     - `dot_product` - Dot/inner product on [`Series`] and [`Expr`].
//!     - `concat_str` - Concat string data in linear time.
//!     - `reinterpret` - Utility to reinterpret bits to signed/unsigned
//!     - `take_opt_iter` - Take from a [`Series`] with [`Iterator<Item=Option<usize>>`](std::iter::Iterator).
//!     - `mode` - [Return the most occurring value(s)](polars_ops::chunked_array::mode)
//!     - `cum_agg` - [`cum_sum`], [`cum_min`], [`cum_max`] aggregation.
//!     - `rolling_window` - rolling window functions, like [`rolling_mean`]
//!     - `interpolate` [interpolate None values](polars_ops::series::interpolate())
//!     - `extract_jsonpath` - [Run jsonpath queries on StringChunked](https://goessner.net/articles/JsonPath/)
//!     - `list` - List utils.
//!         - `list_gather` take sublist by multiple indices
//!     - `rank` - Ranking algorithms.
//!     - `moment` - kurtosis and skew statistics
//!     - `ewma` - Exponential moving average windows
//!     - `abs` - Get absolute values of [`Series`].
//!     - `arange` - Range operation on [`Series`].
//!     - `product` - Compute the product of a [`Series`].
//!     - `diff` - [`diff`] operation.
//!     - `pct_change` - Compute change percentages.
//!     - `unique_counts` - Count unique values in expressions.
//!     - `log` - Logarithms for [`Series`].
//!     - `list_to_struct` - Convert [`List`] to [`Struct`] dtypes.
//!     - `list_count` - Count elements in lists.
//!     - `list_eval` - Apply expressions over list elements.
//!     - `list_sets` - Compute UNION, INTERSECTION, and DIFFERENCE on list types.
//!     - `cumulative_eval` - Apply expressions over cumulatively increasing windows.
//!     - `arg_where` - Get indices where condition holds.
//!     - `search_sorted` - Find indices where elements should be inserted to maintain order.
//!     - `offset_by` - Add an offset to dates that take months and leap years into account.
//!     - `trigonometry` - Trigonometric functions.
//!     - `sign` - Compute the element-wise sign of a [`Series`].
//!     - `propagate_nans` - NaN propagating min/max aggregations.
//!     - `extract_groups` - Extract multiple regex groups from strings.
//!     - `cov` - Covariance and correlation functions.
//!     - `find_many` - Find/replace multiple string patterns at once.
//! * [`DataFrame`] pretty printing
//!     - `fmt` - Activate [`DataFrame`] formatting
//!
//! [`UInt64Chunked`]: crate::datatypes::UInt64Chunked
//! [`cum_sum`]: polars_ops::prelude::cum_sum
//! [`cum_min`]: polars_ops::prelude::cum_min
//! [`cum_max`]: polars_ops::prelude::cum_max
//! [`rolling_mean`]: crate::series::Series#method.rolling_mean
//! [`diff`]: polars_ops::prelude::diff
//! [`List`]: crate::datatypes::DataType::List
//! [`Struct`]: crate::datatypes::DataType::Struct
//!
//! ## Compile times and opt-in data types
//! As mentioned above, Polars [`Series`] are wrappers around
//! [`ChunkedArray<T>`] without the generic parameter `T`.
//! To get rid of the generic parameter, all the possible value of `T` are compiled
//! for [`Series`]. This gets more expensive the more types you want for a [`Series`]. In order to reduce
//! the compile times, we have decided to default to a minimal set of types and make more [`Series`] types
//! opt-in.
//!
//! Note that if you get strange compile time errors, you probably need to opt-in for that [`Series`] dtype.
//! The opt-in dtypes are:
//!
//! | data type               | feature flag      |
//! |-------------------------|-------------------|
//! | Date                    | dtype-date        |
//! | Datetime                | dtype-datetime    |
//! | Time                    | dtype-time        |
//! | Duration                | dtype-duration    |
//! | Int8                    | dtype-i8          |
//! | Int16                   | dtype-i16         |
//! | UInt8                   | dtype-u8          |
//! | UInt16                  | dtype-u16         |
//! | Categorical             | dtype-categorical |
//! | Struct                  | dtype-struct      |
//!
//!
//! Or you can choose on of the preconfigured pre-sets.
//!
//! * `dtype-full` - all opt-in dtypes.
//! * `dtype-slim` - slim preset of opt-in dtypes.
//!
//! ## Performance
//! To gains most performance out of Polars we recommend compiling on a nightly compiler
//! with the features `simd` and `performant` activated. The activated cpu features also influence
//! the amount of simd acceleration we can use.
//!
//! See this the features we activate for our python builds, or if you just run locally and want to
//! use all available features on your cpu, set `RUSTFLAGS='-C target-cpu=native'`.
//!
//! ### Custom allocator
//! An OLAP query engine does a lot of heap allocations. It is recommended to use a custom
//! allocator, (we have found this to have up to ~25% runtime influence).
//! [JeMalloc](https://crates.io/crates/jemallocator) and
//! [Mimalloc](https://crates.io/crates/mimalloc) for instance, show a significant
//! performance gain in runtime as well as memory usage.
//!
//! #### Jemalloc Usage
//! ```ignore
//! use jemallocator::Jemalloc;
//!
//! #[global_allocator]
//! static GLOBAL: Jemalloc = Jemalloc;
//! ```
//!
//! #### Cargo.toml
//! ```toml
//! [dependencies]
//! jemallocator = { version = "*" }
//! ```
//!
//! #### Mimalloc Usage
//!
//! ```ignore
//! use mimalloc::MiMalloc;
//!
//! #[global_allocator]
//! static GLOBAL: MiMalloc = MiMalloc;
//! ```
//!
//! #### Cargo.toml
//! ```toml
//! [dependencies]
//! mimalloc = { version = "*", default-features = false }
//! ```
//!
//! #### Notes
//! [Benchmarks](https://github.com/pola-rs/polars/pull/3108) have shown that on Linux and macOS JeMalloc
//! outperforms Mimalloc on all tasks and is therefore the default allocator used for the Python bindings on Unix platforms.
//!
//! ## Config with ENV vars
//!
//! * `POLARS_FMT_TABLE_FORMATTING` -> define styling of tables using any of the following options (default = UTF8_FULL_CONDENSED). These options are defined by comfy-table which provides examples for each at <https://github.com/Nukesor/comfy-table/blob/main/src/style/presets.rs>
//!   * `ASCII_FULL`
//!   * `ASCII_FULL_CONDENSED`
//!   * `ASCII_NO_BORDERS`
//!   * `ASCII_BORDERS_ONLY`
//!   * `ASCII_BORDERS_ONLY_CONDENSED`
//!   * `ASCII_HORIZONTAL_ONLY`
//!   * `ASCII_MARKDOWN`
//!   * `UTF8_FULL`
//!   * `UTF8_FULL_CONDENSED`
//!   * `UTF8_NO_BORDERS`
//!   * `UTF8_BORDERS_ONLY`
//!   * `UTF8_HORIZONTAL_ONLY`
//!   * `NOTHING`
//! * `POLARS_FMT_TABLE_CELL_ALIGNMENT` -> define cell alignment using any of the following options (default = LEFT):
//!   * `LEFT`
//!   * `CENTER`
//!   * `RIGHT`
//! * `POLARS_FMT_TABLE_DATAFRAME_SHAPE_BELOW` -> print shape information below the table.
//! * `POLARS_FMT_TABLE_HIDE_COLUMN_NAMES` -> hide table column names.
//! * `POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES` -> hide data types for columns.
//! * `POLARS_FMT_TABLE_HIDE_COLUMN_SEPARATOR` -> hide separator that separates column names from rows.
//! * `POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION"` -> omit table shape information.
//! * `POLARS_FMT_TABLE_INLINE_COLUMN_DATA_TYPE` -> put column data type on the same line as the column name.
//! * `POLARS_FMT_TABLE_ROUNDED_CORNERS` -> apply rounded corners to UTF8-styled tables.
//! * `POLARS_FMT_MAX_COLS` -> maximum number of columns shown when formatting DataFrames.
//! * `POLARS_FMT_MAX_ROWS` -> maximum number of rows shown when formatting DataFrames, `-1` to show all.
//! * `POLARS_FMT_STR_LEN` -> maximum number of characters printed per string value.
//! * `POLARS_TABLE_WIDTH` -> width of the tables used during DataFrame formatting.
//! * `POLARS_MAX_THREADS` -> maximum number of threads used to initialize thread pool (on startup).
//! * `POLARS_VERBOSE` -> print logging info to stderr.
//! * `POLARS_NO_PARTITION` -> polars may choose to partition the group_by operation, based on data
//!                            cardinality. Setting this env var will turn partitioned group_by's off.
//! * `POLARS_PARTITION_UNIQUE_COUNT` -> at which (estimated) key count a partitioned group_by should run.
//!                                          defaults to `1000`, any higher cardinality will run default group_by.
//! * `POLARS_FORCE_PARTITION` -> force partitioned group_by if the keys and aggregations allow it.
//! * `POLARS_ALLOW_EXTENSION` -> allows for [`ObjectChunked<T>`] to be used in arrow, opening up possibilities like using
//!                               `T` in complex lazy expressions. However this does require `unsafe` code allow this.
//! * `POLARS_NO_PARQUET_STATISTICS` -> if set, statistics in parquet files are ignored.
//! * `POLARS_PANIC_ON_ERR` -> panic instead of returning an Error.
//! * `POLARS_BACKTRACE_IN_ERR` -> include a Rust backtrace in Error messages.
//! * `POLARS_NO_CHUNKED_JOIN` -> force rechunk before joins.
//!
//! ## User guide
//!
//! If you want to read more, check the [user guide](https://docs.pola.rs/).
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![allow(ambiguous_glob_reexports)]
pub mod docs;
#[doc(hidden)]
pub mod export;
pub mod prelude;
#[cfg(feature = "sql")]
pub mod sql;

pub use polars_core::{
    apply_method_all_arrow_series, chunked_array, datatypes, df, error, frame, functions, series,
    testing,
};
#[cfg(feature = "dtype-categorical")]
pub use polars_core::{enable_string_cache, using_string_cache};
#[cfg(feature = "polars-io")]
pub use polars_io as io;
#[cfg(feature = "lazy")]
pub use polars_lazy as lazy;
#[cfg(feature = "temporal")]
pub use polars_time as time;

/// Polars crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
