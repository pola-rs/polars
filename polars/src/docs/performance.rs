//! # Performance
//!
//! Understanding the memory format used by Arrow/ Polars can really increase performance of your
//! queries. This is especially true for large string data. The figure below shows how an Arrow UTF8
//! array is laid out in memory.
//!
//! The array `["foo", "bar", "ham"]` is encoded by
//!
//! * a concatenated string `"foobarham"`
//! * an offset array indicating the start (and end) of each string `[0, 2, 5, 8]`
//! * a null bitmap, indicating null values
//!
//! ![](https://raw.githubusercontent.com/ritchie46/img/master/polars/arrow/arrow_string.svg)
//!
//! This memory structure is very cache efficient if we are to read the string values. Especially if
//! we compare it to a `Vec<String>`.
//!
//! ![](https://raw.githubusercontent.com/ritchie46/img/master/polars/arrow/pandas_string.svg)
//!
//! However, if we need to reorder the Arrow UTF8 array, we need to swap around all the bytes of the
//! string values, which can become very expensive when we're dealing with large strings. On the
//! other hand, for the `Vec<String>`, we only need to swap pointers around which is only 8 bytes data
//! that have to be moved.
//!
//! If you have a [DataFrame](crate::frame::DataFrame) with a large number of
//! [Utf8Chunked](crate::datatypes::Utf8Chunked) columns and you need to reorder them due to an
//! operation like a FILTER, JOIN, GROUPBY, etc. than this can become quite expensive.
//!
//! ## Categorical type
//! For this reason Polars has a [CategoricalType](https://ritchie46.github.io/polars/polars/datatypes/struct.CategoricalType.html).
//! A `CategoricalChunked` is an array filled with `u32` values that each represent a unique string value.
//! Thereby maintaining cache-efficiency, whilst also making it cheap to move values around.
//!
//! ### Example: Single DataFrame
//!
//! In the example below we show how you can cast a `Utf8Chunked` column to a `CategoricalChunked`.
//!
//! ```rust
//! use polars::prelude::*;
//!
//! fn example(path: &str) -> Result<DataFrame> {
//!     let mut df = CsvReader::from_path(path)?
//!                 .finish()?;
//!
//!     df.may_apply("utf8-column", |s| s.cast::<CategoricalType>())?;
//!     Ok(df)
//! }
//!
//! ```
//!
//! ### Example: Eager join multiple DataFrames on a Categorical
//! When the strings of one column need to be joined with the string data from another `DataFrame`.
//! The `Categorical` data needs to be synchronized (Categories in df A need to point to the same
//! underlying string data as Categories in df B). You can do that by turning the global string cache
//! on.
//!
//! ```rust
//! use polars::prelude::*;
//! use polars::toggle_string_cache;
//!
//! fn example(mut df_a: DataFrame, mut df_b: DataFrame) -> Result<DataFrame> {
//!     // Set a global string cache
//!     toggle_string_cache(true);
//!
//!     df_a.may_apply("a", |s| s.cast::<CategoricalType>())?;
//!     df_b.may_apply("b", |s| s.cast::<CategoricalType>())?;
//!     df_a.join(&df_b, "a", "b", JoinType::Inner)
//! }
//! ```
//!
//! ### Example: Lazy join multiple DataFrames on a Categorical
//! A lazy Query always has a global string cache (unless you opt-out) for the duration of that query (until `collect` is called).
//! The example below shows how you could join two DataFrames with Categorical types.
//!
//! ```rust
//! # #[cfg(feature = "lazy")]
//! # {
//! use polars::prelude::*;
//!
//! fn lazy_example(mut df_a: LazyFrame, mut df_b: LazyFrame) -> Result<DataFrame> {
//!
//!     let q1 = df_a.with_columns(vec![
//!         col("a").cast(DataType::Categorical),
//!     ]);
//!
//!     let q2 = df_b.with_columns(vec![
//!         col("b").cast(DataType::Categorical)
//!     ]);
//!     q1.inner_join(q2, col("a"), col("b"), None).collect()
//! }
//! # }
//! ```
