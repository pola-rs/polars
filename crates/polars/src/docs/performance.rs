//! # Performance
//!
//! Understanding the memory format used by Arrow/Polars can really increase performance of your
//! queries. This is especially true for large string data. The figure below shows how an Arrow UTF8
//! array is laid out in memory.
//!
//! The array `["foo", "bar", "ham"]` is encoded by
//!
//! * a concatenated string `"foobarham"`
//! * an offset array indicating the start (and end) of each string `[0, 2, 5, 8]`
//! * a null bitmap, indicating null values
//!
//! ![](https://raw.githubusercontent.com/pola-rs/polars-static/master/docs/arrow-string.svg)
//!
//! This memory structure is very cache efficient if we are to read the string values. Especially if
//! we compare it to a [`Vec<String>`].
//!
//! ![](https://raw.githubusercontent.com/pola-rs/polars-static/master/docs/pandas-string.svg)
//!
//! However, if we need to reorder the Arrow UTF8 array, we need to swap around all the bytes of the
//! string values, which can become very expensive when we're dealing with large strings. On the
//! other hand, for the [`Vec<String>`], we only need to swap pointers around which is only 8 bytes data
//! that have to be moved.
//!
//! If you have a [`DataFrame`] with a large number of
//! [`StringChunked`] columns and you need to reorder them due to an
//! operation like a FILTER, JOIN, GROUPBY, etc. than this can become quite expensive.
//!
//! ## Categorical type
//! For this reason Polars has a [`CategoricalType`].
//! A [`CategoricalChunked`] is an array filled with `u32` values that each represent a unique string value.
//! Thereby maintaining cache-efficiency, whilst also making it cheap to move values around.
//!
//! [`DataFrame`]: crate::frame::DataFrame
//! [`StringChunked`]: crate::datatypes::StringChunked
//! [`CategoricalType`]: crate::datatypes::CategoricalType
//! [`CategoricalChunked`]: crate::datatypes::CategoricalChunked
//!
//! ### Example: Single DataFrame
//!
//! In the example below we show how you can cast a [`StringChunked`] column to a [`CategoricalChunked`].
//!
//! ```rust
//! use polars::prelude::*;
//!
//! fn example(path: &str) -> PolarsResult<DataFrame> {
//!     let mut df = CsvReader::from_path(path)?
//!                 .finish()?;
//!
//!     df.try_apply("utf8-column", |s| s.categorical().cloned())?;
//!     Ok(df)
//! }
//!
//! ```
//!
//! ### Example: Eager join multiple DataFrames on a Categorical
//! When the strings of one column need to be joined with the string data from another [`DataFrame`].
//! The [`Categorical`] data needs to be synchronized (Categories in df A need to point to the same
//! underlying string data as Categories in df B). You can do that by turning the global string cache
//! on.
//!
//! [`Categorical`]: crate::datatypes::CategoricalChunked
//!
//! ```rust
//! use polars::prelude::*;
//! use polars::enable_string_cache;
//!
//! fn example(mut df_a: DataFrame, mut df_b: DataFrame) -> PolarsResult<DataFrame> {
//!     // Set a global string cache
//!     enable_string_cache();
//!
//!     df_a.try_apply("a", |s| s.categorical().cloned())?;
//!     df_b.try_apply("b", |s| s.categorical().cloned())?;
//!     df_a.join(&df_b, ["a"], ["b"], JoinArgs::new(JoinType::Inner))
//! }
//! ```
//!
//! ### Example: Lazy join multiple DataFrames on a Categorical
//! A lazy Query always has a global string cache (unless you opt-out) for the duration of that query (until [`collect`] is called).
//! The example below shows how you could join two [`DataFrame`]s with [`Categorical`] types.
//!
//! [`collect`]: polars_lazy::frame::LazyFrame::collect
//!
//! ```rust
//! # #[cfg(feature = "lazy")]
//! # {
//! use polars::prelude::*;
//!
//! fn lazy_example(mut df_a: LazyFrame, mut df_b: LazyFrame) -> PolarsResult<DataFrame> {
//!
//!     let q1 = df_a.with_columns(vec![
//!         col("a").cast(DataType::Categorical(None)),
//!     ]);
//!
//!     let q2 = df_b.with_columns(vec![
//!         col("b").cast(DataType::Categorical(None))
//!     ]);
//!     q1.inner_join(q2, col("a"), col("b")).collect()
//! }
//! # }
//! ```
