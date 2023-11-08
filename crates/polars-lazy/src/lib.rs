//! Lazy API of Polars
//!
//! The lazy API of Polars supports a subset of the eager API. Apart from the distributed compute,
//! it is very similar to [Apache Spark](https://spark.apache.org/). You write queries in a
//! domain specific language. These queries translate to a logical plan, which represent your query steps.
//! Before execution this logical plan is optimized and may change the order of operations if this will increase performance.
//! Or implicit type casts may be added such that execution of the query won't lead to a type error (if it can be resolved).
//!
//! # Lazy DSL
//!
//! The lazy API of polars replaces the eager [`DataFrame`] with the [`LazyFrame`], through which
//! the lazy API is exposed.
//! The [`LazyFrame`] represents a logical execution plan: a sequence of operations to perform on a concrete data source.
//! These operations are not executed until we call [`collect`].
//! This allows polars to optimize/reorder the query which may lead to faster queries or fewer type errors.
//!
//! [`DataFrame`]: polars_core::frame::DataFrame
//! [`LazyFrame`]: crate::frame::LazyFrame
//! [`collect`]: crate::frame::LazyFrame::collect
//!
//! In general, a [`LazyFrame`] requires a concrete data source — a [`DataFrame`], a file on disk, etc. — which polars-lazy
//! then applies the user-specified sequence of operations to.
//! To obtain a [`LazyFrame`] from an existing [`DataFrame`], we call the [`lazy`](crate::frame::IntoLazy::lazy) method on
//! the [`DataFrame`].
//! A [`LazyFrame`] can also be obtained through the lazy versions of file readers, such as [`LazyCsvReader`](crate::frame::LazyCsvReader).
//!
//! The other major component of the polars lazy API is [`Expr`](crate::dsl::Expr), which represents an operation to be
//! performed on a [`LazyFrame`], such as mapping over a column, filtering, or groupby-aggregation.
//! [`Expr`] and the functions that produce them can be found in the [dsl module](crate::dsl).
//!
//! [`Expr`]: crate::dsl::Expr
//!
//! Most operations on a [`LazyFrame`] consume the [`LazyFrame`] and return a new [`LazyFrame`] with the updated plan.
//! If you need to use the same [`LazyFrame`] multiple times, you should [`clone`](crate::frame::LazyFrame::clone) it, and optionally
//! [`cache`](crate::frame::LazyFrame::cache) it beforehand.
//!
//! ## Examples
//!
//! #### Adding a new column to a lazy DataFrame
//!
//!```rust
//! #[macro_use] extern crate polars_core;
//! use polars_core::prelude::*;
//! use polars_lazy::prelude::*;
//!
//! let df = df! {
//!     "column_a" => &[1, 2, 3, 4, 5],
//!     "column_b" => &["a", "b", "c", "d", "e"]
//! }.unwrap();
//!
//! let new = df.lazy()
//!     // Note the reverse here!!
//!     .reverse()
//!     .with_column(
//!         // always rename a new column
//!         (col("column_a") * lit(10)).alias("new_column")
//!     )
//!     .collect()
//!     .unwrap();
//!
//! assert!(new.column("new_column")
//!     .unwrap()
//!     .series_equal(
//!         &Series::new("new_column", &[50, 40, 30, 20, 10])
//!     )
//! );
//! ```
//! #### Modifying a column based on some predicate
//!
//!```rust
//! #[macro_use] extern crate polars_core;
//! use polars_core::prelude::*;
//! use polars_lazy::prelude::*;
//!
//! let df = df! {
//!     "column_a" => &[1, 2, 3, 4, 5],
//!     "column_b" => &["a", "b", "c", "d", "e"]
//! }.unwrap();
//!
//! let new = df.lazy()
//!     .with_column(
//!         // value = 100 if x < 3 else x
//!         when(
//!             col("column_a").lt(lit(3))
//!         ).then(
//!             lit(100)
//!         ).otherwise(
//!             col("column_a")
//!         ).alias("new_column")
//!     )
//!     .collect()
//!     .unwrap();
//!
//! assert!(new.column("new_column")
//!     .unwrap()
//!     .series_equal(
//!         &Series::new("new_column", &[100, 100, 3, 4, 5])
//!     )
//! );
//! ```
//! #### Groupby + Aggregations
//!
//!```rust
//! use polars_core::prelude::*;
//! use polars_core::df;
//! use polars_lazy::prelude::*;
//! use arrow::legacy::prelude::QuantileInterpolOptions;
//!
//! fn example() -> PolarsResult<DataFrame> {
//!     let df = df!(
//!         "date" => ["2020-08-21", "2020-08-21", "2020-08-22", "2020-08-23", "2020-08-22"],
//!         "temp" => [20, 10, 7, 9, 1],
//!         "rain" => [0.2, 0.1, 0.3, 0.1, 0.01]
//!     )?;
//!
//!     df.lazy()
//!     .group_by([col("date")])
//!     .agg([
//!         col("rain").min().alias("min_rain"),
//!         col("rain").sum().alias("sum_rain"),
//!         col("rain").quantile(lit(0.5), QuantileInterpolOptions::Nearest).alias("median_rain"),
//!     ])
//!     .sort("date", Default::default())
//!     .collect()
//! }
//! ```
//!
//! #### Calling any function
//!
//! Below we lazily call a custom closure of type `Series => Result<Series>`. Because the closure
//! changes the type/variant of the Series we also define the return type. This is important because
//! due to the laziness the types should be known beforehand. Note that by applying these custom
//! functions you have access the the whole **eager API** of the Series/ChunkedArrays.
//!
//!```rust
//! #[macro_use] extern crate polars_core;
//! use polars_core::prelude::*;
//! use polars_lazy::prelude::*;
//!
//! let df = df! {
//!     "column_a" => &[1, 2, 3, 4, 5],
//!     "column_b" => &["a", "b", "c", "d", "e"]
//! }.unwrap();
//!
//! let new = df.lazy()
//!     .with_column(
//!         col("column_a")
//!         // apply a custom closure Series => Result<Series>
//!         .map(|_s| {
//!             Ok(Some(Series::new("", &[6.0f32, 6.0, 6.0, 6.0, 6.0])))
//!         },
//!         // return type of the closure
//!         GetOutput::from_type(DataType::Float64)).alias("new_column")
//!     )
//!     .collect()
//!     .unwrap();
//! ```
//!
//! #### Joins, filters and projections
//!
//! In the query below we do a lazy join and afterwards we filter rows based on the predicate `a < 2`.
//! And last we select the columns `"b"` and `"c_first"`. In an eager API this query would be very
//! suboptimal because we join on DataFrames with more columns and rows than needed. In this case
//! the query optimizer will do the selection of the columns (projection) and the filtering of the
//! rows (selection) before the join, thereby reducing the amount of work done by the query.
//!
//! ```rust
//! # use polars_core::prelude::*;
//! # use polars_lazy::prelude::*;
//!
//! fn example(df_a: DataFrame, df_b: DataFrame) -> LazyFrame {
//!     df_a.lazy()
//!     .left_join(df_b.lazy(), col("b_left"), col("b_right"))
//!     .filter(
//!         col("a").lt(lit(2))
//!     )
//!     .group_by([col("b")])
//!     .agg(
//!         vec![col("b").first().alias("first_b"), col("c").first().alias("first_c")]
//!      )
//!     .select(&[col("b"), col("c_first")])
//! }
//! ```
//!
//! If we want to do an aggregation on all columns we can use the wildcard operator `*` to achieve this.
//!
//! ```rust
//! # use polars_core::prelude::*;
//! # use polars_lazy::prelude::*;
//!
//! fn aggregate_all_columns(df_a: DataFrame) -> LazyFrame {
//!     df_a.lazy()
//!     .group_by([col("b")])
//!     .agg(
//!         vec![col("*").first()]
//!      )
//! }
//! ```
#![allow(ambiguous_glob_reexports)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
extern crate core;

#[cfg(feature = "dot_diagram")]
mod dot;
pub mod dsl;
pub mod frame;
pub mod physical_plan;
pub mod prelude;
mod scan;
#[cfg(test)]
mod tests;
pub mod utils;
