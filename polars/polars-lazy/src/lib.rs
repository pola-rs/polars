//! Lazy API of Polars
//!
//! The lazy api of Polars supports a subset of the eager api. Apart from the distributed compute,
//! it is very similar to [Apache Spark](https://spark.apache.org/). You write queries in a
//! domain specific language. These queries translate to a logical plan, which represent your query steps.
//! Before execution this logical plan is optimized and may change the order of operations if this will increase performance.
//! Or implicit type casts may be added such that execution of the query won't lead to a type error (if it can be resolved).
//!
//! # Lazy DSL
//!
//! The lazy API of polars can be used as long we operation on one or multiple DataFrame(s) and
//! Series of the same length as the DataFrame. To get started we call the [lazy](crate::frame::IntoLazy::lazy)
//! method. This returns a [LazyFrame](crate::frame::LazyFrame) exposing the lazy API.
//!
//! Lazy operations don't execute until we call [collect](crate::frame::LazyFrame::collect).
//! This allows polars to optimize/reorder the query which may lead to faster queries or less type errors.
//!
//! The DSL is mostly defined by [LazyFrame](crate::frame::LazyFrame) for operations on DataFrames and
//! the [Expr](crate::dsl::Expr) and functions in the [dsl modules](crate::dsl) that operate
//! on expressions.
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
//! use polars_arrow::prelude::QuantileInterpolOptions;
//!
//! fn example() -> PolarsResult<DataFrame> {
//!     let df = df!(
//!     "date" => ["2020-08-21", "2020-08-21", "2020-08-22", "2020-08-23", "2020-08-22"],
//!     "temp" => [20, 10, 7, 9, 1],
//!     "rain" => [0.2, 0.1, 0.3, 0.1, 0.01]
//!     )?;
//!
//!     df.lazy()
//!     .groupby([col("date")])
//!     .agg([
//!         col("rain").min(),
//!         col("rain").sum(),
//!         col("rain").quantile(lit(0.5), QuantileInterpolOptions::Nearest).alias("median_rain"),
//!     ])
//!     .sort("date", Default::default())
//!     .collect()
//!
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
//!     .groupby([col("b")])
//!     .agg(
//!         vec![col("b").first(), col("c").first()]
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
//!     .groupby([col("b")])
//!     .agg(
//!         vec![col("*").first()]
//!      )
//! }
//! ```
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
extern crate core;

#[cfg(feature = "dot_diagram")]
mod dot;
pub mod dsl;
pub mod frame;
pub mod physical_plan;
pub mod prelude;
#[cfg(test)]
mod tests;
pub mod utils;
