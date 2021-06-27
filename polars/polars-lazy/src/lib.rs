//! Lazy API of Polars
//!
//! *Credits to the work of Andy Grove and Ballista/ DataFusion / Apache Arrow, which served as
//! insipration for the lazy API.*
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
//!         &Series::new("valid", &[50, 40, 30, 20, 10])
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
//!         &Series::new("valid", &[100, 100, 3, 4, 5])
//!     )
//! );
//! ```
//! #### Groupby + Aggregations
//!
//!```rust
//! use polars_core::prelude::*;
//! use polars_core::df;
//! use polars_lazy::prelude::*;
//!
//! fn example() -> Result<DataFrame> {
//!     let df = df!(
//!     "date" => ["2020-08-21", "2020-08-21", "2020-08-22", "2020-08-23", "2020-08-22"],
//!     "temp" => [20, 10, 7, 9, 1],
//!     "rain" => [0.2, 0.1, 0.3, 0.1, 0.01]
//!     )?;
//!
//!     df.lazy()
//!     .groupby(vec![col("date")])
//!     .agg(vec![
//!         col("rain").min(),
//!         col("rain").sum(),
//!         col("rain").quantile(0.5).alias("median_rain"),
//!     ])
//!     .sort("date", false)
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
//!             Ok(Series::new("", &[6.0f32, 6.0, 6.0, 6.0, 6.0]))
//!         },
//!         // return type of the closure
//!         Some(DataType::Float64)).alias("new_column")
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
//!     .groupby(vec![col("b")])
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
//!     .groupby(vec![col("b")])
//!     .agg(
//!         vec![col("*").first()]
//!      )
//! }
//! ```
#![cfg_attr(docsrs, feature(doc_cfg))]
#[cfg(all(feature = "datafusion", feature = "compile"))]
mod datafusion;
#[cfg(feature = "compile")]
pub mod dsl;
#[cfg(feature = "compile")]
mod dummies;
#[cfg(feature = "compile")]
pub mod frame;
#[cfg(feature = "compile")]
pub mod functions;
#[cfg(feature = "compile")]
pub mod logical_plan;
#[cfg(feature = "compile")]
pub mod physical_plan;
#[cfg(feature = "compile")]
pub mod prelude;
#[cfg(feature = "compile")]
pub(crate) mod utils;

#[cfg(test)]
mod tests {
    use polars_core::prelude::*;
    use polars_io::prelude::*;
    use std::io::Cursor;

    // physical plan see: datafusion/physical_plan/planner.rs.html#61-63

    pub(crate) fn get_df() -> DataFrame {
        let s = r#"
"sepal.length","sepal.width","petal.length","petal.width","variety"
5.1,3.5,1.4,.2,"Setosa"
4.9,3,1.4,.2,"Setosa"
4.7,3.2,1.3,.2,"Setosa"
4.6,3.1,1.5,.2,"Setosa"
5,3.6,1.4,.2,"Setosa"
5.4,3.9,1.7,.4,"Setosa"
4.6,3.4,1.4,.3,"Setosa"
"#;

        let file = Cursor::new(s);

        let df = CsvReader::new(file)
            // we also check if infer schema ignores errors
            .infer_schema(Some(3))
            .has_header(true)
            .finish()
            .unwrap();
        df
    }
}
