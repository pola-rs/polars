//!
//! # Polars Lazy cookbook
//!
//! This page should serve a cookbook to quickly get you started with polars' query engine.
//! The lazy API allows you to create complex well performing queries on top of Polars eager.
//!
//! ## Tree Of Contents
//!
//! * [Start a lazy computation](#start-a-lazy-computation)
//! * [Filter](#filter)
//! * [Sort](#sort)
//! * [GroupBy](#groupby)
//! * [Joins](#joins)
//! * [Conditionally apply](#conditionally-apply)
//!
//! ## Start a lazy computation
//!
//! ```
//! use polars::prelude::*;
//! use polars::df;
//!
//! # fn example() -> Result<()> {
//! let df = df![
//!     "a" => [1, 2, 3],
//!     "b" => [None, Some("a"), Some("b")]
//! ]?;
//! // from an eager DataFrame
//! let lf: LazyFrame = df.lazy();
//!
//! // scan a csv file lazily
//! let lf: LazyFrame = LazyCsvReader::new("some_path".into())
//!                     .has_header(true)
//!                     .finish();
//!
//! // scan a parquet file lazily
//! let lf: LazyFrame = LazyFrame::new_from_parquet("some_path".into(), None, true);
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Filter
//! ```
//! use polars::prelude::*;
//! use polars::df;
//!
//! # fn example() -> Result<()> {
//! let df = df![
//!     "a" => [1, 2, 3],
//!     "b" => [None, Some("a"), Some("b")]
//! ]?;
//!
//! let filtered = df.lazy()
//!     .filter(col("a").gt(lit(2)))
//!     .collect()?;
//!
//! // filtered:
//!
//! // ╭─────┬─────╮
//! // │ a   ┆ b   │
//! // │ --- ┆ --- │
//! // │ i64 ┆ str │
//! // ╞═════╪═════╡
//! // │ 3   ┆ "c" │
//! // ╰─────┴─────╯
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Sort
//! ```
//! use polars::prelude::*;
//! use polars::df;
//!
//! # fn example() -> Result<()> {
//! let df = df![
//!     "a" => [1, 2, 3],
//!     "b" => ["a", "a", "b"]
//! ]?;
//! // sort this DataFrame by multiple columns
//!
//! // ordering of the columns
//! let reverse = vec![true, false];
//!
//! let sorted = df.lazy()
//!     .sort_by_exprs(vec![col("b"), col("a")], reverse)
//!     .collect()?;
//!
//! // sorted:
//!
//! // ╭─────┬─────╮
//! // │ a   ┆ b   │
//! // │ --- ┆ --- │
//! // │ i64 ┆ str │
//! // ╞═════╪═════╡
//! // │ 1   ┆ "a" │
//! // ├╌╌╌╌╌┼╌╌╌╌╌┤
//! // │ 2   ┆ "a" │
//! // ├╌╌╌╌╌┼╌╌╌╌╌┤
//! // │ 3   ┆ "b" │
//! // ╰─────┴─────╯
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Groupby
//!
//! This example is from the polars [user guide](https://pola-rs.github.io/polars-book/user-guide/howcani/df/groupby.html).
//!
//! ```
//! use polars::prelude::*;
//! # fn example() -> Result<()> {
//!
//!  let df = LazyCsvReader::new("reddit.csv".into())
//!     .has_header(true)
//!     .with_delimiter(b',')
//!     .finish()
//!     .groupby(vec![col("comment_karma")])
//!     .agg(vec![col("name").n_unique().alias("unique_names"), col("link_karma").max()])
//!     // take only 100 rows.
//!     .fetch(100)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Joins
//!
//! ```
//! use polars::prelude::*;
//! use polars::df;
//! # fn example() -> Result<()> {
//! let df_a = df![
//!     "a" => [1, 2, 1, 1],
//!     "b" => ["a", "b", "c", "c"],
//!     "c" => [0, 1, 2, 3]
//! ]?;
//!
//! let df_b = df![
//!     "foo" => [1, 1, 1],
//!     "bar" => ["a", "c", "c"],
//!     "ham" => ["let", "var", "const"]
//! ]?;
//!
//! let lf_a = df_a.clone().lazy();
//! let lf_b = df_b.clone().lazy();
//!
//! let joined = lf_a.join(lf_b, vec![col("a")], vec![col("foo")], JoinType::Outer).collect()?;
//! // joined:
//!
//! // ╭─────┬─────┬─────┬──────┬─────────╮
//! // │ b   ┆ c   ┆ a   ┆ bar  ┆ ham     │
//! // │ --- ┆ --- ┆ --- ┆ ---  ┆ ---     │
//! // │ str ┆ i64 ┆ i64 ┆ str  ┆ str     │
//! // ╞═════╪═════╪═════╪══════╪═════════╡
//! // │ "a" ┆ 0   ┆ 1   ┆ "a"  ┆ "let"   │
//! // ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
//! // │ "a" ┆ 0   ┆ 1   ┆ "c"  ┆ "var"   │
//! // ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
//! // │ "a" ┆ 0   ┆ 1   ┆ "c"  ┆ "const" │
//! // ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
//! // │ "b" ┆ 1   ┆ 2   ┆ null ┆ null    │
//! // ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
//! // │ "c" ┆ 2   ┆ 1   ┆ null ┆ null    │
//! // ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
//! // │ "c" ┆ 3   ┆ 1   ┆ null ┆ null    │
//! // ╰─────┴─────┴─────┴──────┴─────────╯
//!
//! // other join syntax options
//! # let lf_a = df_a.clone().lazy();
//! # let lf_b = df_b.clone().lazy();
//! let inner = lf_a.inner_join(lf_b, col("a"), col("foo")).collect()?;
//!
//! # let lf_a = df_a.clone().lazy();
//! # let lf_b = df_b.clone().lazy();
//! let left = lf_a.left_join(lf_b, col("a"), col("foo")).collect()?;
//!
//! # let lf_a = df_a.clone().lazy();
//! # let lf_b = df_b.clone().lazy();
//! let outer = lf_a.outer_join(lf_b, col("a"), col("foo")).collect()?;
//!
//! # let lf_a = df_a.clone().lazy();
//! # let lf_b = df_b.clone().lazy();
//! let joined_with_builder = lf_a.join_builder()
//!     .with(lf_b)
//!     .left_on(vec![col("a")])
//!     .right_on(vec![col("foo")])
//!     .how(JoinType::Inner)
//!     .force_parallel(true)
//!     .finish()
//!     .collect()?;
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Conditionally apply
//! If we want to create a new column based on some condition, we can use the `.when()/.then()/.otherwise()` expressions.
//!
//! * `when` - accpets a predicate epxression
//! * `then` - expression to use when `predicate == true`
//! * `otherwise` - expression to use when `predicate == false`
//!
//! ```
//! use polars::prelude::*;
//! use polars::df;
//! # fn example() -> Result<()> {
//! let df = df![
//!     "range" => [1, 2, 3, 4, 5, 6, 8, 9, 10],
//!     "left" => (0..10).map(|_| Some("foo")).collect::<Vec<_>>(),
//!     "right" => (0..10).map(|_| Some("bar")).collect::<Vec<_>>()
//! ]?;
//!
//! let new = df.lazy()
//!     .with_column(when(col("range").gt_eq(lit(5)))
//!         .then(col("left"))
//!         .otherwise(col("right")).alias("foo_or_bar")
//!     ).collect()?;
//!
//! // new:
//!
//! // ╭───────┬───────┬───────┬────────────╮
//! // │ range ┆ left  ┆ right ┆ foo_or_bar │
//! // │ ---   ┆ ---   ┆ ---   ┆ ---        │
//! // │ i64   ┆ str   ┆ str   ┆ str        │
//! // ╞═══════╪═══════╪═══════╪════════════╡
//! // │ 0     ┆ "foo" ┆ "bar" ┆ "bar"      │
//! // ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
//! // │ 1     ┆ "foo" ┆ "bar" ┆ "bar"      │
//! // ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
//! // │ 2     ┆ "foo" ┆ "bar" ┆ "bar"      │
//! // ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
//! // │ 3     ┆ "foo" ┆ "bar" ┆ "bar"      │
//! // ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
//! // │ ...   ┆ ...   ┆ ...   ┆ ...        │
//! // ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
//! // │ 5     ┆ "foo" ┆ "bar" ┆ "foo"      │
//! // ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
//! // │ 6     ┆ "foo" ┆ "bar" ┆ "foo"      │
//! // ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
//! // │ 7     ┆ "foo" ┆ "bar" ┆ "foo"      │
//! // ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
//! // │ 8     ┆ "foo" ┆ "bar" ┆ "foo"      │
//! // ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
//! // │ 9     ┆ "foo" ┆ "bar" ┆ "foo"      │
//! // ╰───────┴───────┴───────┴────────────╯
//!
//! # Ok(())
//! # }
//! ```
