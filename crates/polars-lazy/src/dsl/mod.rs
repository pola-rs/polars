//! Domain specific language for the Lazy API.
//!
//! This DSL revolves around the [`Expr`] type, which represents an abstract
//! operation on a DataFrame, such as mapping over a column, filtering, group_by, or aggregation.
//! In general, functions on [`LazyFrame`]s consume the [`LazyFrame`] and produce a new [`LazyFrame`] representing
//! the result of applying the function and passed expressions to the consumed LazyFrame.
//! At runtime, when [`LazyFrame::collect`](crate::frame::LazyFrame::collect) is called, the expressions that comprise
//! the [`LazyFrame`]'s logical plan are materialized on the actual underlying Series.
//! For instance, `let expr = col("x").pow(lit(2)).alias("x2");` would produce an expression representing the abstract
//! operation of squaring the column `"x"` and naming the resulting column `"x2"`, and to apply this operation to a
//! [`LazyFrame`], you'd use `let lazy_df = lazy_df.with_column(expr);`.
//! (Of course, a column named `"x"` must either exist in the original DataFrame or be produced by one of the preceding
//! operations on the [`LazyFrame`].)
//!
//! [`LazyFrame`]: crate::frame::LazyFrame
//!
//! There are many, many free functions that this module exports that produce an [`Expr`] from scratch; [`col`] and
//! [`lit`] are two examples.
//! Expressions also have several methods, such as [`pow`](`Expr::pow`) and [`alias`](`Expr::alias`), that consume them
//! and produce a new expression.
//!
//! Several expressions are only available when the necessary feature is enabled.
//! Examples of features that unlock specialized expression include `string`, `temporal`, and `dtype-categorical`.
//! These specialized expressions provide implementations of functions that you'd otherwise have to implement by hand.
//!
//! Because of how abstract and flexible the [`Expr`] type is, care must be take to ensure you only attempt to perform
//! sensible operations with them.
//! For instance, as mentioned above, you have to make sure any columns you reference already exist in the LazyFrame.
//! Furthermore, there is nothing stopping you from calling, for example, [`any`](`Expr::any`) with an expression
//! that will yield an `f64` column (instead of `bool`), or `col("string") - col("f64")`, which would attempt
//! to subtract an `f64` Series from a `string` Series.
//! These kinds of invalid operations will only yield an error at runtime, when
//! [`collect`](crate::frame::LazyFrame::collect) is called on the [`LazyFrame`].

#[cfg(any(feature = "cumulative_eval", feature = "list_eval"))]
mod eval;
pub mod functions;
mod into;
#[cfg(feature = "list_eval")]
mod list;

#[cfg(any(feature = "cumulative_eval", feature = "list_eval"))]
pub use eval::*;
pub use functions::*;
#[cfg(any(feature = "cumulative_eval", feature = "list_eval"))]
use into::IntoExpr;
#[cfg(feature = "list_eval")]
pub use list::*;
pub use polars_plan::dsl::*;
pub use polars_plan::plans::UdfSchema;
