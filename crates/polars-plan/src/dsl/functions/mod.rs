//! # Functions
//!
//! Functions on expressions that might be useful.
#[cfg(feature = "business")]
mod business;
#[cfg(feature = "dtype-struct")]
mod coerce;
mod concat;
#[cfg(feature = "cov")]
mod correlation;
pub(crate) mod horizontal;
#[cfg(any(feature = "range", feature = "arg_where"))]
mod index;
#[cfg(feature = "range")]
mod range;
mod repeat;
mod selectors;
mod syntactic_sugar;
#[cfg(feature = "temporal")]
mod temporal;

pub use arity::*;
#[cfg(all(feature = "business", feature = "dtype-date"))]
pub use business::business_day_count;
#[cfg(feature = "dtype-struct")]
pub use coerce::*;
pub use concat::*;
#[cfg(feature = "cov")]
pub use correlation::*;
pub use horizontal::{
    all_horizontal, any_horizontal, coalesce, fold_exprs, max_horizontal, mean_horizontal,
    min_horizontal, reduce_exprs, sum_horizontal,
};
#[cfg(feature = "dtype-struct")]
pub use horizontal::{cum_fold_exprs, cum_reduce_exprs};
#[cfg(any(feature = "range", feature = "arg_where"))]
pub use index::*;
#[cfg(all(
    feature = "range",
    any(feature = "dtype-date", feature = "dtype-datetime")
))]
pub use range::date_range; // This shouldn't be necessary, but clippy complains about dead code
#[cfg(all(feature = "range", feature = "dtype-time"))]
pub use range::time_range; // This shouldn't be necessary, but clippy complains about dead code
#[cfg(feature = "range")]
pub use range::*;
pub use repeat::*;
pub use selectors::*;
pub use syntactic_sugar::*;
#[cfg(feature = "temporal")]
pub use temporal::*;

#[cfg(feature = "arg_where")]
use crate::dsl::function_expr::FunctionExpr;
use crate::dsl::function_expr::ListFunction;
#[cfg(all(feature = "concat_str", feature = "strings"))]
use crate::dsl::function_expr::StringFunction;
use crate::dsl::*;

/// Return the number of rows in the context.
pub fn len() -> Expr {
    Expr::Len
}

/// First column in a DataFrame.
pub fn first() -> Selector {
    nth(0)
}

/// Last column in a DataFrame.
pub fn last() -> Selector {
    nth(-1)
}

/// Nth column in a DataFrame.
pub fn nth(n: i64) -> Selector {
    Selector::ByIndex {
        indices: [n].into(),
        strict: true,
    }
}

/// Create a Literal Expression from `L`. A literal expression behaves like a column that contains a single distinct
/// value.
///
/// The column is automatically of the "correct" length to make the operations work. Often this is determined by the
/// length of the `LazyFrame` it is being used with. For instance, `lazy_df.with_column(lit(5).alias("five"))` creates a
/// new column named "five" that is the length of the Dataframe (at the time `collect` is called), where every value in
/// the column is `5`.
pub fn lit<L: Literal>(t: L) -> Expr {
    t.lit()
}
