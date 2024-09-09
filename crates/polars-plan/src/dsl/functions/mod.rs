//! # Functions
//!
//! Functions on expressions that might be useful.
mod arity;
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
pub use business::*;
#[cfg(feature = "dtype-struct")]
pub use coerce::*;
pub use concat::*;
#[cfg(feature = "cov")]
pub use correlation::*;
pub use horizontal::*;
#[cfg(any(feature = "range", feature = "arg_where"))]
pub use index::*;
#[cfg(feature = "dtype-struct")]
use polars_core::utils::get_supertype;
#[cfg(all(feature = "range", feature = "temporal"))]
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
