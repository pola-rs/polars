//! # Functions
//!
//! Functions on expressions that might be useful.
mod arity;
mod coerce;
mod concat;
mod correlation;
mod horizontal;
mod index;
#[cfg(feature = "range")]
mod range;
mod repeat;
mod selectors;
mod syntactic_sugar;
mod temporal;

pub use arity::*;
pub use coerce::*;
pub use concat::*;
pub use correlation::*;
pub use horizontal::*;
pub use index::*;
#[cfg(feature = "temporal")]
use polars_core::export::arrow::temporal_conversions::{MICROSECONDS, MILLISECONDS, NANOSECONDS};
#[cfg(feature = "temporal")]
use polars_core::utils::arrow::temporal_conversions::SECONDS_IN_DAY;
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
pub use temporal::*;

#[cfg(feature = "arg_where")]
use crate::dsl::function_expr::FunctionExpr;
use crate::dsl::function_expr::ListFunction;
#[cfg(all(feature = "concat_str", feature = "strings"))]
use crate::dsl::function_expr::StringFunction;
use crate::dsl::*;
