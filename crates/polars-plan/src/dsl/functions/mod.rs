//! # Functions
//!
//! Functions on expressions that might be useful.
//!
mod arity;
mod coerce;
mod concat;
mod correlation;
mod horizontal;
mod index;
mod range;
mod selectors;
mod syntactic_sugar;
mod temporal;

use std::ops::{BitAnd, BitOr};

pub use arity::*;
pub use coerce::*;
pub use concat::*;
pub use correlation::*;
pub use horizontal::*;
pub use index::*;
#[cfg(feature = "temporal")]
use polars_core::export::arrow::temporal_conversions::NANOSECONDS;
#[cfg(feature = "temporal")]
use polars_core::utils::arrow::temporal_conversions::SECONDS_IN_DAY;
#[cfg(feature = "dtype-struct")]
use polars_core::utils::get_supertype;
pub use range::*;
pub use selectors::*;
pub use syntactic_sugar::*;
pub use temporal::*;

#[cfg(feature = "arg_where")]
use crate::dsl::function_expr::FunctionExpr;
use crate::dsl::function_expr::ListFunction;
#[cfg(all(feature = "concat_str", feature = "strings"))]
use crate::dsl::function_expr::StringFunction;
use crate::dsl::*;
