//! Domain specific language for the Lazy api.
#[cfg(feature = "cumulative_eval")]
mod eval;
#[cfg(feature = "compile")]
pub mod functions;
mod into;
#[cfg(feature = "list")]
mod list;

use into::IntoExpr;
pub use polars_plan::dsl::*;
