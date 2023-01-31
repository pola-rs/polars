//! Domain specific language for the Lazy api.
#[cfg(any(feature = "cumulative_eval", feature = "list_eval"))]
mod eval;
pub mod functions;
mod into;
mod list;

#[cfg(any(feature = "cumulative_eval", feature = "list_eval"))]
pub use eval::*;
pub use functions::*;
#[cfg(any(feature = "cumulative_eval", feature = "list_eval"))]
use into::IntoExpr;
pub use list::*;
pub use polars_plan::dsl::*;
pub use polars_plan::logical_plan::UdfSchema;
