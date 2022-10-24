//! Domain specific language for the Lazy api.
#[cfg(feature = "cumulative_eval")]
mod eval;
pub mod functions;
mod into;
mod list;

#[cfg(feature = "cumulative_eval")]
pub use eval::*;
pub use functions::*;
#[cfg(feature = "cumulative_eval")]
use into::IntoExpr;
pub use list::*;
pub use polars_plan::dsl::*;
pub use polars_plan::logical_plan::UdfSchema;
