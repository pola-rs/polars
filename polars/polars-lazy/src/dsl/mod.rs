//! Domain specific language for the Lazy api.
#[cfg(feature = "compile")]
pub mod functions;

pub use polars_plan::dsl::*;