use polars_core::prelude::*;

#[cfg(feature = "strings")]
mod string_namespace;
#[cfg(feature = "strings")]
pub use string_namespace::*;
