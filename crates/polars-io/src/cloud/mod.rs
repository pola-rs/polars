//! Interface with cloud storage through the object_store crate.

#[cfg(feature = "cloud")]
use std::borrow::Cow;
#[cfg(feature = "cloud")]
use std::sync::Arc;

#[cfg(feature = "cloud")]
use object_store::local::LocalFileSystem;
#[cfg(feature = "cloud")]
use object_store::ObjectStore;
#[cfg(feature = "cloud")]
use polars_core::prelude::{polars_bail, PolarsError, PolarsResult};

#[cfg(feature = "cloud")]
mod adaptors;
#[cfg(feature = "cloud")]
mod glob;
#[cfg(feature = "cloud")]
mod object_store_setup;
pub mod options;

#[cfg(feature = "cloud")]
pub use adaptors::*;
#[cfg(feature = "cloud")]
pub use glob::*;
#[cfg(feature = "cloud")]
pub use object_store_setup::*;
pub use options::*;
