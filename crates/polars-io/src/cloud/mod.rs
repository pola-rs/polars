//! Interface with cloud storage through the object_store crate.

#[cfg(feature = "cloud")]
mod adaptors;
#[cfg(feature = "cloud")]
mod glob;
#[cfg(feature = "cloud")]
mod object_store_setup;
pub mod options;
#[cfg(feature = "cloud")]
mod polars_object_store;

#[cfg(feature = "cloud")]
pub use adaptors::*;
#[cfg(feature = "cloud")]
pub use glob::*;
#[cfg(feature = "cloud")]
pub use object_store_setup::*;
pub use options::*;
#[cfg(feature = "cloud")]
pub use polars_object_store::*;
