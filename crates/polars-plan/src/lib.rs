#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![cfg_attr(feature = "nightly", allow(clippy::needless_pass_by_ref_mut))] // remove once stable

#[cfg(feature = "compile")]
pub mod constants;
#[cfg(feature = "compile")]
pub mod dot;
#[cfg(feature = "compile")]
pub mod dsl;
#[cfg(feature = "compile")]
pub mod frame;
#[cfg(feature = "compile")]
pub mod global;
#[cfg(feature = "compile")]
pub mod logical_plan;
#[cfg(feature = "compile")]
pub mod prelude;
#[cfg(feature = "compile")]
pub mod utils;
