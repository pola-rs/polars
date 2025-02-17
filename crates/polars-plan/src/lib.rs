#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![cfg_attr(feature = "nightly", allow(clippy::needless_pass_by_ref_mut))] // remove once stable
#![cfg_attr(feature = "nightly", allow(clippy::blocks_in_conditions))] // Remove once stable.

extern crate core;

#[cfg(feature = "polars_cloud")]
pub mod client;
pub mod constants;
pub mod dsl;
pub mod frame;
pub mod global;
pub mod plans;
pub mod prelude;
pub mod utils;
