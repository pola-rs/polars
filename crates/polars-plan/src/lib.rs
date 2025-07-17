#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![cfg_attr(feature = "nightly", allow(clippy::needless_pass_by_ref_mut))] // remove once stable
#![cfg_attr(feature = "nightly", allow(clippy::blocks_in_conditions))] // Remove once stable.
#![cfg_attr(
    feature = "allow_unused",
    allow(unused, dead_code, irrefutable_let_patterns)
)] // Maybe be caused by some feature
// combinations
extern crate core;

pub mod callback;
#[cfg(feature = "polars_cloud_client")]
pub mod client;
pub mod constants;
pub mod dsl;
pub mod frame;
pub mod plans;
pub mod prelude;
pub mod utils;
