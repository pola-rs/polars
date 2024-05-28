#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![cfg_attr(feature = "nightly", allow(clippy::needless_pass_by_ref_mut))] // remove once stable
#![cfg_attr(feature = "nightly", allow(clippy::blocks_in_conditions))] // Remove once stable.

extern crate core;

pub mod constants;
pub mod dsl;
pub mod frame;
pub mod global;
pub mod logical_plan;
pub mod prelude;
// Activate later
// mod reduce;
pub mod utils;
