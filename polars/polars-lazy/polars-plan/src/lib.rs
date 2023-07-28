#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![cfg_attr(feature = "nightly", allow(clippy::needless_pass_by_ref_mut))] // remove once stable

pub mod constants;
pub mod dot;
pub mod dsl;
pub mod frame;
pub mod global;
pub mod logical_plan;
pub mod prelude;
pub mod utils;
