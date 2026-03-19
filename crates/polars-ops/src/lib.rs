#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(feature = "nightly", feature(unicode_internals))]
#![cfg_attr(feature = "nightly", allow(internal_features))]
#![cfg_attr(
    feature = "allow_unused",
    allow(unused, dead_code, irrefutable_let_patterns)
)] // Maybe be caused by some feature

pub mod chunked_array;
#[cfg(feature = "pivot")]
pub use frame::unpivot;
pub mod frame;
pub mod prelude;
pub mod series;
