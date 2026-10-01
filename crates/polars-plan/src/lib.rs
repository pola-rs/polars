#![cfg_attr(docsrs, feature(doc_cfg))]
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
pub mod traversal;
pub mod utils;

pub(crate) fn unique_column_name() -> polars_utils::pl_str::PlSmallStr {
    static COUNTER: polars_utils::relaxed_cell::RelaxedCell<u64> =
        polars_utils::relaxed_cell::RelaxedCell::new_u64(0);
    let idx = COUNTER.fetch_add(1);
    polars_utils::format_pl_smallstr!("_POLARS_TMP_PLAN_{idx}")
}
