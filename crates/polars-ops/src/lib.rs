#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(feature = "nightly", allow(internal_features))]
#![cfg_attr(feature = "nightly", feature(float_erf, titlecase))]
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

pub(crate) fn unique_column_name() -> polars_utils::pl_str::PlSmallStr {
    static COUNTER: polars_utils::relaxed_cell::RelaxedCell<u64> =
        polars_utils::relaxed_cell::RelaxedCell::new_u64(0);
    let idx = COUNTER.fetch_add(1);
    polars_utils::format_pl_smallstr!("_POLARS_TMP_OPS_{idx}")
}
