mod cache;
mod executor;
mod ext_context;
mod filter;
mod gather;
mod group_by;
mod group_by_dynamic;
pub(super) mod group_by_rolling;
mod group_by_streaming;
mod hconcat;
mod join;
#[cfg(feature = "merge_sorted")]
mod merge_sorted;
mod projection;
mod projection_simple;
mod projection_utils;
mod scan;
mod slice;
mod sort;
mod stack;
mod udf;
mod union;
mod unique;

use std::borrow::Cow;

pub use executor::*;
pub use filter::column_to_mask;
use polars_core::runtime::RAYON;
use polars_plan::utils::*;
use projection_utils::*;
use rayon::prelude::*;

// Avoid the scheduling and concatenation overhead for small inputs.
const MIN_ROWS_PER_VERTICAL_PARTITION: usize = 1_024;

// Split by rows when horizontal work alone cannot saturate the thread pool.
fn should_split_vertically(n_rows: usize, n_horizontal: usize, n_threads: usize) -> bool {
    n_threads > 1
        && n_horizontal > 0
        && n_horizontal < n_threads
        && n_rows / n_threads >= MIN_ROWS_PER_VERTICAL_PARTITION
}

pub(super) use self::cache::*;
pub(super) use self::ext_context::*;
pub(super) use self::filter::*;
pub(super) use self::gather::*;
pub(super) use self::group_by::*;
#[cfg(feature = "dynamic_group_by")]
pub(super) use self::group_by_dynamic::*;
#[cfg(feature = "dynamic_group_by")]
pub(super) use self::group_by_rolling::GroupByRollingExec;
pub(super) use self::group_by_streaming::*;
pub(super) use self::hconcat::*;
pub(super) use self::join::*;
#[cfg(feature = "merge_sorted")]
pub(super) use self::merge_sorted::*;
pub(super) use self::projection::*;
pub(super) use self::projection_simple::*;
pub(super) use self::scan::*;
pub(super) use self::slice::*;
pub(super) use self::sort::*;
pub(super) use self::stack::*;
pub(super) use self::udf::*;
pub(super) use self::union::*;
pub(super) use self::unique::*;
use crate::prelude::*;

#[cfg(test)]
mod tests {
    use super::should_split_vertically;

    #[test]
    fn test_should_split_vertically() {
        assert!(!should_split_vertically(100_000, 1, 1));
        assert!(!should_split_vertically(100_000, 0, 12));
        assert!(!should_split_vertically(12 * 1_024 - 1, 1, 12));
        assert!(should_split_vertically(12 * 1_024, 1, 12));
        assert!(!should_split_vertically(100_000, 12, 12));
    }
}
