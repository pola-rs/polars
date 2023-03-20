mod config;
mod convert;
mod dispatcher;

pub use convert::{create_pipeline, get_dummy_operator, get_operator, get_sink, swap_join_order};
pub use dispatcher::PipeLine;
use polars_core::POOL;

pub use crate::executors::sinks::groupby::aggregates::can_convert_to_hash_agg;

pub(crate) fn morsels_per_sink() -> usize {
    POOL.current_num_threads()
}

// Number of OOC partitions.
// proxy for RAM size multiplier
pub(crate) const PARTITION_SIZE: usize = 64;

// env vars
pub(crate) static FORCE_OOC_GROUPBY: &str = "POLARS_FORCE_OOC_GROUPBY";
pub(crate) static FORCE_OOC_SORT: &str = "POLARS_FORCE_OOC_SORT";
