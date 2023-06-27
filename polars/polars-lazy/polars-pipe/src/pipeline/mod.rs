mod config;
mod convert;
mod dispatcher;

pub use convert::{create_pipeline, get_dummy_operator, get_operator, get_sink, swap_join_order};
pub use dispatcher::PipeLine;
use polars_core::prelude::*;
use polars_core::POOL;

pub use crate::executors::sinks::groupby::aggregates::can_convert_to_hash_agg;

pub(crate) fn morsels_per_sink() -> usize {
    POOL.current_num_threads()
}

// Number of OOC partitions.
// proxy for RAM size multiplier
pub(crate) const PARTITION_SIZE: usize = 64;

// env vars
pub(crate) static FORCE_OOC: &str = "POLARS_FORCE_OOC";

/// ideal chunk size we strive to have
/// scale the chunk size depending on the number of
/// columns. With 10 columns we use a chunk size of 40_000
pub(crate) fn determine_chunk_size(n_cols: usize, n_threads: usize) -> PolarsResult<usize> {
    if let Ok(val) = std::env::var("POLARS_STREAMING_CHUNK_SIZE") {
        val.parse().map_err(
            |_| polars_err!(ComputeError: "could not parse 'POLARS_STREAMING_CHUNK_SIZE' env var"),
        )
    } else {
        let thread_factor = std::cmp::max(12 / n_threads, 1);
        Ok(std::cmp::max(50_000 / n_cols * thread_factor, 1000))
    }
}
