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
