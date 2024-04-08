mod config;
mod convert;
mod dispatcher;

pub use convert::{
    create_pipeline, get_dummy_operator, get_operator, get_sink, swap_join_order, CallBacks,
};
pub use dispatcher::{execute_pipeline, PipeLine};
use polars_core::prelude::*;
use polars_core::POOL;
use polars_utils::cell::SyncUnsafeCell;

pub use crate::executors::sinks::group_by::aggregates::can_convert_to_hash_agg;
use crate::operators::{Operator, Sink};

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
        Ok(std::cmp::max(50_000 / n_cols.max(1) * thread_factor, 1000))
    }
}

type PhysSink = Box<dyn Sink>;
/// A physical operator/sink per thread.
type ThreadedOperator = Vec<PhysOperator>;
type ThreadedOperatorMut<'a> = &'a mut [PhysOperator];
type ThreadedSinkMut<'a> = &'a mut [PhysSink];

#[repr(transparent)]
pub(crate) struct PhysOperator {
    inner: SyncUnsafeCell<Box<dyn Operator>>,
}

impl From<Box<dyn Operator>> for PhysOperator {
    fn from(value: Box<dyn Operator>) -> Self {
        Self {
            inner: SyncUnsafeCell::new(value),
        }
    }
}

impl PhysOperator {
    pub(crate) fn get_mut(&mut self) -> &mut dyn Operator {
        &mut **self.inner.get_mut()
    }

    pub(crate) fn get_ref(&self) -> &dyn Operator {
        unsafe { &**self.inner.get() }
    }
}
