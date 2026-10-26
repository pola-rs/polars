mod decay;
mod global_alloc;
mod memory_manager;
mod spill_context;
mod spill_file;
mod spill_frame;
mod spill_token;

pub use global_alloc::{Allocator, estimate_memory_usage};
pub use memory_manager::{MemoryReservation, memory_manager};
use polars_utils::relaxed_cell::RelaxedCell;
pub use spill_context::{
    LeastRecentSpillContext, MostRecentSpillContext, ParameterFreeSpillContext, RandomSpillContext,
    SpillContextParam, WeakSpillContext,
};
pub use spill_file::{flush_ooc_cleanup, init_ooc_cleaner};
pub use spill_frame::{PinnedFrameMut, SpillFrame};
pub use spill_token::{PinnedMut, PinnedRef, SpillToken};

pub trait Spillable: Send + Sync + 'static {
    type Spilled;

    /// Estimates how many bytes this object takes up in memory.
    fn estimate_byte_size(&self) -> usize;

    /// Spills this value, returning a spilled representation.
    fn spill(&self, context_id: &str) -> impl Future<Output = Self::Spilled> + Send;

    /// Given a previously spilled representation
    fn unspill(location: &Self::Spilled) -> impl Future<Output = Self> + Send;

    /// The number of bytes a spilled representation takes on disk.
    fn spilled_bytes(_location: &Self::Spilled) -> u64 {
        0
    }
}

static BYTES_SPILLED_TO_DISK: RelaxedCell<u64> = RelaxedCell::new_u64(0);

// Tests that change global spill or allocator state run one at a time.
#[cfg(test)]
fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
