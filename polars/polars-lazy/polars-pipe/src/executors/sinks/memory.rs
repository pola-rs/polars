use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use polars_utils::sys::MEMINFO;

#[derive(Clone)]
pub(super) struct MemTracker {
    // available memory at the start of this node
    available_mem: Arc<AtomicUsize>,
    used_by_node: Arc<AtomicUsize>,
    fetch_count: Arc<AtomicUsize>,
    thread_count: usize,
}

impl MemTracker {
    pub(super) fn new(thread_count: usize) -> Self {
        let out = Self {
            available_mem: Default::default(),
            used_by_node: Default::default(),
            fetch_count: Arc::new(AtomicUsize::new(1)),
            thread_count,
        };
        out.refresh_memory();
        out
    }

    /// This shouldn't be called often as this is expensive.
    fn refresh_memory(&self) {
        self.available_mem
            .store(MEMINFO.free() as usize, Ordering::Relaxed);
    }

    /// Get available memory of the system measured on latest refresh.
    pub(super) fn get_available(&self) -> usize {
        // once in every n passes we fetch mem usage.
        let fetch_count = self.fetch_count.fetch_add(1, Ordering::Relaxed);
        if fetch_count % (64 * self.thread_count) == 0 {
            self.refresh_memory()
        }
        self.available_mem.load(Ordering::Relaxed)
    }

    /// Increment the used memory and return the previous value.
    pub(super) fn fetch_add(&self, add: usize) -> usize {
        self.used_by_node.fetch_add(add, Ordering::Relaxed)
    }
}
