use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use polars_utils::sys::MEMINFO;

use crate::pipeline::FORCE_OOC;

const TO_MB: usize = 2 << 19;

#[derive(Clone)]
pub(super) struct MemTracker {
    // available memory at the start of this node
    available_mem: Arc<AtomicUsize>,
    used_by_node: Arc<AtomicUsize>,
    fetch_count: Arc<AtomicUsize>,
    thread_count: usize,
    available_at_start: usize,
    refresh_interval: usize,
}

impl MemTracker {
    pub(super) fn new(thread_count: usize) -> Self {
        let refresh_interval = if std::env::var(FORCE_OOC).is_ok() {
            1
        } else {
            64
        };

        let mut out = Self {
            available_mem: Default::default(),
            used_by_node: Default::default(),
            fetch_count: Arc::new(AtomicUsize::new(1)),
            thread_count,
            available_at_start: 0,
            refresh_interval,
        };
        let available = MEMINFO.free() as usize;
        out.available_mem.store(available, Ordering::Relaxed);
        out.available_at_start = available;
        out
    }

    /// This shouldn't be called often as this is expensive.
    pub fn refresh_memory(&self) {
        self.available_mem
            .store(MEMINFO.free() as usize, Ordering::Relaxed);
    }

    /// Get available memory of the system measured on latest refresh.
    pub(super) fn get_available(&self) -> usize {
        // once in every n passes we fetch mem usage.
        let fetch_count = self.fetch_count.fetch_add(1, Ordering::Relaxed);

        if fetch_count % (self.refresh_interval * self.thread_count) == 0 {
            self.refresh_memory()
        }
        self.available_mem.load(Ordering::Relaxed)
    }

    pub(super) fn get_available_latest(&self) -> usize {
        self.refresh_memory();
        self.fetch_count.store(0, Ordering::Relaxed);
        self.available_mem.load(Ordering::Relaxed)
    }

    pub(super) fn free_memory_fraction_since_start(&self) -> f64 {
        // We divide first to reduce the precision loss in floats.
        // We also add 1.0 to available_at_start to prevent division by zero.
        let available_at_start = (self.available_at_start / TO_MB) as f64 + 1.0;
        let available = (self.get_available() / TO_MB) as f64;
        available / available_at_start
    }

    /// Increment the used memory and return the previous value.
    pub(super) fn fetch_add(&self, add: usize) -> usize {
        self.used_by_node.fetch_add(add, Ordering::Relaxed)
    }
}
