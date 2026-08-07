use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

static GLOBAL_ALLOC_SIZE: AtomicU64 = AtomicU64::new(0);

/// Returns an estimate of the total amount of bytes allocated.
///
/// This can be up to OOC_DRIFT_THRESHOLD * num_threads bytes less than or
/// greater than the true memory usage.
pub fn estimate_memory_usage() -> u64 {
    let bytes = GLOBAL_ALLOC_SIZE.load(Ordering::Relaxed);
    if bytes > i64::MAX as u64 {
        // Drift + moving allocations between threads allows for underflow,
        // so this is best reported as zero.
        0
    } else {
        bytes
    }
}

thread_local! {
    static LOCAL_ALLOC_DRIFT: Cell<i64> = const {
        Cell::new(0)
    };
}

#[inline(always)]
fn update_alloc_size(bytes: i64) {
    LOCAL_ALLOC_DRIFT.with(|drift| {
        let new = drift.get().wrapping_add(bytes);
        if new.unsigned_abs() <= polars_config::get_ooc_drift_threshold() {
            drift.set(new);
        } else {
            GLOBAL_ALLOC_SIZE.fetch_add(new as u64, Ordering::AcqRel);
            drift.set(0)
        }
    })
}

#[cfg(all(
    feature = "fast_alloc",
    target_family = "unix",
    not(target_os = "emscripten"),
))]
static UNDERLYING_ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(
    feature = "fast_alloc",
    any(not(target_family = "unix"), target_os = "emscripten"),
))]
static UNDERLYING_ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(not(feature = "fast_alloc"))]
static UNDERLYING_ALLOC: std::alloc::System = std::alloc::System;

pub struct Allocator;

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        update_alloc_size(layout.size() as i64);
        unsafe { UNDERLYING_ALLOC.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        update_alloc_size(layout.size() as i64);
        unsafe { UNDERLYING_ALLOC.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        update_alloc_size(-(layout.size() as i64));
        unsafe { UNDERLYING_ALLOC.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        update_alloc_size(new_size as i64 - layout.size() as i64);
        unsafe { UNDERLYING_ALLOC.realloc(ptr, layout, new_size) }
    }
}
