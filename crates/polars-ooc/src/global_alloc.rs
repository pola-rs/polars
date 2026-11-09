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

#[cfg(test)]
pub(crate) fn add_test_usage(bytes: i64) {
    GLOBAL_ALLOC_SIZE.fetch_add(bytes as u64, Ordering::Relaxed);
}

/// The initial drift of a thread. It makes the first allocation of a thread
/// flush, which registers the flush at thread exit. We use two thread-local
/// variables because thread-locals with destructors can be slower to access.
const UNREGISTERED_DRIFT: i64 = i64::MIN / 2;

thread_local! {
    static LOCAL_ALLOC_DRIFT: Cell<i64> = const {
        Cell::new(UNREGISTERED_DRIFT)
    };
    static FLUSH_AT_EXIT: FlushAtExit = const {
        FlushAtExit(Cell::new(false))
    };
}

// Contains a bool to check for first registry.
struct FlushAtExit(Cell<bool>);

impl Drop for FlushAtExit {
    fn drop(&mut self) {
        flush_local_drift();
    }
}

#[inline(always)]
fn update_alloc_size(bytes: i64) {
    LOCAL_ALLOC_DRIFT.with(|drift| {
        let new = drift.get().wrapping_add(bytes);
        drift.set(new);
        if new.unsigned_abs() > polars_config::get_ooc_drift_threshold() {
            flush_local_drift();
        }
    })
}

/// Adds the allocations of this thread that are not yet counted to the total.
#[cold]
#[inline(never)]
pub(crate) fn flush_local_drift() {
    let mut drift_bytes = LOCAL_ALLOC_DRIFT.with(|drift| drift.replace(0));
    let first_flush = FLUSH_AT_EXIT
        .try_with(|registered| !registered.0.replace(true))
        .unwrap_or(false);
    if first_flush {
        drift_bytes = drift_bytes.wrapping_sub(UNREGISTERED_DRIFT);
    }

    let old = GLOBAL_ALLOC_SIZE.fetch_add(drift_bytes as u64, Ordering::AcqRel);
    let total = old.wrapping_add(drift_bytes as u64);
    if total <= i64::MAX as u64 && total > polars_config::get_oomkill_threshold_bytes() {
        oomkill();
    }
}

#[cold]
#[inline(never)]
fn oomkill() -> ! {
    // Can't allocate here, so write directly to stderr.
    const MSG: &[u8] = b"polars: memory usage exceeded POLARS_OOMKILL_THRESHOLD_MB or POLARS_OOMKILL_THRESHOLD_FRACTION, aborting\n";
    unsafe {
        libc::write(2, MSG.as_ptr().cast(), MSG.len() as _);
        libc::abort()
    }
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
