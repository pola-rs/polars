//! Keeps the tracked memory plus jemalloc's unused (dirty) pages under
//! [`polars_config::get_ooc_dirty_limit_bytes`]. jemalloc calls the hook before
//! an allocation takes pages that are not resident, and purges the returned
//! number of bytes of dirty pages first.

#[cfg(all(
    feature = "fast_alloc",
    target_family = "unix",
    not(target_os = "emscripten"),
))]
mod jemalloc {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, Ordering};

    use tikv_jemalloc_sys as jemalloc_sys;

    use crate::estimate_memory_usage;

    /// Purged on top of the excess, so that the next allocations don't purge again.
    const SLACK_BYTES: u64 = 64 << 20;

    /// Fewer dirty pages than this are not worth visiting every arena for.
    const MIN_PURGE_BYTES: u64 = 16 << 20;

    static INSTALLED: AtomicBool = AtomicBool::new(false);

    type PurgeBeforeGrowHook = extern "C" fn(usize, usize) -> usize;

    /// Runs inside malloc, so it must not allocate. The allocator counts an allocation
    /// before calling jemalloc, so the usage already includes it.
    extern "C" fn purge_before_grow(_size: usize, dirty_bytes: usize) -> usize {
        let dirty_bytes = dirty_bytes as u64;
        let usage = estimate_memory_usage() + dirty_bytes;
        let excess = usage.saturating_sub(polars_config::get_ooc_dirty_limit_bytes());
        let purge = (excess + SLACK_BYTES).min(dirty_bytes);
        if excess == 0 || purge < MIN_PURGE_BYTES {
            return 0;
        }
        purge.try_into().unwrap_or(usize::MAX)
    }

    /// Called from the allocator, so it must not allocate or read the config.
    pub(crate) fn install() {
        if INSTALLED.swap(true, Ordering::Relaxed) {
            return;
        }
        let mut hook: PurgeBeforeGrowHook = purge_before_grow;
        // Without the hook, dirty pages are only returned by decay.
        unsafe {
            jemalloc_sys::mallctl(
                c"experimental.hooks.purge_before_grow".as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                (&raw mut hook).cast::<c_void>(),
                size_of::<PurgeBeforeGrowHook>(),
            );
        }
    }
}

#[cfg(all(
    feature = "fast_alloc",
    target_family = "unix",
    not(target_os = "emscripten"),
))]
pub(crate) use jemalloc::install;

#[cfg(not(all(
    feature = "fast_alloc",
    target_family = "unix",
    not(target_os = "emscripten"),
)))]
pub(crate) fn install() {}
