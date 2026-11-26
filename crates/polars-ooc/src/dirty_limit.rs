//! Limits the memory jemalloc holds, its active and unused (dirty) pages, to
//! [`polars_config::get_ooc_dirty_limit_bytes`]. Before an allocation would take
//! it above the limit, jemalloc purges dirty pages first.

#[cfg(all(
    feature = "fast_alloc",
    target_family = "unix",
    not(target_os = "emscripten"),
))]
mod jemalloc {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicU64, Ordering};

    use tikv_jemalloc_sys as jemalloc_sys;

    /// The limit last given to jemalloc, where 0 means no limit and `UNKNOWN` that
    /// it must be given again.
    static JEMALLOC_LIMIT: AtomicU64 = AtomicU64::new(0);
    const UNKNOWN: u64 = u64::MAX;

    fn current_limit() -> u64 {
        match polars_config::get_ooc_dirty_limit_bytes() {
            u64::MAX => 0,
            limit => limit.max(1),
        }
    }

    /// Gives jemalloc the current limit if it changed. Called from the allocator,
    /// so it must not allocate.
    pub(crate) fn sync_limit() {
        let limit = current_limit();
        if JEMALLOC_LIMIT.load(Ordering::Relaxed) == limit {
            return;
        }
        JEMALLOC_LIMIT.store(limit, Ordering::Relaxed);
        let mut jemalloc_limit: usize = limit.try_into().unwrap_or(0);
        // Without the limit, dirty pages are only returned by decay.
        unsafe {
            jemalloc_sys::mallctl(
                c"experimental.active_dirty_purge_threshold".as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                (&raw mut jemalloc_limit).cast::<c_void>(),
                size_of::<usize>(),
            );
        }
        // A thread that read an older limit may have set it after a newer one.
        if current_limit() != limit {
            JEMALLOC_LIMIT.store(UNKNOWN, Ordering::Relaxed);
        }
    }
}

#[cfg(all(
    feature = "fast_alloc",
    target_family = "unix",
    not(target_os = "emscripten"),
))]
pub(crate) use jemalloc::sync_limit;

#[cfg(not(all(
    feature = "fast_alloc",
    target_family = "unix",
    not(target_os = "emscripten"),
)))]
pub(crate) fn sync_limit() {}
