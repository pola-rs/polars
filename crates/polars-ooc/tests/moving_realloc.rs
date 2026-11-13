//! A moving reallocation holds its old allocation while it allocates the new
//! one. The dirty page limit has to count that old allocation, but the tracked
//! memory, and so the OOM-kill threshold, must not.
#![cfg(all(
    feature = "fast_alloc",
    target_family = "unix",
    not(target_os = "emscripten"),
))]

use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::ffi::{CString, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use polars_ooc::{Allocator, estimate_memory_usage};
use tikv_jemalloc_sys::{
    MALLOCX_ARENA, MALLOCX_TCACHE_NONE, dallocx, extent_hooks_t, mallctl, mallocx,
};

#[global_allocator]
static ALLOC: Allocator = Allocator;

const MIB: usize = 1 << 20;

type Hook = extern "C" fn(usize, usize, usize) -> usize;

static POLARS_HOOK: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static LAST_DIRTY: AtomicUsize = AtomicUsize::new(0);
static LAST_MOVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static LAST_PURGE: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static RECORDING: Cell<bool> = const { Cell::new(false) };
}

/// Calls the hook of Polars and records its input and result.
extern "C" fn record_hook(size: usize, dirty_bytes: usize, move_bytes: usize) -> usize {
    let polars_hook: Hook = unsafe { std::mem::transmute(POLARS_HOOK.load(Ordering::SeqCst)) };
    let purge = polars_hook(size, dirty_bytes, move_bytes);
    if RECORDING.with(|r| r.get()) {
        CALLS.fetch_add(1, Ordering::SeqCst);
        LAST_DIRTY.store(dirty_bytes, Ordering::SeqCst);
        LAST_MOVE_BYTES.store(move_bytes, Ordering::SeqCst);
        LAST_PURGE.store(purge, Ordering::SeqCst);
    }
    purge
}

fn ctl<T>(name: &str, old: Option<&mut T>, new: Option<&mut T>) {
    let name = CString::new(name).unwrap();
    let mut old_len = size_of::<T>();
    let (old_ptr, old_len_ptr) = match old {
        Some(old) => ((old as *mut T).cast::<c_void>(), &raw mut old_len),
        None => (std::ptr::null_mut(), std::ptr::null_mut()),
    };
    let (new_ptr, new_len) = match new {
        Some(new) => ((new as *mut T).cast::<c_void>(), size_of::<T>()),
        None => (std::ptr::null_mut(), 0),
    };
    let ret = unsafe { mallctl(name.as_ptr(), old_ptr, old_len_ptr, new_ptr, new_len) };
    assert_eq!(ret, 0, "{name:?}");
}

/// Creates an arena that returns unused pages to the OS right away. Without
/// merge, its large allocations can't grow in place.
fn new_arena(no_merge: bool) -> u32 {
    let mut arena = 0u32;
    if no_merge {
        let mut default_hooks: *mut extent_hooks_t = std::ptr::null_mut();
        ctl("arena.0.extent_hooks", Some(&mut default_hooks), None);
        let mut hooks = unsafe { *default_hooks };
        hooks.merge = None;
        let mut hooks: *mut extent_hooks_t = Box::leak(Box::new(hooks));
        let name = c"arenas.create";
        let mut len = size_of::<u32>();
        let ret = unsafe {
            mallctl(
                name.as_ptr(),
                (&raw mut arena).cast(),
                &mut len,
                (&raw mut hooks).cast(),
                size_of::<*mut extent_hooks_t>(),
            )
        };
        assert_eq!(ret, 0);
    } else {
        ctl::<u32>("arenas.create", Some(&mut arena), None);
    }
    set_decay(arena, 0);
    arena
}

fn set_decay(arena: u32, mut decay_ms: isize) {
    ctl(
        &format!("arena.{arena}.dirty_decay_ms"),
        None,
        Some(&mut decay_ms),
    );
    ctl(
        &format!("arena.{arena}.muzzy_decay_ms"),
        None,
        Some(&mut decay_ms),
    );
}

fn purge(arena: u32) {
    ctl::<()>(&format!("arena.{arena}.purge"), None, None);
}

fn arena_flags(arena: u32) -> i32 {
    MALLOCX_ARENA(arena as usize) | MALLOCX_TCACHE_NONE
}

/// Leaves `bytes` of dirty pages in `arena`, and returns the dirty pages of all
/// arenas.
fn make_dirty(arena: u32, probe_arena: u32, bytes: usize) -> usize {
    unsafe {
        let p = mallocx(bytes, arena_flags(arena));
        assert!(!p.is_null());
        dallocx(p, arena_flags(arena));
        purge(probe_arena);
        CALLS.store(0, Ordering::SeqCst);
        RECORDING.with(|r| r.set(true));
        let p = mallocx(MIB, arena_flags(probe_arena));
        RECORDING.with(|r| r.set(false));
        assert!(!p.is_null());
        dallocx(p, arena_flags(probe_arena));
    }
    assert_eq!(CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(LAST_MOVE_BYTES.load(Ordering::SeqCst), 0);
    let dirty = LAST_DIRTY.load(Ordering::SeqCst);
    assert!(dirty >= bytes);
    dirty
}

fn set_env(var: &str, value: Option<String>) {
    unsafe {
        match value {
            Some(value) => std::env::set_var(var, value),
            None => std::env::remove_var(var),
        }
    }
    polars_config::config().reload_env_var(var);
}

fn set_limits(dirty_limit: Option<usize>, oomkill_threshold: Option<usize>) {
    let total = polars_config::total_memory() as f64;
    set_env(
        "POLARS_OOC_DIRTY_LIMIT_FRACTION",
        dirty_limit.map(|b| (b as f64 / total).to_string()),
    );
    set_env(
        "POLARS_OOMKILL_THRESHOLD_MB",
        oomkill_threshold.map(|b| b.div_ceil(1_000_000).to_string()),
    );
}

fn set_thread_arena(mut arena: u32) {
    ctl("thread.arena", None, Some(&mut arena));
}

/// Reallocates `ptr` from `old_size` to `new_size` with the dirty page limit and
/// the OOM-kill threshold halfway between the memory with and without the old
/// allocation. Returns the new pointer, the old allocation bytes the hook saw
/// and the bytes it purged.
unsafe fn realloc_near_limit(
    ptr: *mut u8,
    old_size: usize,
    new_size: usize,
    dirty: usize,
) -> (*mut u8, usize, usize) {
    let layout = Layout::from_size_align(old_size, 8).unwrap();
    let usage_after = estimate_memory_usage() as usize + new_size - old_size;
    set_limits(
        Some(usage_after + dirty + old_size / 2),
        Some(usage_after + old_size / 2),
    );
    CALLS.store(0, Ordering::SeqCst);
    RECORDING.with(|r| r.set(true));
    let new_ptr = unsafe { ALLOC.realloc(ptr, layout, new_size) };
    RECORDING.with(|r| r.set(false));
    set_limits(None, None);
    assert!(!new_ptr.is_null());
    assert_eq!(CALLS.load(Ordering::SeqCst), 1);
    (
        new_ptr,
        LAST_MOVE_BYTES.load(Ordering::SeqCst),
        LAST_PURGE.load(Ordering::SeqCst),
    )
}

#[test]
fn moving_realloc_counts_old_allocation_for_dirty_limit_only() {
    const OLD: usize = 32 * MIB;
    const NEW: usize = 64 * MIB;
    const DIRTY: usize = 48 * MIB;

    // The first flush of a thread installs the hook.
    drop(vec![
        0u8;
        2 * polars_config::get_ooc_drift_threshold() as usize
    ]);
    let mut polars_hook = 0usize;
    ctl(
        "experimental.hooks.purge_before_grow",
        Some(&mut polars_hook),
        None,
    );
    assert_ne!(polars_hook, 0, "Polars did not install its hook");
    POLARS_HOOK.store(polars_hook, Ordering::SeqCst);
    let mut hook = record_hook as Hook;
    ctl(
        "experimental.hooks.purge_before_grow",
        None,
        Some(&mut hook),
    );

    let probe_arena = new_arena(false);
    let dirty_arena = new_arena(false);
    set_decay(dirty_arena, -1);
    let moving_arena = new_arena(true);
    let in_place_arena = new_arena(false);
    let old_layout = Layout::from_size_align(OLD, 8).unwrap();

    unsafe {
        // A move holds the old allocation on top of the limit, so it purges. The
        // OOM-kill threshold is below the memory with the old allocation, but it
        // does not count it, so it does not abort.
        set_thread_arena(moving_arena);
        let old = ALLOC.alloc(old_layout);
        assert!(!old.is_null());
        old.write_bytes(7, OLD);
        purge(moving_arena);
        let dirty = make_dirty(dirty_arena, probe_arena, DIRTY);
        let (new, move_bytes, purged) = realloc_near_limit(old, OLD, NEW, dirty);
        assert_ne!(new, old, "the reallocation did not move");
        assert_eq!(move_bytes, OLD);
        assert!(purged > 0);
        assert!(std::slice::from_raw_parts(new, OLD).iter().all(|b| *b == 7));
        ALLOC.dealloc(new, Layout::from_size_align(NEW, 8).unwrap());

        // Growing in place holds no old allocation, so it stays under the limit.
        set_thread_arena(in_place_arena);
        let layout = Layout::from_size_align(NEW, 8).unwrap();
        let p = ALLOC.alloc(layout);
        assert!(!p.is_null());
        assert_eq!(ALLOC.realloc(p, layout, OLD), p);
        purge(in_place_arena);
        let dirty = make_dirty(dirty_arena, probe_arena, DIRTY);
        let (new, move_bytes, purged) = realloc_near_limit(p, OLD, NEW, dirty);
        assert_eq!(new, p, "the reallocation did not grow in place");
        assert_eq!(move_bytes, 0);
        assert_eq!(purged, 0);
        ALLOC.dealloc(new, layout);
    }
}
