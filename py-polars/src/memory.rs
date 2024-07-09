//! Utilities for dealing with memory allocations.

use std::alloc::GlobalAlloc;

use libc::{c_int, c_uint, size_t, uintptr_t};

// When debug_assertions is enabled, use Python's tracemalloc to track memory
// allocations. This is a useful feature for production use too, but has a
// potential performance impact and so would need additional benchmarking. In
// addition, these APIs are not part of the limited Python ABI Polars uses,
// though they are unchanged between 3.7 and 3.12.
#[cfg(not(target_os = "windows"))]
extern "C" {
    fn PyTraceMalloc_Track(domain: c_uint, ptr: uintptr_t, size: size_t) -> c_int;
    fn PyTraceMalloc_Untrack(domain: c_uint, ptr: uintptr_t) -> c_int;
}

// Windows has issues linking to the tracemalloc APIs, so the functionality is
// disabled. We have fake implementations just to make sure we don't have
// issues building.
#[cfg(target_os = "windows")]
#[allow(non_snake_case)]
fn PyTraceMalloc_Track(_domain: c_uint, _ptr: uintptr_t, _size: size_t) -> c_int {
    -2
}

#[cfg(target_os = "windows")]
#[allow(non_snake_case)]
fn PyTraceMalloc_Untrack(_domain: c_uint, _ptr: uintptr_t) -> c_int {
    -2
}

/// Allocations require a domain to identify them when registering with
/// tracemalloc. Following NumPy's lead, we just pick a random constant that is
/// unlikely to clash with anyone else.
const TRACEMALLOC_DOMAIN: c_uint = 36740582;

/// Wrap an existing allocator, and register allocations and frees with Python's
/// `tracemalloc`. Registration functionality is disabled on Windows.
pub struct TracemallocAllocator<A: GlobalAlloc> {
    wrapped_alloc: A,
}

impl<A: GlobalAlloc> TracemallocAllocator<A> {
    /// Wrap the allocator such that allocations are registered with
    /// tracemalloc.
    #[allow(dead_code)]
    pub const fn new(wrapped_alloc: A) -> Self {
        Self { wrapped_alloc }
    }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for TracemallocAllocator<A> {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let result = self.wrapped_alloc.alloc(layout);
        PyTraceMalloc_Track(TRACEMALLOC_DOMAIN, result as uintptr_t, layout.size());
        result
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        PyTraceMalloc_Untrack(TRACEMALLOC_DOMAIN, ptr as uintptr_t);
        self.wrapped_alloc.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        let result = self.wrapped_alloc.alloc_zeroed(layout);
        PyTraceMalloc_Track(TRACEMALLOC_DOMAIN, result as uintptr_t, layout.size());
        result
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        let result = self.wrapped_alloc.realloc(ptr, layout, new_size);
        PyTraceMalloc_Track(TRACEMALLOC_DOMAIN, result as uintptr_t, new_size);
        result
    }
}
