//! Utilities for dealing with memory allocations.

use std::alloc::GlobalAlloc;

use libc::{c_int, c_uint, size_t, uintptr_t};


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
        PyTraceMalloc_Untrack(TRACEMALLOC_DOMAIN, ptr as uintptr_t);
        let result = self.wrapped_alloc.realloc(ptr, layout, new_size);
        PyTraceMalloc_Track(TRACEMALLOC_DOMAIN, result as uintptr_t, new_size);
        result
    }
}
