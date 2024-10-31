#[cfg(all(
    target_family = "unix",
    not(allocator = "default"),
    not(allocator = "mimalloc"),
))]
use jemallocator::Jemalloc;
#[cfg(all(
    not(debug_assertions),
    not(allocator = "default"),
    any(not(target_family = "unix"), allocator = "mimalloc"),
))]
use mimalloc::MiMalloc;

#[cfg(all(
    debug_assertions,
    target_family = "unix",
    not(allocator = "default"),
    not(allocator = "mimalloc"),
))]
use crate::memory::TracemallocAllocator;

#[global_allocator]
#[cfg(all(
    not(debug_assertions),
    not(allocator = "mimalloc"),
    not(allocator = "default"),
    target_family = "unix",
))]
static ALLOC: Jemalloc = Jemalloc;

#[global_allocator]
#[cfg(all(
    not(debug_assertions),
    not(allocator = "default"),
    any(not(target_family = "unix"), allocator = "mimalloc"),
))]
static ALLOC: MiMalloc = MiMalloc;

// On Windows tracemalloc does work. However, we build abi3 wheels, and the
// relevant C APIs are not part of the limited stable CPython API. As a result,
// linking breaks on Windows if we use tracemalloc C APIs. So we only use this
// on Unix for now.
#[global_allocator]
#[cfg(all(
    debug_assertions,
    target_family = "unix",
    not(allocator = "default"),
    not(allocator = "mimalloc"),
))]
static ALLOC: TracemallocAllocator<Jemalloc> = TracemallocAllocator::new(Jemalloc);

use std::alloc::Layout;
use std::ffi::{c_char, c_void};

use pyo3::ffi::PyCapsule_New;
use pyo3::{Bound, PyAny, PyResult, Python};

unsafe extern "C" fn alloc(size: usize, align: usize) -> *mut u8 {
    std::alloc::alloc(Layout::from_size_align_unchecked(size, align))
}

unsafe extern "C" fn dealloc(ptr: *mut u8, size: usize, align: usize) {
    std::alloc::dealloc(ptr, Layout::from_size_align_unchecked(size, align))
}

unsafe extern "C" fn alloc_zeroed(size: usize, align: usize) -> *mut u8 {
    std::alloc::alloc_zeroed(Layout::from_size_align_unchecked(size, align))
}

unsafe extern "C" fn realloc(ptr: *mut u8, size: usize, align: usize, new_size: usize) -> *mut u8 {
    std::alloc::realloc(
        ptr,
        Layout::from_size_align_unchecked(size, align),
        new_size,
    )
}

#[repr(C)]
struct AllocatorCapsule {
    alloc: unsafe extern "C" fn(usize, usize) -> *mut u8,
    dealloc: unsafe extern "C" fn(*mut u8, usize, usize),
    alloc_zeroed: unsafe extern "C" fn(usize, usize) -> *mut u8,
    realloc: unsafe extern "C" fn(*mut u8, usize, usize, usize) -> *mut u8,
}

static ALLOCATOR_CAPSULE: AllocatorCapsule = AllocatorCapsule {
    alloc,
    alloc_zeroed,
    dealloc,
    realloc,
};

static ALLOCATOR_CAPSULE_NAME: &[u8] = b"polars.polars._allocator\0";

pub fn create_allocator_capsule(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    unsafe {
        Bound::from_owned_ptr_or_err(
            py,
            PyCapsule_New(
                &ALLOCATOR_CAPSULE as *const AllocatorCapsule
            // Users of this capsule is not allowed to modify it.
            as *mut c_void,
                ALLOCATOR_CAPSULE_NAME.as_ptr() as *const c_char,
                None,
            ),
        )
    }
}
