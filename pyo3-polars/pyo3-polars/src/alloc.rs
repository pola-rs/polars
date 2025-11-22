use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::c_char;

use once_cell::race::OnceRef;
use pyo3::ffi::{PyCapsule_Import, Py_IsInitialized};
use pyo3::Python;

unsafe extern "C" fn fallback_alloc(size: usize, align: usize) -> *mut u8 {
    System.alloc(Layout::from_size_align_unchecked(size, align))
}

unsafe extern "C" fn fallback_dealloc(ptr: *mut u8, size: usize, align: usize) {
    System.dealloc(ptr, Layout::from_size_align_unchecked(size, align))
}

unsafe extern "C" fn fallback_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
    System.alloc_zeroed(Layout::from_size_align_unchecked(size, align))
}

unsafe extern "C" fn fallback_realloc(
    ptr: *mut u8,
    size: usize,
    align: usize,
    new_size: usize,
) -> *mut u8 {
    System.realloc(
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

static FALLBACK_ALLOCATOR_CAPSULE: AllocatorCapsule = AllocatorCapsule {
    alloc: fallback_alloc,
    alloc_zeroed: fallback_alloc_zeroed,
    dealloc: fallback_dealloc,
    realloc: fallback_realloc,
};

static ALLOCATOR_CAPSULE_NAME: &[u8] = b"polars.polars._allocator\0";

/// A memory allocator that relays allocations to the allocator used by Polars.
///
/// You can use it as the global memory allocator:
///
/// ```rust
/// use pyo3_polars::PolarsAllocator;
///
/// #[global_allocator]
/// static ALLOC: PolarsAllocator = PolarsAllocator::new();
/// ```
///
/// If the allocator capsule (`polars.polars._allocator`) is not available,
/// this allocator fallbacks to [`std::alloc::System`].
pub struct PolarsAllocator(OnceRef<'static, AllocatorCapsule>);

impl PolarsAllocator {
    fn get_allocator(&self) -> &'static AllocatorCapsule {
        // Do not allocate in this function,
        // otherwise it will cause infinite recursion.
        self.0.get_or_init(|| {
            let r = (unsafe { Py_IsInitialized() } != 0)
                .then(|| {
                    Python::attach(|_| unsafe {
                        let capsule =
                            (PyCapsule_Import(ALLOCATOR_CAPSULE_NAME.as_ptr() as *const c_char, 0)
                                as *const AllocatorCapsule)
                                .as_ref();
                        if capsule.is_none() {
                            pyo3::ffi::PyErr_Clear();
                        }
                        capsule
                    })
                })
                .flatten();
            #[cfg(debug_assertions)]
            if r.is_none() {
                // Do not use eprintln; it may alloc.
                let msg = b"failed to get allocator capsule\n";
                #[allow(clippy::useless_conversion)]
                unsafe {
                    libc::write(
                        2,
                        msg.as_ptr() as *const libc::c_void,
                        // Use try_into as types differ per OS
                        msg.len().try_into().unwrap(),
                    )
                };
            }
            r.unwrap_or(&FALLBACK_ALLOCATOR_CAPSULE)
        })
    }

    /// Create a `PolarsAllocator`.
    pub const fn new() -> Self {
        PolarsAllocator(OnceRef::new())
    }
}

impl Default for PolarsAllocator {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl GlobalAlloc for PolarsAllocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        (self.get_allocator().alloc)(layout.size(), layout.align())
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        (self.get_allocator().dealloc)(ptr, layout.size(), layout.align());
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        (self.get_allocator().alloc_zeroed)(layout.size(), layout.align())
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        (self.get_allocator().realloc)(ptr, layout.size(), layout.align(), new_size)
    }
}
