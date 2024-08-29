use once_cell::sync::Lazy;
static PAGE_SIZE: Lazy<usize> = Lazy::new(|| {
    #[cfg(target_family = "unix")]
    unsafe {
        libc::sysconf(libc::_SC_PAGESIZE) as usize
    }
    #[cfg(not(target_family = "unix"))]
    {
        4096
    }
});

/// # Safety
/// This may break aliasing rules, make sure you are the only owner.
#[allow(clippy::mut_from_ref)]
pub unsafe fn to_mutable_slice<T: Copy>(s: &[T]) -> &mut [T] {
    let ptr = s.as_ptr() as *mut T;
    let len = s.len();
    std::slice::from_raw_parts_mut(ptr, len)
}

/// # Safety
///
/// This should only be called with pointers to valid memory.
unsafe fn prefetch_l2_impl(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        unsafe { _mm_prefetch(ptr as *const _, _MM_HINT_T1) };
    }

    #[cfg(all(target_arch = "aarch64", feature = "nightly"))]
    {
        use std::arch::aarch64::*;
        unsafe { _prefetch(ptr as *const _, _PREFETCH_READ, _PREFETCH_LOCALITY2) };
    }
}

/// Attempt to prefetch the memory in the slice to the L2 cache.
pub fn prefetch_l2(slice: &[u8]) {
    if slice.is_empty() {
        return;
    }

    // @TODO: We can play a bit more with this prefetching. Maybe introduce a maximum number of
    // prefetches as to not overwhelm the processor. The linear prefetcher should pick it up
    // at a certain point.

    for i in (0..slice.len()).step_by(*PAGE_SIZE) {
        unsafe { prefetch_l2_impl(slice[i..].as_ptr()) };
    }

    unsafe { prefetch_l2_impl(slice[slice.len() - 1..].as_ptr()) }
}

/// `madvise()` with `MADV_SEQUENTIAL` on unix systems. This is a no-op on non-unix systems.
pub fn madvise_sequential(slice: &[u8]) {
    #[cfg(target_family = "unix")]
    madvise(slice, libc::MADV_SEQUENTIAL);
}

/// `madvise()` with `MADV_WILLNEED` on unix systems. This is a no-op on non-unix systems.
pub fn madvise_willneed(slice: &[u8]) {
    #[cfg(target_family = "unix")]
    madvise(slice, libc::MADV_WILLNEED);
}

/// `madvise()` with `MADV_POPULATE_READ` on linux systems. This a no-op on non-linux systems.
pub fn madvise_populate_read(#[allow(unused)] slice: &[u8]) {
    #[cfg(target_os = "linux")]
    madvise(slice, libc::MADV_POPULATE_READ);
}

#[cfg(target_family = "unix")]
fn madvise(slice: &[u8], advice: libc::c_int) {
    let ptr = slice.as_ptr();

    let align = ptr as usize % *PAGE_SIZE;
    let ptr = ptr.wrapping_sub(align);
    let len = slice.len() + align;

    if unsafe { libc::madvise(ptr as *mut libc::c_void, len, advice) } != 0 {
        let err = std::io::Error::last_os_error();
        if let std::io::ErrorKind::InvalidInput = err.kind() {
            panic!("{}", err);
        }
    }
}
