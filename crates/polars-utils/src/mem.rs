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
pub unsafe fn prefetch_l2(ptr: *const u8) {
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
