use std::sync::LazyLock;

/// # Safety
/// This may break aliasing rules, make sure you are the only owner.
#[allow(clippy::mut_from_ref)]
pub unsafe fn to_mutable_slice<T: Copy>(s: &[T]) -> &mut [T] {
    let ptr = s.as_ptr() as *mut T;
    let len = s.len();
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

pub static PAGE_SIZE: LazyLock<usize> = LazyLock::new(|| {
    #[cfg(target_family = "unix")]
    unsafe {
        libc::sysconf(libc::_SC_PAGESIZE) as usize
    }
    #[cfg(not(target_family = "unix"))]
    {
        4096
    }
});

pub mod prefetch {
    use super::PAGE_SIZE;

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
    pub fn madvise_sequential(#[allow(unused)] slice: &[u8]) {
        #[cfg(target_family = "unix")]
        madvise(slice, libc::MADV_SEQUENTIAL);
    }

    /// `madvise()` with `MADV_WILLNEED` on unix systems. This is a no-op on non-unix systems.
    pub fn madvise_willneed(#[allow(unused)] slice: &[u8]) {
        #[cfg(target_family = "unix")]
        madvise(slice, libc::MADV_WILLNEED);
    }

    /// `madvise()` with `MADV_POPULATE_READ` on linux systems. This a no-op on non-linux systems.
    pub fn madvise_populate_read(#[allow(unused)] slice: &[u8]) {
        #[cfg(target_os = "linux")]
        madvise(slice, libc::MADV_POPULATE_READ);
    }

    /// Forcibly reads at least one byte each page.
    pub fn force_populate_read(slice: &[u8]) {
        for i in (0..slice.len()).step_by(*PAGE_SIZE) {
            std::hint::black_box(slice[i]);
        }

        std::hint::black_box(slice.last().copied());
    }

    #[cfg(target_family = "unix")]
    fn madvise(slice: &[u8], advice: libc::c_int) {
        if slice.is_empty() {
            return;
        }
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

    pub fn no_prefetch(_: &[u8]) {}

    /// Get the configured memory prefetch function.
    pub fn get_memory_prefetch_func(verbose: bool) -> fn(&[u8]) -> () {
        let memory_prefetch_func = match std::env::var("POLARS_MEMORY_PREFETCH").ok().as_deref() {
            None => {
                // madvise_willneed performed the best on both MacOS on Apple Silicon and Ubuntu on x86-64,
                // using PDS-H query 3 SF=10 after clearing file cache as a benchmark.
                #[cfg(target_family = "unix")]
                {
                    madvise_willneed
                }
                #[cfg(not(target_family = "unix"))]
                {
                    no_prefetch
                }
            },
            Some("no_prefetch") => no_prefetch,
            Some("prefetch_l2") => prefetch_l2,
            Some("madvise_sequential") => {
                #[cfg(target_family = "unix")]
                {
                    madvise_sequential
                }
                #[cfg(not(target_family = "unix"))]
                {
                    panic!(
                        "POLARS_MEMORY_PREFETCH=madvise_sequential is not supported by this system"
                    );
                }
            },
            Some("madvise_willneed") => {
                #[cfg(target_family = "unix")]
                {
                    madvise_willneed
                }
                #[cfg(not(target_family = "unix"))]
                {
                    panic!(
                        "POLARS_MEMORY_PREFETCH=madvise_willneed is not supported by this system"
                    );
                }
            },
            Some("madvise_populate_read") => {
                #[cfg(target_os = "linux")]
                {
                    madvise_populate_read
                }
                #[cfg(not(target_os = "linux"))]
                {
                    panic!(
                        "POLARS_MEMORY_PREFETCH=madvise_populate_read is not supported by this system"
                    );
                }
            },
            Some("force_populate_read") => force_populate_read,
            Some(v) => panic!("invalid value for POLARS_MEMORY_PREFETCH: {}", v),
        };

        if verbose {
            let func_name = match memory_prefetch_func as usize {
                v if v == no_prefetch as usize => "no_prefetch",
                v if v == prefetch_l2 as usize => "prefetch_l2",
                v if v == madvise_sequential as usize => "madvise_sequential",
                v if v == madvise_willneed as usize => "madvise_willneed",
                v if v == madvise_populate_read as usize => "madvise_populate_read",
                v if v == force_populate_read as usize => "force_populate_read",
                _ => unreachable!(),
            };

            eprintln!("memory prefetch function: {}", func_name);
        }

        memory_prefetch_func
    }
}
