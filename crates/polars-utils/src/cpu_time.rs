//! Process and thread CPU clocks. Only differences between readings are meaningful.

/// CPU time consumed by this process (all threads, user + system) in
/// nanoseconds, or `None` if the platform clock is unavailable.
#[cfg(unix)]
pub fn process_cpu_ns() -> Option<u64> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable timespec for the duration of the call.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    (rc == 0).then(|| ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64)
}

#[cfg(windows)]
pub fn process_cpu_ns() -> Option<u64> {
    use std::mem::MaybeUninit;

    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    let mut creation = MaybeUninit::<FILETIME>::uninit();
    let mut exit = MaybeUninit::<FILETIME>::uninit();
    let mut kernel = MaybeUninit::<FILETIME>::uninit();
    let mut user = MaybeUninit::<FILETIME>::uninit();

    // SAFETY: all four out-params are valid writable FILETIMEs.
    let ok = unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            creation.as_mut_ptr(),
            exit.as_mut_ptr(),
            kernel.as_mut_ptr(),
            user.as_mut_ptr(),
        )
    };
    if ok == 0 {
        return None;
    }

    // SAFETY: GetProcessTimes succeeded, so all four are initialised.
    let (kernel, user) = unsafe { (kernel.assume_init(), user.assume_init()) };
    let to_ns = |f: FILETIME| {
        let ticks = ((f.dwHighDateTime as u64) << 32) | f.dwLowDateTime as u64;
        ticks * 100 // FILETIME is in 100ns units.
    };
    Some(to_ns(kernel) + to_ns(user))
}

#[cfg(not(any(unix, windows)))]
pub fn process_cpu_ns() -> Option<u64> {
    None
}

/// CPU time consumed by the calling thread (user + system) in nanoseconds, or
/// `None` if unavailable. Excludes time the thread is descheduled.
#[cfg(unix)]
pub fn thread_cpu_ns() -> Option<u64> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable timespec for the duration of the call.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    (rc == 0).then(|| ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64)
}

/// Returns `None`: Windows' thread CPU clock is too coarse to time individual polls.
#[cfg(windows)]
pub fn thread_cpu_ns() -> Option<u64> {
    None
}

#[cfg(not(any(unix, windows)))]
pub fn thread_cpu_ns() -> Option<u64> {
    None
}
