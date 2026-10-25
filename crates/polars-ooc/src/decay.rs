//! While data is being spilled or is spilled, jemalloc returns freed memory to
//! the OS right away (decay 0), so the process stays close to the memory
//! budget. The old decay is restored once nothing is spilled for a while.

#[cfg(all(
    feature = "fast_alloc",
    target_family = "unix",
    not(target_os = "emscripten"),
))]
mod jemalloc {
    use std::ffi::{CStr, c_void};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{LazyLock, Mutex};
    use std::time::{Duration, Instant};

    use polars_config::config;
    use tikv_jemalloc_sys as jemalloc_sys;

    use crate::memory_manager::memory_manager;

    const POLL_INTERVAL: Duration = Duration::from_millis(100);

    // jemalloc uses at most MALLOCX_ARENA_LIMIT - 1 automatic arenas.
    const MAX_AUTO_ARENAS: u32 = 4094;

    static ACTIVE: AtomicBool = AtomicBool::new(false);
    static LAST_SPILL_NS: AtomicU64 = AtomicU64::new(0);
    static START: LazyLock<Instant> = LazyLock::new(Instant::now);
    static STATE: Mutex<DecayState> = Mutex::new(DecayState::EMPTY);

    struct DecayState {
        // The default decay from before the first spill, while active.
        saved: Option<Decay>,
        timer_running: bool,
    }

    impl DecayState {
        const EMPTY: Self = Self {
            saved: None,
            timer_running: false,
        };
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Decay {
        dirty_ms: isize,
        muzzy_ms: isize,
    }

    /// The decay while spilling. Freed pages become muzzy right away.
    fn pressure_decay(saved: Decay) -> Decay {
        Decay {
            dirty_ms: 0,
            muzzy_ms: saved.muzzy_ms,
        }
    }

    fn now_ns() -> u64 {
        START.elapsed().as_nanos() as u64
    }

    pub(crate) fn note_spill() {
        LAST_SPILL_NS.store(now_ns(), Ordering::Relaxed);
        // Pairs with the store to ACTIVE in `try_exit`: either the exit sees our
        // activity guard, or we see ACTIVE == false and enter again.
        if !ACTIVE.load(Ordering::SeqCst) {
            enter();
        }
    }

    #[cold]
    #[inline(never)]
    fn enter() {
        if !config().ooc_pressure_decay() {
            return;
        }
        let mut state = STATE.lock().unwrap();
        if !enter_locked(&mut state) || state.timer_running {
            return;
        }
        let spawned = std::thread::Builder::new()
            .name("polars-ooc-decay".into())
            .spawn(run_timer);
        if spawned.is_ok() {
            state.timer_running = true;
        } else {
            // Without a timer the decay would never be restored.
            let saved = state.saved.take().unwrap();
            restore(saved);
            ACTIVE.store(false, Ordering::SeqCst);
        }
    }

    /// Returns false if the decay could not be read.
    fn enter_locked(state: &mut DecayState) -> bool {
        if state.saved.is_none() {
            let Some(saved) = default_decay() else {
                return false;
            };
            // Defaults first, so arenas created from now on start with it.
            let pressure = pressure_decay(saved);
            set_default_decay(pressure);
            set_arena_decay(|_| pressure);
            purge_all_arenas();
            state.saved = Some(saved);
            if config().verbose() {
                eprintln!(
                    "ooc: spilling, setting allocator dirty decay to 0 (was {} ms)",
                    saved.dirty_ms
                );
            }
        }
        ACTIVE.store(true, Ordering::SeqCst);
        true
    }

    fn run_timer() {
        let mut restored = None;
        loop {
            std::thread::sleep(POLL_INTERVAL);
            let mut state = STATE.lock().unwrap();
            if let Some(saved) = state.saved {
                restored = None;
                let grace_ns = config().ooc_pressure_decay_grace_ms() * 1_000_000;
                if try_exit(saved, grace_ns) {
                    state.saved = None;
                    restored = Some(saved);
                }
                continue;
            }
            // One more pass for arenas created while restoring.
            if let Some(saved) = restored {
                restore_arena_decay(saved);
            }
            state.timer_running = false;
            return;
        }
    }

    /// Must be called while holding STATE. Returns whether the decay was restored.
    fn try_exit(saved: Decay, grace_ns: u64) -> bool {
        ACTIVE.store(false, Ordering::SeqCst);
        let idle_ns = now_ns().saturating_sub(LAST_SPILL_NS.load(Ordering::Relaxed));
        let mm = memory_manager();
        if idle_ns < grace_ns || mm.has_activity() || mm.should_spill() {
            ACTIVE.store(true, Ordering::SeqCst);
            return false;
        }
        restore(saved);
        if config().verbose() {
            eprintln!("ooc: nothing spilled, restoring allocator decay");
        }
        true
    }

    fn restore(saved: Decay) {
        set_default_decay(saved);
        restore_arena_decay(saved);
    }

    /// Gives every arena the decay jemalloc gives it when it is created.
    fn restore_arena_decay(saved: Decay) {
        let oversize = oversize_arena_index(
            read::<u32>(c"opt.narenas").unwrap_or(0),
            read::<usize>(c"opt.oversize_threshold").unwrap_or(0),
        );
        // ENOENT when jemalloc is built without background threads.
        let background_threads = read::<bool>(c"background_thread").unwrap_or(false);
        set_arena_decay(|i| decay_at_creation(Some(i) == oversize, background_threads, saved));
    }

    fn oversize_arena_index(opt_narenas: u32, oversize_threshold: usize) -> Option<u32> {
        (oversize_threshold > 0).then(|| opt_narenas.min(MAX_AUTO_ARENAS))
    }

    /// The decay of a new arena, from `arena_create_huge_arena` in jemalloc.
    fn decay_at_creation(is_oversize: bool, background_threads: bool, defaults: Decay) -> Decay {
        if !is_oversize || background_threads {
            return defaults;
        }
        // Without background threads the oversize arena purges right away.
        Decay {
            dirty_ms: defaults.dirty_ms.min(0),
            muzzy_ms: defaults.muzzy_ms.min(0),
        }
    }

    fn default_decay() -> Option<Decay> {
        Some(Decay {
            dirty_ms: read(c"arenas.dirty_decay_ms")?,
            muzzy_ms: read(c"arenas.muzzy_decay_ms")?,
        })
    }

    fn set_default_decay(decay: Decay) {
        write(c"arenas.dirty_decay_ms", decay.dirty_ms);
        write(c"arenas.muzzy_decay_ms", decay.muzzy_ms);
    }

    /// Sets the decay of every existing arena. Writing to an arena that does
    /// not exist fails without creating it.
    fn set_arena_decay(decay_of: impl Fn(u32) -> Decay) {
        let (Some(mut dirty_mib), Some(mut muzzy_mib)) = (
            name_to_mib::<3>(c"arena.0.dirty_decay_ms"),
            name_to_mib::<3>(c"arena.0.muzzy_decay_ms"),
        ) else {
            return;
        };
        let num_arenas = read::<u32>(c"arenas.narenas").unwrap_or(0);
        for i in 0..num_arenas {
            let decay = decay_of(i);
            dirty_mib[1] = i as usize;
            muzzy_mib[1] = i as usize;
            write_mib(&dirty_mib, decay.dirty_ms);
            write_mib(&muzzy_mib, decay.muzzy_ms);
        }
    }

    fn purge_all_arenas() {
        // 4096 is MALLCTL_ARENAS_ALL.
        unsafe {
            jemalloc_sys::mallctl(
                c"arena.4096.purge".as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            );
        }
    }

    fn read<T: Copy + Default>(name: &CStr) -> Option<T> {
        let mut value = T::default();
        let mut len = size_of::<T>();
        let ret = unsafe {
            jemalloc_sys::mallctl(
                name.as_ptr(),
                (&raw mut value).cast::<c_void>(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (ret == 0 && len == size_of::<T>()).then_some(value)
    }

    fn write<T: Copy>(name: &CStr, mut value: T) -> bool {
        let ret = unsafe {
            jemalloc_sys::mallctl(
                name.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                (&raw mut value).cast::<c_void>(),
                size_of::<T>(),
            )
        };
        ret == 0
    }

    fn name_to_mib<const N: usize>(name: &CStr) -> Option<[usize; N]> {
        let mut mib = [0usize; N];
        let mut len = N;
        let ret =
            unsafe { jemalloc_sys::mallctlnametomib(name.as_ptr(), mib.as_mut_ptr(), &mut len) };
        (ret == 0 && len == N).then_some(mib)
    }

    fn write_mib<T: Copy>(mib: &[usize], mut value: T) -> bool {
        let ret = unsafe {
            jemalloc_sys::mallctlbymib(
                mib.as_ptr(),
                mib.len(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                (&raw mut value).cast::<c_void>(),
                size_of::<T>(),
            )
        };
        ret == 0
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::memory_manager::OocActivity;

        fn thread_arena() -> u32 {
            // Allocating through jemalloc makes sure this thread has an arena.
            unsafe { jemalloc_sys::free(jemalloc_sys::malloc(64)) };
            read::<u32>(c"thread.arena").unwrap()
        }

        fn arena_decay(i: u32) -> Decay {
            let mut dirty_mib = name_to_mib::<3>(c"arena.0.dirty_decay_ms").unwrap();
            let mut muzzy_mib = name_to_mib::<3>(c"arena.0.muzzy_decay_ms").unwrap();
            dirty_mib[1] = i as usize;
            muzzy_mib[1] = i as usize;
            Decay {
                dirty_ms: read_mib(&dirty_mib).unwrap(),
                muzzy_ms: read_mib(&muzzy_mib).unwrap(),
            }
        }

        fn read_mib<T: Copy + Default>(mib: &[usize]) -> Option<T> {
            let mut value = T::default();
            let mut len = size_of::<T>();
            let ret = unsafe {
                jemalloc_sys::mallctlbymib(
                    mib.as_ptr(),
                    mib.len(),
                    (&raw mut value).cast::<c_void>(),
                    &mut len,
                    std::ptr::null_mut(),
                    0,
                )
            };
            (ret == 0).then_some(value)
        }

        // Sets a known default and gives the current arenas that default.
        fn reset(defaults: Decay) {
            set_default_decay(defaults);
            set_arena_decay(|_| defaults);
        }

        #[test]
        fn enter_and_exit_restore_decay() {
            let _guard = crate::test_lock();
            let defaults = Decay {
                dirty_ms: 250,
                muzzy_ms: 750,
            };
            reset(defaults);
            let arena = thread_arena();

            let mut state = DecayState::EMPTY;
            for _ in 0..2 {
                assert!(enter_locked(&mut state));
                assert_eq!(state.saved, Some(defaults));
                assert_eq!(default_decay(), Some(pressure_decay(defaults)));
                assert_eq!(arena_decay(arena), pressure_decay(defaults));

                assert!(try_exit(state.saved.take().unwrap(), 0));
                assert_eq!(default_decay(), Some(defaults));
                assert_eq!(arena_decay(arena), defaults);
            }
        }

        #[test]
        fn exit_waits_for_activity_and_grace() {
            let _guard = crate::test_lock();
            let defaults = Decay {
                dirty_ms: 300,
                muzzy_ms: 900,
            };
            reset(defaults);
            let mut state = DecayState::EMPTY;
            assert!(enter_locked(&mut state));
            let saved = state.saved.unwrap();

            let activity = OocActivity::new();
            assert!(!try_exit(saved, 0));
            assert!(ACTIVE.load(Ordering::SeqCst));
            assert_eq!(default_decay(), Some(pressure_decay(defaults)));
            drop(activity);

            LAST_SPILL_NS.store(now_ns(), Ordering::Relaxed);
            assert!(!try_exit(saved, 60_000_000_000));
            assert_eq!(default_decay(), Some(pressure_decay(defaults)));

            assert!(try_exit(saved, 0));
            assert!(!ACTIVE.load(Ordering::SeqCst));
            assert_eq!(default_decay(), Some(defaults));
        }

        #[test]
        fn arena_created_while_active_gets_default() {
            let _guard = crate::test_lock();
            let defaults = Decay {
                dirty_ms: 400,
                muzzy_ms: 800,
            };
            reset(defaults);
            let mut state = DecayState::EMPTY;
            assert!(enter_locked(&mut state));

            // A new arena, created while active, starts with the pressure decay.
            let new_arena = read::<u32>(c"arenas.create").unwrap();
            assert_eq!(arena_decay(new_arena), pressure_decay(defaults));

            assert!(try_exit(state.saved.take().unwrap(), 0));
            assert_eq!(arena_decay(new_arena), defaults);
        }

        struct ResetGrace;

        impl Drop for ResetGrace {
            fn drop(&mut self) {
                unsafe { std::env::remove_var("POLARS_OOC_PRESSURE_DECAY_GRACE_MS") };
                config().reload_env_var("POLARS_OOC_PRESSURE_DECAY_GRACE_MS");
            }
        }

        #[test]
        fn timer_restores_after_grace() {
            let _guard = crate::test_lock();
            let defaults = Decay {
                dirty_ms: 200,
                muzzy_ms: 600,
            };
            reset(defaults);
            let grace = Duration::from_millis(300);
            let _reset_grace = ResetGrace;
            unsafe {
                std::env::set_var(
                    "POLARS_OOC_PRESSURE_DECAY_GRACE_MS",
                    grace.as_millis().to_string(),
                )
            };
            config().reload_env_var("POLARS_OOC_PRESSURE_DECAY_GRACE_MS");

            let start = Instant::now();
            note_spill();
            assert!(ACTIVE.load(Ordering::SeqCst));
            assert_eq!(default_decay(), Some(pressure_decay(defaults)));

            // The timer restores and clears `saved` under STATE.
            while STATE.lock().unwrap().saved.is_some() {
                assert!(start.elapsed() < grace + 10 * POLL_INTERVAL);
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(start.elapsed() >= grace);
            assert_eq!(default_decay(), Some(defaults));

            // The timer ends after one more pass.
            while STATE.lock().unwrap().timer_running {
                assert!(start.elapsed() < grace + 20 * POLL_INTERVAL);
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        #[test]
        fn oversize_arena_decay() {
            let defaults = Decay {
                dirty_ms: 500,
                muzzy_ms: 1000,
            };
            assert_eq!(oversize_arena_index(96, 8 << 20), Some(96));
            assert_eq!(oversize_arena_index(4096, 8 << 20), Some(4094));
            assert_eq!(oversize_arena_index(96, 0), None);
            assert_eq!(decay_at_creation(false, false, defaults), defaults);
            assert_eq!(decay_at_creation(true, true, defaults), defaults);
            assert_eq!(
                decay_at_creation(true, false, defaults),
                Decay {
                    dirty_ms: 0,
                    muzzy_ms: 0,
                }
            );
            let never = Decay {
                dirty_ms: -1,
                muzzy_ms: 0,
            };
            assert_eq!(decay_at_creation(true, false, never), never);
        }
    }
}

#[cfg(all(
    feature = "fast_alloc",
    target_family = "unix",
    not(target_os = "emscripten"),
))]
pub(crate) use jemalloc::note_spill;

#[cfg(not(all(
    feature = "fast_alloc",
    target_family = "unix",
    not(target_os = "emscripten"),
)))]
pub(crate) fn note_spill() {}
