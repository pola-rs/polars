use std::sync::atomic::AtomicBool;
use std::sync::{Arc, LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Duration;

use fs4::fs_std::FileExt;

use super::utils::FILE_CACHE_PREFIX;
use crate::pl_async;

pub(super) static GLOBAL_FILE_CACHE_LOCK: LazyLock<GlobalLock> = LazyLock::new(|| {
    let path = FILE_CACHE_PREFIX.join(".process-lock");

    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|err| {
            panic!("failed to open/create global file cache lockfile: {err}");
        })
        .unwrap();

    let at_bool = Arc::new(AtomicBool::new(false));
    // Holding this access tracker prevents the background task from
    // unlocking the lock.
    let access_tracker = AccessTracker(at_bool.clone());
    let notify_lock_acquired = Arc::new(tokio::sync::Notify::new());
    let notify_lock_acquired_2 = notify_lock_acquired.clone();

    pl_async::get_runtime().spawn(async move {
        let access_tracker = at_bool;
        let notify_lock_acquired = notify_lock_acquired_2;
        let verbose = false;

        loop {
            if verbose {
                eprintln!("file cache background unlock: waiting for acquisition notification");
            }

            notify_lock_acquired.notified().await;

            if verbose {
                eprintln!("file cache background unlock: got acquisition notification");
            }

            loop {
                if !access_tracker.swap(false, std::sync::atomic::Ordering::Relaxed) {
                    if let Some(unlocked_by_this_call) = GLOBAL_FILE_CACHE_LOCK.try_unlock() {
                        if unlocked_by_this_call && verbose {
                            eprintln!(
                                "file cache background unlock: unlocked global file cache lockfile"
                            );
                        }
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
        }
    });

    GlobalLock {
        inner: RwLock::new(GlobalLockData { file, state: None }),
        access_tracker,
        notify_lock_acquired,
    }
});

pub(super) enum LockedState {
    /// Shared between threads and other processes.
    Shared,
    #[allow(dead_code)]
    /// Locked exclusively by the eviction task of this process.
    Eviction,
}

#[allow(dead_code)]
pub(super) type GlobalFileCacheGuardAny<'a> = RwLockReadGuard<'a, GlobalLockData>;
pub(super) type GlobalFileCacheGuardExclusive<'a> = RwLockWriteGuard<'a, GlobalLockData>;

pub(super) struct GlobalLockData {
    file: std::fs::File,
    state: Option<LockedState>,
}

pub(super) struct GlobalLock {
    inner: RwLock<GlobalLockData>,
    access_tracker: AccessTracker,
    notify_lock_acquired: Arc<tokio::sync::Notify>,
}

/// Tracks access to the global lock:
/// * The inner `bool` is used to delay the background unlock task from unlocking
///   the global lock until 3 seconds after the last lock attempt.
/// * The `Arc` ref-count is used as a semaphore that allows us to block exclusive
///   lock attempts while temporarily releasing the `RwLock`.
#[derive(Clone)]
struct AccessTracker(Arc<AtomicBool>);

impl Drop for AccessTracker {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

struct NotifyOnDrop(Arc<tokio::sync::Notify>);

impl Drop for NotifyOnDrop {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}

impl GlobalLock {
    fn get_access_tracker(&self) -> AccessTracker {
        let at = self.access_tracker.clone();
        at.0.store(true, std::sync::atomic::Ordering::Relaxed);
        at
    }

    /// Returns
    /// * `None` - Could be locked (ambiguous)
    /// * `Some(true)` - Unlocked (by this function call)
    /// * `Some(false)` - Unlocked (was not locked)
    fn try_unlock(&self) -> Option<bool> {
        if let Ok(mut this) = self.inner.try_write() {
            if Arc::strong_count(&self.access_tracker.0) <= 2 {
                return if this.state.take().is_some() {
                    FileExt::unlock(&this.file).unwrap();
                    Some(true)
                } else {
                    Some(false)
                };
            }
        }
        None
    }

    /// Acquire a shared lock.
    pub(super) fn lock_shared(&self) -> GlobalFileCacheGuardAny<'_> {
        let access_tracker = self.get_access_tracker();
        let _notify_on_drop = NotifyOnDrop(self.notify_lock_acquired.clone());

        {
            let this = self.inner.read().unwrap();

            if let Some(LockedState::Shared) = this.state {
                return this;
            }
        }

        {
            let mut this = self.inner.write().unwrap();

            if let Some(LockedState::Eviction) = this.state {
                FileExt::unlock(&this.file).unwrap();
                this.state = None;
            }

            if this.state.is_none() {
                FileExt::lock_shared(&this.file).unwrap();
                this.state = Some(LockedState::Shared);
            }
        }

        // Safety: Holding the access tracker guard maintains an Arc refcount
        // > 2, which prevents automatic unlock.
        debug_assert!(Arc::strong_count(&access_tracker.0) > 2);

        {
            let this = self.inner.read().unwrap();

            if let Some(LockedState::Eviction) = this.state {
                // Try again
                drop(this);
                return self.lock_shared();
            }

            assert!(
                this.state.is_some(),
                "impl error: global file cache lock was unlocked"
            );
            this
        }
    }

    /// Acquire an exclusive lock on the cache directory. Holding this lock freezes
    /// all cache operations except for reading from already-opened data files.
    #[allow(dead_code)]
    pub(super) fn try_lock_eviction(&self) -> Option<GlobalFileCacheGuardExclusive<'_>> {
        let access_tracker = self.get_access_tracker();

        if let Ok(mut this) = self.inner.try_write() {
            if
            // 3:
            // * the Lazy<GlobalLock>
            // * the global unlock background task
            // * this function
            Arc::strong_count(&access_tracker.0) > 3 {
                return None;
            }

            let _notify_on_drop = NotifyOnDrop(self.notify_lock_acquired.clone());

            if let Some(ref state) = this.state {
                if matches!(state, LockedState::Eviction) {
                    return Some(this);
                }
            }

            if this.state.take().is_some() {
                FileExt::unlock(&this.file).unwrap();
            }

            if this.file.try_lock_exclusive().is_ok() {
                this.state = Some(LockedState::Eviction);
                return Some(this);
            }
        }
        None
    }
}
