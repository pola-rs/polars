use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Duration;

#[cfg(not(target_family = "wasm"))]
use fs4::FileExt;
use once_cell::sync::Lazy;
use polars_core::config;

use crate::pl_async;

pub(super) static GLOBAL_FILE_CACHE_LOCK: Lazy<GlobalLock> = Lazy::new(|| {
    let path = std::env::var("POLARS_TEMP_DIR")
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    let path = PathBuf::from(path).join("polars/file-cache/");
    let _ = std::fs::create_dir_all(&path);
    let path = path.join(".process-lock");

    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|err| {
            panic!("failed to open/create global file cache lockfile: {}", err);
        })
        .unwrap();

    let at_bool = Arc::new(AtomicBool::new(false));
    // Holding this access tracker prevents the background task from
    // unlocking the lock.
    let access_tracker = AccessTracker(at_bool.clone());

    pl_async::get_runtime().spawn(async move {
        let access_tracker = at_bool;
        let verbose = config::verbose();

        loop {
            if !access_tracker.swap(false, std::sync::atomic::Ordering::Relaxed)
                && GLOBAL_FILE_CACHE_LOCK.try_unlock()
                && verbose
            {
                eprintln!("unlocked global file cache lockfile");
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
    });

    GlobalLock {
        inner: RwLock::new(GlobalLockData { file, state: None }),
        access_tracker,
    }
});

pub(super) enum LockedState {
    Shared,
    #[allow(dead_code)]
    Exclusive,
}

#[allow(dead_code)]
pub(super) type GlobalFileCacheGuardAny<'a> = RwLockReadGuard<'a, GlobalLockData>;
pub(super) type GlobalFileCacheGuardExclusive<'a> = RwLockWriteGuard<'a, GlobalLockData>;

#[derive(Clone)]
struct AccessTracker(Arc<AtomicBool>);

pub(super) struct GlobalLockData {
    file: std::fs::File,
    state: Option<LockedState>,
}

pub(super) struct GlobalLock {
    inner: RwLock<GlobalLockData>,
    access_tracker: AccessTracker,
}

impl Drop for AccessTracker {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

impl GlobalLock {
    fn get_access_tracker(&self) -> AccessTracker {
        let at = self.access_tracker.clone();
        at.0.store(true, std::sync::atomic::Ordering::Relaxed);
        at
    }

    fn try_unlock(&self) -> bool {
        if let Ok(mut this) = self.inner.try_write() {
            if Arc::strong_count(&self.access_tracker.0) <= 2 && this.state.take().is_some() {
                this.file.unlock().unwrap();
                return true;
            }
        }
        false
    }

    /// Acquire either a shared or exclusive lock. This always returns a read-guard
    /// to allow for better parallelism within the current process. The tradeoff
    /// is that we may hold an exclusive lock on the global lockfile for longer
    /// than we need to (since we don't transition to a shared lock state),
    /// which blocks other processes.
    pub(super) fn lock_any(&self) -> GlobalFileCacheGuardAny {
        let access_tracker = self.get_access_tracker();

        {
            let this = self.inner.read().unwrap();

            if this.state.is_some() {
                return this;
            }
        }

        {
            let mut this = self.inner.write().unwrap();

            if this.state.is_none() {
                this.file.lock_shared().unwrap();
                this.state = Some(LockedState::Shared);
            }
        }

        // Safety: Holding the access tracker guard maintains an Arc refcount
        // > 2, which prevents automatic unlock.
        debug_assert!(Arc::strong_count(&access_tracker.0) > 2);

        {
            let this = self.inner.read().unwrap();
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
    pub(super) fn try_lock_exclusive(&self) -> Option<GlobalFileCacheGuardExclusive> {
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

            if let Some(ref state) = this.state {
                if matches!(state, LockedState::Exclusive) {
                    return Some(this);
                }
            }

            if this.state.take().is_some() {
                this.file.unlock().unwrap();
            }

            if this.file.try_lock_exclusive().is_ok() {
                this.state = Some(LockedState::Exclusive);
                return Some(this);
            }
        }
        None
    }
}
