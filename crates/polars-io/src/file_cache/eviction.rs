use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fs4::fs_std::FileExt;
use polars_error::{PolarsError, PolarsResult};

use super::cache_lock::{GlobalFileCacheGuardExclusive, GLOBAL_FILE_CACHE_LOCK};
use super::metadata::EntryMetadata;
use crate::pl_async;

#[derive(Debug, Clone)]
pub(super) struct EvictionCandidate {
    path: PathBuf,
    metadata_path: PathBuf,
    metadata_last_modified: SystemTime,
    ttl: u64,
}

pub(super) struct EvictionManager {
    pub(super) data_dir: Box<Path>,
    pub(super) metadata_dir: Box<Path>,
    pub(super) files_to_remove: Option<Vec<EvictionCandidate>>,
    pub(super) min_ttl: Arc<AtomicU64>,
    pub(super) notify_ttl_updated: Arc<tokio::sync::Notify>,
}

impl EvictionCandidate {
    fn update_ttl(&mut self) {
        let Ok(metadata_last_modified) =
            std::fs::metadata(&self.metadata_path).map(|md| md.modified().unwrap())
        else {
            self.ttl = 0;
            return;
        };

        if self.metadata_last_modified == metadata_last_modified {
            return;
        }

        let Ok(ref mut file) = std::fs::OpenOptions::new()
            .read(true)
            .open(&self.metadata_path)
        else {
            self.ttl = 0;
            return;
        };

        let ttl = EntryMetadata::try_from_reader(file)
            .map(|x| x.ttl)
            .unwrap_or(0);

        self.metadata_last_modified = metadata_last_modified;
        self.ttl = ttl;
    }

    fn should_remove(&self, now: &SystemTime) -> bool {
        let Ok(metadata) = std::fs::metadata(&self.path) else {
            return false;
        };

        if let Ok(duration) = now.duration_since(
            metadata
                .accessed()
                .unwrap_or_else(|_| metadata.modified().unwrap()),
        ) {
            duration.as_secs() >= self.ttl
        } else {
            false
        }
    }

    fn try_evict(
        &mut self,
        now: &SystemTime,
        verbose: bool,
        _guard: &GlobalFileCacheGuardExclusive,
    ) {
        self.update_ttl();
        let path = &self.path;

        if !path.exists() {
            if verbose {
                eprintln!(
                    "[EvictionManager] evict_files: skipping {} (path no longer exists)",
                    path.to_str().unwrap()
                );
            }
            return;
        }

        let metadata = std::fs::metadata(path).unwrap();

        let since_last_accessed = match now.duration_since(
            metadata
                .accessed()
                .unwrap_or_else(|_| metadata.modified().unwrap()),
        ) {
            Ok(v) => v.as_secs(),
            Err(_) => {
                if verbose {
                    eprintln!("[EvictionManager] evict_files: skipping {} (last accessed time was updated)", path.to_str().unwrap());
                }
                return;
            },
        };

        if since_last_accessed < self.ttl {
            if verbose {
                eprintln!(
                    "[EvictionManager] evict_files: skipping {} (last accessed time was updated)",
                    path.to_str().unwrap()
                );
            }
            return;
        }

        {
            let file = std::fs::OpenOptions::new().read(true).open(path).unwrap();

            if file.try_lock_exclusive().is_err() {
                if verbose {
                    eprintln!(
                        "[EvictionManager] evict_files: skipping {} (file is locked)",
                        self.path.to_str().unwrap()
                    );
                }
                return;
            }
        }

        if let Err(err) = std::fs::remove_file(path) {
            if verbose {
                eprintln!(
                    "[EvictionManager] evict_files: error removing file: {} ({})",
                    path.to_str().unwrap(),
                    err
                );
            }
        } else if verbose {
            eprintln!(
                "[EvictionManager] evict_files: removed file at {}",
                path.to_str().unwrap()
            );
        }
    }
}

impl EvictionManager {
    /// # Safety
    /// The following directories exist:
    /// * `self.data_dir`
    /// * `self.metadata_dir`
    pub(super) fn run_in_background(mut self) {
        let verbose = false;

        if verbose {
            eprintln!(
                "[EvictionManager] creating cache eviction background task, self.min_ttl = {}",
                self.min_ttl.load(std::sync::atomic::Ordering::Relaxed)
            );
        }

        pl_async::get_runtime().spawn(async move {
            // Give some time at startup for other code to run.
            tokio::time::sleep(Duration::from_secs(3)).await;
            let mut last_eviction_time;

            loop {
                let this: &'static mut Self = unsafe { std::mem::transmute(&mut self) };

                let result = tokio::task::spawn_blocking(|| this.update_file_list())
                    .await
                    .unwrap();

                last_eviction_time = Instant::now();

                match result {
                    Ok(_) if self.files_to_remove.as_ref().unwrap().is_empty() => {},
                    Ok(_) => loop {
                        if let Some(guard) = GLOBAL_FILE_CACHE_LOCK.try_lock_exclusive() {
                            if verbose {
                                eprintln!(
                                    "[EvictionManager] got exclusive cache lock, evicting {} files",
                                    self.files_to_remove.as_ref().unwrap().len()
                                );
                            }

                            tokio::task::block_in_place(|| self.evict_files(&guard));
                            break;
                        }
                        tokio::time::sleep(Duration::from_secs(7)).await;
                    },
                    Err(err) => {
                        if verbose {
                            eprintln!("[EvictionManager] error updating file list: {}", err);
                        }
                    },
                }

                loop {
                    let min_ttl = self.min_ttl.load(std::sync::atomic::Ordering::Relaxed);
                    let sleep_interval = std::cmp::max(min_ttl / 4, {
                        #[cfg(debug_assertions)]
                        {
                            3
                        }
                        #[cfg(not(debug_assertions))]
                        {
                            60
                        }
                    });

                    let since_last_eviction =
                        Instant::now().duration_since(last_eviction_time).as_secs();
                    let sleep_interval = sleep_interval.saturating_sub(since_last_eviction);
                    let sleep_interval = Duration::from_secs(sleep_interval);

                    tokio::select! {
                        _ = self.notify_ttl_updated.notified() => {
                            continue;
                        }
                        _ = tokio::time::sleep(sleep_interval) => {
                            break;
                        }
                    }
                }
            }
        });
    }

    fn update_file_list(&mut self) -> PolarsResult<()> {
        let data_files_iter = match std::fs::read_dir(self.data_dir.as_ref()) {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("failed to read data directory: {}", e);

                return Err(PolarsError::IO {
                    error: e.into(),
                    msg: Some(msg.into()),
                });
            },
        };

        let metadata_files_iter = match std::fs::read_dir(self.metadata_dir.as_ref()) {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("failed to read metadata directory: {}", e);

                return Err(PolarsError::IO {
                    error: e.into(),
                    msg: Some(msg.into()),
                });
            },
        };

        let mut files_to_remove = Vec::with_capacity(
            data_files_iter
                .size_hint()
                .1
                .unwrap_or(data_files_iter.size_hint().0)
                + metadata_files_iter
                    .size_hint()
                    .1
                    .unwrap_or(metadata_files_iter.size_hint().0),
        );

        let now = SystemTime::now();

        for file in data_files_iter {
            let file = file?;
            let path = file.path();

            let hash = path
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .get(..32)
                .unwrap();
            let metadata_path = self.metadata_dir.join(hash);

            let mut eviction_candidate = EvictionCandidate {
                path,
                metadata_path,
                metadata_last_modified: UNIX_EPOCH,
                ttl: 0,
            };
            eviction_candidate.update_ttl();

            if eviction_candidate.should_remove(&now) {
                files_to_remove.push(eviction_candidate);
            }
        }

        for file in metadata_files_iter {
            let file = file?;
            let path = file.path();
            let metadata_path = path.clone();

            let mut eviction_candidate = EvictionCandidate {
                path,
                metadata_path,
                metadata_last_modified: UNIX_EPOCH,
                ttl: 0,
            };

            eviction_candidate.update_ttl();

            if eviction_candidate.should_remove(&now) {
                files_to_remove.push(eviction_candidate);
            }
        }

        self.files_to_remove = Some(files_to_remove);

        Ok(())
    }

    /// # Panics
    /// Panics if `self.files_to_remove` is `None`.
    fn evict_files(&mut self, _guard: &GlobalFileCacheGuardExclusive) {
        let verbose = false;
        let mut files_to_remove = self.files_to_remove.take().unwrap();
        let now = &SystemTime::now();

        for eviction_candidate in files_to_remove.iter_mut() {
            eviction_candidate.try_evict(now, verbose, _guard);
        }
    }
}
