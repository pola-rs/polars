use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use fs4::FileExt;
use polars_core::config;
use polars_error::PolarsResult;

use super::cache_lock::{GlobalFileCacheGuardExclusive, GLOBAL_FILE_CACHE_LOCK};
use crate::pl_async;

pub(super) struct EvictionManager {
    pub(super) prefix: Arc<Path>,
    pub(super) files_to_remove: Option<(SystemTime, Vec<PathBuf>)>,
    pub(super) limit_since_last_access: Duration,
}

impl EvictionManager {
    pub(super) fn run_in_background(mut self) {
        let verbose = config::verbose();
        let sleep_interval = std::cmp::max(self.limit_since_last_access.as_secs() / 4, 60);

        if verbose {
            eprintln!(
                "[EvictionManager] creating cache eviction background task with limit_since_last_accessed = {}, sleep_interval = {}",
                self.limit_since_last_access.as_secs(),
                sleep_interval,
            );
        }

        pl_async::get_runtime().spawn(async move {
            // Give some time at startup for other code to run.
            tokio::time::sleep(Duration::from_secs(3)).await;

            loop {
                let mut sleep_interval = sleep_interval;

                let this: &'static mut Self = unsafe { std::mem::transmute(&mut self) };

                match tokio::task::spawn_blocking(|| this.update_file_list())
                    .await
                    .unwrap()
                {
                    Ok(_) if self.files_to_remove.as_ref().unwrap().1.is_empty() => {},
                    Ok(_) => loop {
                        if let Some(guard) = GLOBAL_FILE_CACHE_LOCK.try_lock_exclusive() {
                            if verbose {
                                eprintln!(
                                    "[EvictionManager] got exclusive cache lock, evicting {} files",
                                    self.files_to_remove.as_ref().unwrap().1.len()
                                );
                            }

                            tokio::task::block_in_place(|| self.evict_files(&guard));
                            break;
                        }
                        sleep_interval = sleep_interval.saturating_sub(7);
                        tokio::time::sleep(Duration::from_secs(7)).await;
                    },
                    Err(err) => {
                        if verbose {
                            eprintln!("[EvictionManager] error updating file list: {}", err);
                        }
                    },
                }

                tokio::time::sleep(Duration::from_secs(sleep_interval)).await;
            }
        });
    }

    fn update_file_list(&mut self) -> PolarsResult<()> {
        let data_dir = &self.prefix.join("d/");
        let metadata_dir = &self.prefix.join("m/");

        let data_files_iter = std::fs::read_dir(data_dir).unwrap();
        let metadata_files_iter = std::fs::read_dir(metadata_dir).unwrap();
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

        let mut f = |file: std::fs::DirEntry| {
            let metadata = file.metadata()?;

            if let Ok(since_last_accessed) = now.duration_since(
                metadata
                    .accessed()
                    .unwrap_or_else(|_| metadata.modified().unwrap()),
            ) {
                if since_last_accessed >= self.limit_since_last_access {
                    files_to_remove.push(file.path());
                }
            }

            std::io::Result::Ok(())
        };

        for file in data_files_iter {
            f(file?)?;
        }

        for file in metadata_files_iter {
            f(file?)?;
        }

        self.files_to_remove = Some((now, files_to_remove));

        Ok(())
    }

    /// # Panics
    /// Panics if `self.files_to_remove` is `None`.
    fn evict_files(&mut self, _guard: &GlobalFileCacheGuardExclusive) {
        let verbose = config::verbose();
        let files_to_remove = self.files_to_remove.take().unwrap().1;
        let now = SystemTime::now();

        for path in &files_to_remove {
            if !path.exists() {
                if verbose {
                    eprintln!(
                        "[EvictionManager] evict_files: skipping {} (path no longer exists)",
                        path.to_str().unwrap()
                    );
                }
                continue;
            }

            let metadata = std::fs::metadata(path).unwrap();

            let since_last_accessed = match now.duration_since(
                metadata
                    .accessed()
                    .unwrap_or_else(|_| metadata.modified().unwrap()),
            ) {
                Ok(v) => v,
                Err(_) => {
                    if verbose {
                        eprintln!("[EvictionManager] evict_files: skipping {} (last accessed time was updated)", path.to_str().unwrap());
                    }
                    continue;
                },
            };

            if since_last_accessed < self.limit_since_last_access {
                if verbose {
                    eprintln!(
                        "[EvictionManager] evict_files: skipping {} (last accessed time was updated)",
                        path.to_str().unwrap()
                    );
                }
                continue;
            }

            let file = std::fs::OpenOptions::new().read(true).open(path).unwrap();

            if file.try_lock_exclusive().is_err() {
                if verbose {
                    eprintln!(
                        "[EvictionManager] evict_files: skipping {} (file is locked)",
                        path.to_str().unwrap()
                    );
                }
                continue;
            }

            drop(file);

            if let Err(err) = std::fs::remove_file(path) {
                if verbose {
                    eprintln!(
                        "[EvictionManager] evict_files: error removing file: {} ({})",
                        path.to_str().unwrap(),
                        err
                    );
                }
            } else {
                eprintln!(
                    "[EvictionManager] evict_files: removed file at {}",
                    path.to_str().unwrap()
                );
            }
        }
    }
}
