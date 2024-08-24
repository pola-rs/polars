use std::io::{Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use fs4::fs_std::FileExt;
use polars_core::config;
use polars_error::{polars_bail, to_compute_err, PolarsError, PolarsResult};
use polars_utils::flatten;

use super::cache_lock::{self, GLOBAL_FILE_CACHE_LOCK};
use super::file_fetcher::{FileFetcher, RemoteMetadata};
use super::file_lock::{FileLock, FileLockAnyGuard};
use super::metadata::{EntryMetadata, FileVersion};
use super::utils::update_last_accessed;

pub(super) const DATA_PREFIX: u8 = b'd';
pub(super) const METADATA_PREFIX: u8 = b'm';

struct CachedData {
    last_modified: u64,
    metadata: Arc<EntryMetadata>,
    data_file_path: PathBuf,
}

struct Inner {
    uri: Arc<str>,
    uri_hash: String,
    path_prefix: Arc<Path>,
    metadata: FileLock<PathBuf>,
    cached_data: Option<CachedData>,
    ttl: Arc<AtomicU64>,
    file_fetcher: Arc<dyn FileFetcher>,
}

struct EntryData {
    uri: Arc<str>,
    inner: Mutex<Inner>,
    ttl: Arc<AtomicU64>,
}

pub struct FileCacheEntry(EntryData);

impl EntryMetadata {
    fn matches_remote_metadata(&self, remote_metadata: &RemoteMetadata) -> bool {
        self.remote_version == remote_metadata.version && self.local_size == remote_metadata.size
    }
}

impl Inner {
    fn try_open_assume_latest(&mut self) -> PolarsResult<std::fs::File> {
        let verbose = config::verbose();

        {
            let cache_guard = GLOBAL_FILE_CACHE_LOCK.lock_any();
            // We want to use an exclusive lock here to avoid an API call in the case where only the
            // local TTL was updated.
            let metadata_file = &mut self.metadata.acquire_exclusive().unwrap();
            update_last_accessed(metadata_file);

            if let Ok(metadata) = self.try_get_metadata(metadata_file, &cache_guard) {
                let data_file_path = self.get_cached_data_file_path();

                if metadata.compare_local_state(data_file_path).is_ok() {
                    if verbose {
                        eprintln!("[file_cache::entry] try_open_assume_latest: opening already fetched file for uri = {}", self.uri.clone());
                    }
                    return Ok(finish_open(data_file_path, metadata_file));
                }
            }
        }

        if verbose {
            eprintln!(
                "[file_cache::entry] try_open_assume_latest: did not find cached file for uri = {}",
                self.uri.clone()
            );
        }

        self.try_open_check_latest()
    }

    fn try_open_check_latest(&mut self) -> PolarsResult<std::fs::File> {
        let verbose = config::verbose();
        let remote_metadata = &self.file_fetcher.fetch_metadata()?;
        let cache_guard = GLOBAL_FILE_CACHE_LOCK.lock_any();

        {
            let metadata_file = &mut self.metadata.acquire_shared().unwrap();
            update_last_accessed(metadata_file);

            if let Ok(metadata) = self.try_get_metadata(metadata_file, &cache_guard) {
                if metadata.matches_remote_metadata(remote_metadata) {
                    let data_file_path = self.get_cached_data_file_path();

                    if metadata.compare_local_state(data_file_path).is_ok() {
                        if verbose {
                            eprintln!("[file_cache::entry] try_open_check_latest: opening already fetched file for uri = {}", self.uri.clone());
                        }
                        return Ok(finish_open(data_file_path, metadata_file));
                    }
                }
            }
        }

        let metadata_file = &mut self.metadata.acquire_exclusive().unwrap();
        let metadata = self
            .try_get_metadata(metadata_file, &cache_guard)
            // Safety: `metadata_file` is an exclusive guard.
            .unwrap_or_else(|_| {
                Arc::new(EntryMetadata::new(
                    self.uri.clone(),
                    self.ttl.load(std::sync::atomic::Ordering::Relaxed),
                ))
            });

        if metadata.matches_remote_metadata(remote_metadata) {
            let data_file_path = self.get_cached_data_file_path();

            if metadata.compare_local_state(data_file_path).is_ok() {
                if verbose {
                    eprintln!(
                        "[file_cache::entry] try_open_check_latest: opening already fetched file (lost race) for uri = {}",
                        self.uri.clone()
                    );
                }
                return Ok(finish_open(data_file_path, metadata_file));
            }
        }

        if verbose {
            eprintln!(
                "[file_cache::entry] try_open_check_latest: fetching new data file for uri = {}, remote_version = {:?}, remote_size = {}",
                self.uri.clone(),
                remote_metadata.version,
                remote_metadata.size
            );
        }

        let data_file_path = &get_data_file_path(
            self.path_prefix.to_str().unwrap().as_bytes(),
            self.uri_hash.as_bytes(),
            &remote_metadata.version,
        );
        // Remove the file if it exists, since it doesn't match the metadata.
        // This could be left from an aborted process.
        let _ = std::fs::remove_file(data_file_path);
        if !self.file_fetcher.fetches_as_symlink() {
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(data_file_path)
                .map_err(PolarsError::from)?;
            file.lock_exclusive().unwrap();
            if file.allocate(remote_metadata.size).is_err() {
                polars_bail!(
                    ComputeError: "failed to allocate {} bytes to download uri = {}",
                    remote_metadata.size,
                    self.uri.as_ref()
                );
            }
        }
        self.file_fetcher.fetch(data_file_path)?;

        // Don't do this on windows as it will break setting last accessed times.
        #[cfg(target_family = "unix")]
        if !self.file_fetcher.fetches_as_symlink() {
            let mut perms = std::fs::metadata(data_file_path.clone())
                .unwrap()
                .permissions();
            perms.set_readonly(true);
            std::fs::set_permissions(data_file_path, perms).unwrap();
        }

        let data_file_metadata = std::fs::metadata(data_file_path).unwrap();
        let local_last_modified = super::utils::last_modified_u64(&data_file_metadata);
        let local_size = data_file_metadata.len();

        if local_size != remote_metadata.size {
            polars_bail!(ComputeError: "downloaded file size ({}) does not match expected size ({})", local_size, remote_metadata.size);
        }

        let mut metadata = metadata;
        let metadata = Arc::make_mut(&mut metadata);
        metadata.local_last_modified = local_last_modified;
        metadata.local_size = local_size;
        metadata.remote_version = remote_metadata.version.clone();

        if let Err(e) = metadata.compare_local_state(data_file_path) {
            panic!("metadata mismatch after file fetch: {}", e);
        }

        let data_file = finish_open(data_file_path, metadata_file);

        metadata_file.set_len(0).unwrap();
        metadata_file.seek(SeekFrom::Start(0)).unwrap();
        metadata
            .try_write(&mut **metadata_file)
            .map_err(to_compute_err)?;

        Ok(data_file)
    }

    /// Try to read the metadata from disk. If `F` is an exclusive guard, this
    /// will update the TTL stored in the metadata file if it does not match.
    fn try_get_metadata<F: FileLockAnyGuard>(
        &mut self,
        metadata_file: &mut F,
        _cache_guard: &cache_lock::GlobalFileCacheGuardAny,
    ) -> PolarsResult<Arc<EntryMetadata>> {
        let last_modified = super::utils::last_modified_u64(&metadata_file.metadata().unwrap());
        let ttl = self.ttl.load(std::sync::atomic::Ordering::Relaxed);

        for _ in 0..2 {
            if let Some(ref cached) = self.cached_data {
                if cached.last_modified == last_modified {
                    if cached.metadata.ttl != ttl {
                        polars_bail!(ComputeError: "TTL mismatch");
                    }

                    if cached.metadata.uri != self.uri {
                        unimplemented!(
                            "hash collision: uri1 = {}, uri2 = {}, hash = {}",
                            cached.metadata.uri,
                            self.uri,
                            self.uri_hash,
                        );
                    }

                    return Ok(cached.metadata.clone());
                }
            }

            // Ensure cache is unset if read fails
            self.cached_data = None;

            let mut metadata =
                EntryMetadata::try_from_reader(&mut **metadata_file).map_err(to_compute_err)?;

            // Note this means if multiple processes on the same system set a
            // different TTL for the same path, the metadata file will constantly
            // get overwritten.
            if metadata.ttl != ttl {
                if F::IS_EXCLUSIVE {
                    metadata.ttl = ttl;
                    metadata_file.set_len(0).unwrap();
                    metadata_file.seek(SeekFrom::Start(0)).unwrap();
                    metadata
                        .try_write(&mut **metadata_file)
                        .map_err(to_compute_err)?;
                } else {
                    polars_bail!(ComputeError: "TTL mismatch");
                }
            }

            let metadata = Arc::new(metadata);
            let data_file_path = get_data_file_path(
                self.path_prefix.to_str().unwrap().as_bytes(),
                self.uri_hash.as_bytes(),
                &metadata.remote_version,
            );
            self.cached_data = Some(CachedData {
                last_modified,
                metadata,
                data_file_path,
            });
        }

        unreachable!();
    }

    /// # Panics
    /// Panics if `self.cached_data` is `None`.
    fn get_cached_data_file_path(&self) -> &Path {
        &self.cached_data.as_ref().unwrap().data_file_path
    }
}

impl FileCacheEntry {
    pub(crate) fn new(
        uri: Arc<str>,
        uri_hash: String,
        path_prefix: Arc<Path>,
        file_fetcher: Arc<dyn FileFetcher>,
        file_cache_ttl: u64,
    ) -> Self {
        let metadata = FileLock::from(get_metadata_file_path(
            path_prefix.to_str().unwrap().as_bytes(),
            uri_hash.as_bytes(),
        ));

        debug_assert!(
            Arc::ptr_eq(&uri, file_fetcher.get_uri()),
            "impl error: entry uri != file_fetcher uri"
        );

        let ttl = Arc::new(AtomicU64::from(file_cache_ttl));

        Self(EntryData {
            uri: uri.clone(),
            inner: Mutex::new(Inner {
                uri,
                uri_hash,
                path_prefix,
                metadata,
                cached_data: None,
                ttl: ttl.clone(),
                file_fetcher,
            }),
            ttl,
        })
    }

    pub fn uri(&self) -> &Arc<str> {
        &self.0.uri
    }

    /// Directly returns the cached file if it finds one without checking if
    /// there is a newer version on the remote. This does not make any API calls
    /// if it finds a cached file, otherwise it simply downloads the file.
    pub fn try_open_assume_latest(&self) -> PolarsResult<std::fs::File> {
        self.0.inner.lock().unwrap().try_open_assume_latest()
    }

    /// Returns the cached file after ensuring it is up to date against the remote
    /// This will always perform at least 1 API call for fetching metadata.
    pub fn try_open_check_latest(&self) -> PolarsResult<std::fs::File> {
        self.0.inner.lock().unwrap().try_open_check_latest()
    }

    pub fn update_ttl(&self, ttl: u64) {
        self.0.ttl.store(ttl, std::sync::atomic::Ordering::Relaxed);
    }
}

fn finish_open<F: FileLockAnyGuard>(data_file_path: &Path, _metadata_guard: &F) -> std::fs::File {
    let file = {
        #[cfg(not(target_family = "windows"))]
        {
            std::fs::OpenOptions::new()
                .read(true)
                .open(data_file_path)
                .unwrap()
        }
        // windows requires write access to update the last accessed time
        #[cfg(target_family = "windows")]
        {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(data_file_path)
                .unwrap()
        }
    };
    update_last_accessed(&file);
    if file.try_lock_shared().is_err() {
        panic!(
            "finish_open: could not acquire shared lock on data file at {}",
            data_file_path.to_str().unwrap()
        );
    }
    file
}

/// `[prefix]/d/[uri hash][last modified]`
fn get_data_file_path(
    path_prefix: &[u8],
    uri_hash: &[u8],
    remote_version: &FileVersion,
) -> PathBuf {
    let owned;
    let path = flatten(
        &[
            path_prefix,
            &[b'/', DATA_PREFIX, b'/'],
            uri_hash,
            match remote_version {
                FileVersion::Timestamp(v) => {
                    owned = Some(format!("{:013x}", v));
                    owned.as_deref().unwrap()
                },
                FileVersion::ETag(v) => v.as_str(),
                FileVersion::Uninitialized => panic!("impl error: version not initialized"),
            }
            .as_bytes(),
        ],
        None,
    );
    PathBuf::from(std::str::from_utf8(&path).unwrap())
}

/// `[prefix]/m/[uri hash]`
fn get_metadata_file_path(path_prefix: &[u8], uri_hash: &[u8]) -> PathBuf {
    let bytes = flatten(
        &[path_prefix, &[b'/', METADATA_PREFIX, b'/'], uri_hash],
        None,
    );
    let s = std::str::from_utf8(bytes.as_slice()).unwrap();
    PathBuf::from(s)
}
