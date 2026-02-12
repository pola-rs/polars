use std::sync::atomic::AtomicU64;
use std::sync::{Arc, LazyLock, RwLock};

use polars_core::config;
use polars_error::PolarsResult;
use polars_utils::aliases::PlHashMap;
use polars_utils::pl_path::PlRefPath;

use super::entry::{DATA_PREFIX, FileCacheEntry, METADATA_PREFIX};
use super::eviction::EvictionManager;
use super::file_fetcher::FileFetcher;
use super::utils::FILE_CACHE_PREFIX;
use crate::path_utils::ensure_directory_init;

pub static FILE_CACHE: LazyLock<FileCache> = LazyLock::new(|| {
    let prefix = FILE_CACHE_PREFIX.clone();

    if config::verbose() {
        eprintln!("file cache prefix: {}", prefix);
    }

    let min_ttl = Arc::new(AtomicU64::from(get_env_file_cache_ttl()));
    let notify_ttl_updated = Arc::new(tokio::sync::Notify::new());

    let metadata_dir = prefix.join(std::str::from_utf8(&[METADATA_PREFIX]).unwrap());
    if let Err(err) = ensure_directory_init(metadata_dir.as_std_path()) {
        panic!(
            "failed to create file cache metadata directory: path = {}, err = {}",
            metadata_dir, err
        )
    }

    let data_dir = prefix.join(std::str::from_utf8(&[DATA_PREFIX]).unwrap());

    if let Err(err) = ensure_directory_init(data_dir.as_std_path()) {
        panic!(
            "failed to create file cache data directory: path = {}, err = {}",
            data_dir, err
        )
    }

    EvictionManager {
        data_dir,
        metadata_dir,
        files_to_remove: None,
        min_ttl: min_ttl.clone(),
        notify_ttl_updated: notify_ttl_updated.clone(),
    }
    .run_in_background();

    // Safety: We have created the data and metadata directories.
    unsafe { FileCache::new_unchecked(prefix, min_ttl, notify_ttl_updated) }
});

pub struct FileCache {
    prefix: PlRefPath,
    entries: Arc<RwLock<PlHashMap<PlRefPath, Arc<FileCacheEntry>>>>,
    min_ttl: Arc<AtomicU64>,
    notify_ttl_updated: Arc<tokio::sync::Notify>,
}

impl FileCache {
    /// # Safety
    /// The following directories exist:
    /// * `{prefix}/{METADATA_PREFIX}/`
    /// * `{prefix}/{DATA_PREFIX}/`
    unsafe fn new_unchecked(
        prefix: PlRefPath,
        min_ttl: Arc<AtomicU64>,
        notify_ttl_updated: Arc<tokio::sync::Notify>,
    ) -> Self {
        Self {
            prefix,
            entries: Default::default(),
            min_ttl,
            notify_ttl_updated,
        }
    }

    /// If `uri` is a local path, it must be an absolute path. This is not exposed
    /// for now - initialize entries using `init_entries_from_uri_list` instead.
    pub(super) fn init_entry(
        &self,
        uri: PlRefPath,
        get_file_fetcher: &dyn Fn() -> PolarsResult<Arc<dyn FileFetcher>>,
        ttl: u64,
    ) -> PolarsResult<Arc<FileCacheEntry>> {
        let verbose = config::verbose();

        // Local paths must be absolute or else the cache would be wrong.
        if !uri.has_scheme() {
            debug_assert_eq!(
                std::fs::canonicalize(uri.as_str())
                    .ok()
                    .and_then(|x| PlRefPath::try_from_pathbuf(x).ok())
                    .as_ref(),
                Some(&uri)
            )
        }

        if self
            .min_ttl
            .fetch_min(ttl, std::sync::atomic::Ordering::Relaxed)
            < ttl
        {
            self.notify_ttl_updated.notify_one();
        }

        {
            let entries = self.entries.read().unwrap();

            if let Some(entry) = entries.get(&uri) {
                if verbose {
                    eprintln!(
                        "[file_cache] init_entry: return existing entry for uri = {}",
                        uri.clone()
                    );
                }
                entry.update_ttl(ttl);
                return Ok(entry.clone());
            }
        }

        let uri_hash = blake3::hash(uri.as_bytes()).to_hex()[..32].to_string();

        {
            let mut entries = self.entries.write().unwrap();

            // May have been raced
            if let Some(entry) = entries.get(&uri) {
                if verbose {
                    eprintln!(
                        "[file_cache] init_entry: return existing entry for uri = {} (lost init race)",
                        uri.clone()
                    );
                }
                entry.update_ttl(ttl);
                return Ok(entry.clone());
            }

            if verbose {
                eprintln!(
                    "[file_cache] init_entry: creating new entry for uri = {uri}, hash = {uri_hash}"
                );
            }

            let entry = Arc::new(FileCacheEntry::new(
                uri.clone(),
                uri_hash,
                self.prefix.clone(),
                get_file_fetcher()?,
                ttl,
            ));
            entries.insert(uri.clone(), entry.clone());
            Ok(entry)
        }
    }

    /// This function can accept relative local paths.
    pub fn get_entry(&self, path: PlRefPath) -> Option<Arc<FileCacheEntry>> {
        if path.has_scheme() {
            self.entries.read().unwrap().get(&path).cloned()
        } else {
            let p =
                PlRefPath::try_from_pathbuf(std::fs::canonicalize(path.as_str()).unwrap()).unwrap();
            self.entries.read().unwrap().get(&p).cloned()
        }
    }
}

pub fn get_env_file_cache_ttl() -> u64 {
    std::env::var("POLARS_FILE_CACHE_TTL")
        .map(|x| x.parse::<u64>().expect("integer"))
        .unwrap_or(60 * 60)
}
