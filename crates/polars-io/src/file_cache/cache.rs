use std::path::Path;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, LazyLock, RwLock};

use polars_core::config;
use polars_error::PolarsResult;
use polars_utils::aliases::PlHashMap;
use polars_utils::plpath::PlPathRef;

use super::entry::{DATA_PREFIX, FileCacheEntry, METADATA_PREFIX};
use super::eviction::EvictionManager;
use super::file_fetcher::FileFetcher;
use super::utils::FILE_CACHE_PREFIX;
use crate::path_utils::ensure_directory_init;

pub static FILE_CACHE: LazyLock<FileCache> = LazyLock::new(|| {
    let prefix = FILE_CACHE_PREFIX.as_ref();
    let prefix = Arc::<Path>::from(prefix);

    if config::verbose() {
        eprintln!("file cache prefix: {}", prefix.to_str().unwrap());
    }

    let min_ttl = Arc::new(AtomicU64::from(get_env_file_cache_ttl()));
    let notify_ttl_updated = Arc::new(tokio::sync::Notify::new());

    let metadata_dir = prefix
        .as_ref()
        .join(std::str::from_utf8(&[METADATA_PREFIX]).unwrap())
        .into_boxed_path();
    if let Err(err) = ensure_directory_init(&metadata_dir) {
        panic!(
            "failed to create file cache metadata directory: path = {}, err = {}",
            metadata_dir.to_str().unwrap(),
            err
        )
    }

    let data_dir = prefix
        .as_ref()
        .join(std::str::from_utf8(&[DATA_PREFIX]).unwrap())
        .into_boxed_path();

    if let Err(err) = ensure_directory_init(&data_dir) {
        panic!(
            "failed to create file cache data directory: path = {}, err = {}",
            data_dir.to_str().unwrap(),
            err
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
    prefix: Arc<Path>,
    entries: Arc<RwLock<PlHashMap<Arc<str>, Arc<FileCacheEntry>>>>,
    min_ttl: Arc<AtomicU64>,
    notify_ttl_updated: Arc<tokio::sync::Notify>,
}

impl FileCache {
    /// # Safety
    /// The following directories exist:
    /// * `{prefix}/{METADATA_PREFIX}/`
    /// * `{prefix}/{DATA_PREFIX}/`
    unsafe fn new_unchecked(
        prefix: Arc<Path>,
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
        uri: Arc<str>,
        get_file_fetcher: &dyn Fn() -> PolarsResult<Arc<dyn FileFetcher>>,
        ttl: u64,
    ) -> PolarsResult<Arc<FileCacheEntry>> {
        let verbose = config::verbose();

        #[cfg(debug_assertions)]
        {
            // Local paths must be absolute or else the cache would be wrong.
            if !PlPathRef::new(uri.as_ref()).is_cloud_url() {
                let path = Path::new(uri.as_ref());
                assert_eq!(path, std::fs::canonicalize(path).unwrap().as_path());
            }
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

            if let Some(entry) = entries.get(uri.as_ref()) {
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
            if let Some(entry) = entries.get(uri.as_ref()) {
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
            entries.insert(uri, entry.clone());
            Ok(entry)
        }
    }

    /// This function can accept relative local paths.
    pub fn get_entry(&self, addr: PlPathRef<'_>) -> Option<Arc<FileCacheEntry>> {
        match addr {
            PlPathRef::Local(p) => {
                let p = std::fs::canonicalize(p).unwrap();
                self.entries
                    .read()
                    .unwrap()
                    .get(p.to_str().unwrap())
                    .map(Arc::clone)
            },
            PlPathRef::Cloud(p) => self.entries.read().unwrap().get(p.uri()).map(Arc::clone),
        }
    }
}

pub fn get_env_file_cache_ttl() -> u64 {
    std::env::var("POLARS_FILE_CACHE_TTL")
        .map(|x| x.parse::<u64>().expect("integer"))
        .unwrap_or(60 * 60)
}
