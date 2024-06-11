use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use once_cell::sync::Lazy;
use polars_core::config;
use polars_error::PolarsResult;
use polars_utils::aliases::PlHashMap;

use super::entry::{FileCacheEntry, DATA_PREFIX, METADATA_PREFIX};
use super::eviction::EvictionManager;
use super::file_fetcher::FileFetcher;
use super::utils::FILE_CACHE_PREFIX;
use crate::prelude::is_cloud_url;

pub static FILE_CACHE: Lazy<FileCache> = Lazy::new(|| {
    let prefix = FILE_CACHE_PREFIX.as_ref();
    let prefix = Arc::<Path>::from(prefix);

    if config::verbose() {
        eprintln!("file cache prefix: {}", prefix.to_str().unwrap());
    }

    EvictionManager {
        prefix: prefix.clone(),
        files_to_remove: None,
        limit_since_last_access: Duration::from_secs(
            std::env::var("POLARS_FILE_CACHE_TTL")
                .map(|x| x.parse::<u64>().expect("integer"))
                .unwrap_or(60 * 60),
        ),
    }
    .run_in_background();

    FileCache::new(prefix)
});

pub struct FileCache {
    prefix: Arc<Path>,
    entries: Arc<RwLock<PlHashMap<Arc<str>, Arc<FileCacheEntry>>>>,
}

impl FileCache {
    fn new(prefix: Arc<Path>) -> Self {
        let path = &prefix
            .as_ref()
            .join(std::str::from_utf8(&[METADATA_PREFIX]).unwrap());
        let _ = std::fs::create_dir_all(path);
        assert!(
            path.is_dir(),
            "failed to create file cache metadata directory: {}",
            path.to_str().unwrap(),
        );

        let path = &prefix
            .as_ref()
            .join(std::str::from_utf8(&[DATA_PREFIX]).unwrap());
        let _ = std::fs::create_dir_all(path);
        assert!(
            path.is_dir(),
            "failed to create file cache data directory: {}",
            path.to_str().unwrap(),
        );

        Self {
            prefix,
            entries: Default::default(),
        }
    }

    /// If `uri` is a local path, it must be an absolute path.
    pub fn init_entry<F: Fn() -> PolarsResult<Arc<dyn FileFetcher>>>(
        &self,
        uri: Arc<str>,
        get_file_fetcher: F,
    ) -> PolarsResult<Arc<FileCacheEntry>> {
        let verbose = config::verbose();

        #[cfg(debug_assertions)]
        {
            // Local paths must be absolute or else the cache would be wrong.
            if !crate::utils::is_cloud_url(uri.as_ref()) {
                let path = Path::new(uri.as_ref());
                assert_eq!(path, std::fs::canonicalize(path).unwrap().as_path());
            }
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
                return Ok(entry.clone());
            }
        }

        let uri_hash = blake3::hash(uri.as_bytes())
            .to_hex()
            .get(..32)
            .unwrap()
            .to_string();

        {
            let mut entries = self.entries.write().unwrap();

            // May have been raced
            if let Some(entry) = entries.get(uri.as_ref()) {
                if verbose {
                    eprintln!("[file_cache] init_entry: return existing entry for uri = {} (lost init race)", uri.clone());
                }
                return Ok(entry.clone());
            }

            if verbose {
                eprintln!(
                    "[file_cache] init_entry: creating new entry for uri = {}, hash = {}",
                    uri.clone(),
                    uri_hash.clone()
                );
            }

            let entry = Arc::new(FileCacheEntry::new(
                uri.clone(),
                uri_hash,
                self.prefix.clone(),
                get_file_fetcher()?,
            ));
            entries.insert_unique_unchecked(uri, entry.clone());
            Ok(entry.clone())
        }
    }

    /// This function can accept relative local paths.
    pub fn get_entry(&self, uri: &str) -> Option<Arc<FileCacheEntry>> {
        if is_cloud_url(uri) {
            self.entries.read().unwrap().get(uri).map(Arc::clone)
        } else {
            let uri = std::fs::canonicalize(uri).unwrap();
            self.entries
                .read()
                .unwrap()
                .get(uri.to_str().unwrap())
                .map(Arc::clone)
        }
    }
}
