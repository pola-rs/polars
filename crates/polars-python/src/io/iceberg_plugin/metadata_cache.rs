//! Cache for immutable Iceberg metadata files read by the plugin through host storage.
//!
//! Counterpart of `polars.io.iceberg._cache` for scans planned by the plugin, with the same
//! scoping (computed in Python, `plugin_storage_scope`). It is owned by the Python
//! `IcebergMetadataFileCache`, so it has the same size (`POLARS_ICEBERG_METADATA_CACHE_MB`) and is
//! dropped by `reset_metadata_file_cache()`.
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use parking_lot::Mutex;
use polars_utils::aliases::PlHashMap;
use pyo3::prelude::*;
use tokio::sync::OnceCell;

/// Whether the file at `location` is immutable and can be cached: manifest lists and manifests
/// (Avro files with a write-time UUID in the name), as in the Python cache, and table metadata
/// files (`<version>-<uuid>.metadata.json`), which the plugin reads where PyIceberg reuses the
/// loaded table.
pub fn is_cacheable(location: &str) -> bool {
    let name = location.rsplit('/').next().unwrap_or(location);
    (name.ends_with(".avro") || name.ends_with(".metadata.json")) && contains_uuid(name)
}

/// Whether `s` contains a UUID (`8-4-4-4-12` hex digits, either case).
fn contains_uuid(s: &str) -> bool {
    const GROUPS: [usize; 5] = [8, 4, 4, 4, 12];
    const LEN: usize = 36;

    let b = s.as_bytes();
    (0..b.len().saturating_sub(LEN - 1)).any(|start| {
        let mut i = start;
        GROUPS.iter().enumerate().all(|(g, &n)| {
            let group_ok = b[i..i + n].iter().all(u8::is_ascii_hexdigit);
            i += n;
            let sep_ok = g == GROUPS.len() - 1 || {
                let ok = b[i] == b'-';
                i += 1;
                ok
            };
            group_ok && sep_ok
        })
    })
}

/// Hit and miss counts.
#[derive(Default)]
pub struct CacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
}

#[derive(Default)]
struct Entries {
    /// Key -> (bytes, recency stamp).
    map: PlHashMap<String, (Bytes, u64)>,
    /// Recency stamp -> key, least recent first.
    order: BTreeMap<u64, String>,
    next_stamp: u64,
    total_bytes: u64,
}

impl Entries {
    fn get(&mut self, key: &str) -> Option<Bytes> {
        let stamp = self.next_stamp;
        let (data, old_stamp) = self.map.get_mut(key)?;
        let key = self.order.remove(old_stamp).unwrap();
        *old_stamp = stamp;
        self.order.insert(stamp, key);
        self.next_stamp += 1;
        Some(data.clone())
    }
}

/// Byte cache with LRU eviction bounded by total size. Keys count towards the size.
pub struct MetadataFileCache {
    max_bytes: u64,
    entries: Mutex<Entries>,
    /// Fetches in progress, so that concurrent misses on one key fetch once.
    in_flight: Mutex<PlHashMap<String, Arc<OnceCell<Bytes>>>>,
}

impl MetadataFileCache {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            entries: Mutex::default(),
            in_flight: Mutex::default(),
        }
    }

    pub fn total_bytes(&self) -> u64 {
        self.entries.lock().total_bytes
    }

    pub fn num_entries(&self) -> usize {
        self.entries.lock().map.len()
    }

    fn get(&self, key: &str) -> Option<Bytes> {
        self.entries.lock().get(key)
    }

    fn put(&self, key: &str, data: Bytes) {
        let size = (key.len() + data.len()) as u64;
        if size > self.max_bytes {
            return;
        }

        let mut entries = self.entries.lock();
        if entries.map.contains_key(key) {
            return;
        }

        let stamp = entries.next_stamp;
        entries.next_stamp += 1;
        entries.map.insert(key.to_owned(), (data, stamp));
        entries.order.insert(stamp, key.to_owned());
        entries.total_bytes += size;

        while entries.total_bytes > self.max_bytes {
            let (_, key) = entries.order.pop_first().unwrap();
            let (data, _) = entries.map.remove(&key).unwrap();
            entries.total_bytes -= (key.len() + data.len()) as u64;
        }
    }

    /// Return the cached bytes of `key`, fetching them on a miss; `stats` counts it.
    pub async fn get_or_fetch<E, F, Fut>(
        &self,
        key: &str,
        stats: &CacheStats,
        fetch: F,
    ) -> Result<Bytes, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Bytes, E>>,
    {
        if let Some(data) = self.get(key) {
            stats.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(data);
        }

        let cell = self
            .in_flight
            .lock()
            .entry(key.to_owned())
            .or_default()
            .clone();

        let mut fetched = false;
        let result = cell
            .get_or_try_init(|| async {
                // Filled by a fetch that finished between the lookup and taking the cell.
                if let Some(data) = self.get(key) {
                    return Ok(data);
                }
                fetched = true;
                let data = fetch().await?;
                self.put(key, data.clone());
                Ok(data)
            })
            .await
            .cloned();

        {
            let mut in_flight = self.in_flight.lock();
            // A newer cell for the same key may have replaced this one.
            if in_flight.get(key).is_some_and(|c| Arc::ptr_eq(c, &cell)) {
                in_flight.remove(key);
            }
        }

        let counter = if fetched { &stats.misses } else { &stats.hits };
        counter.fetch_add(1, Ordering::Relaxed);

        result
    }
}

/// The metadata file cache, with the scope of this scan's storage configuration.
#[derive(Clone)]
pub struct ScopedMetadataCache {
    pub cache: Arc<MetadataFileCache>,
    /// Fingerprint of the storage configuration; cache keys are `{scope}:{url}`.
    pub scope: String,
    /// Cache reads of this scan.
    pub stats: Arc<CacheStats>,
}

/// Handle to a [`MetadataFileCache`], owned by Python's `IcebergMetadataFileCache`.
#[pyclass(frozen)]
pub struct PyIcebergMetadataFileCache(pub Arc<MetadataFileCache>);

#[pymethods]
impl PyIcebergMetadataFileCache {
    #[new]
    fn new(max_bytes: u64) -> Self {
        Self(Arc::new(MetadataFileCache::new(max_bytes)))
    }

    #[getter]
    fn total_bytes(&self) -> u64 {
        self.0.total_bytes()
    }

    fn __len__(&self) -> usize {
        self.0.num_entries()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_cacheable() {
        let uuid = "1f6e4c0a-9B2d-4e5f-8a7b-0c1d2e3f4a5b";
        assert!(is_cacheable(&format!(
            "s3://b/t/metadata/snap-1-0-{uuid}.avro"
        )));
        assert!(is_cacheable(&format!("s3://b/t/metadata/{uuid}-m0.avro")));
        assert!(is_cacheable(&format!(
            "s3://b/t/metadata/00001-{uuid}.metadata.json"
        )));
        assert!(!is_cacheable("s3://b/t/metadata/v1.metadata.json"));
        assert!(!is_cacheable("s3://b/t/metadata/snap-1.avro"));
        assert!(!is_cacheable(&format!("s3://b/{uuid}/data.parquet")));
        assert!(!is_cacheable(&format!("s3://b/{uuid}/x.avro")));
        assert!(!is_cacheable("1f6e4c0a-9b2d-4e5f-8a7b-0c1d2e3f4a5.avro"));
    }

    #[test]
    fn test_lru_eviction_by_size() {
        let cache = MetadataFileCache::new(10);
        cache.put("a", Bytes::from_static(b"1234"));
        cache.put("b", Bytes::from_static(b"1234"));
        assert_eq!(cache.total_bytes(), 10);
        // Touch `a`, so that `b` is evicted.
        assert!(cache.get("a").is_some());
        cache.put("c", Bytes::from_static(b"12"));
        assert!(cache.get("b").is_none());
        assert!(cache.get("a").is_some() && cache.get("c").is_some());
        assert_eq!(cache.total_bytes(), 8);
        // Larger than the cache.
        cache.put("d", Bytes::from_static(b"12345678910"));
        assert!(cache.get("d").is_none());
    }
}
