use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use hashbrown::HashMap;
use polars_core::runtime::ASYNC;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use tokio::sync::RwLock;

type DynErr = Box<dyn std::error::Error + Send + Sync>;

use std::sync::LazyLock;

const DEFAULT_DNS_CACHE_TTL_SECS: u64 = 5;

/// Process-wide DNS cache shared by all `CachingResolver` instances, so resolved
/// addrs survive client teardown/rebuild (e.g. object-store cache eviction).
///
/// Assumes resolution is process-uniform: the key is the hostname alone, so this
/// must not be shared if per-client resolver config is ever introduced.
///
/// Entries are never evicted. This is ok for object-store endpoints (low cardinality).
/// Revisit with a size cap if keys become externally driven (e.g. per-bucket
/// virtual-hosted hosts at scale).
static DNS_CACHE: LazyLock<RwLock<HashMap<String, CachedAddrs>>> = LazyLock::new(Default::default);

pub(crate) fn get_dns_cache_ttl() -> Duration {
    let ttl = Duration::from_secs(
        std::env::var("POLARS_DNS_CACHE_TTL_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_DNS_CACHE_TTL_SECS),
    );

    if polars_config::config().verbose() {
        eprintln!("[dns_cache] ttl: {}s", ttl.as_secs());
    }

    ttl
}

const DEFAULT_DNS_MAX_STALE_SECS: u64 = 300;

pub(crate) fn get_dns_max_stale() -> Option<Duration> {
    let max_stale = Duration::from_secs(
        std::env::var("POLARS_DNS_MAX_STALE_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_DNS_MAX_STALE_SECS),
    );
    if polars_config::config().verbose() {
        if max_stale.is_zero() {
            eprintln!("[dns_cache] max stale: <stale serve disabled>");
        } else {
            eprintln!("[dns_cache] max stale: {}s", max_stale.as_secs());
        }
    }
    if max_stale.is_zero() {
        None
    } else {
        Some(max_stale)
    }
}

#[derive(Debug)]
struct CachedAddrs {
    addrs: Arc<Vec<SocketAddr>>,
    fetched_at: Instant,
    /// True while a background refresh for this host is in flight (single-flight gate).
    refreshing: Arc<AtomicBool>,
}

/// Shuffle resolver with basic DNS cache. TTL is fixed and set by the calling site.
/// The resolver serve policy:
/// - case fresh: serve from cache;
/// - case expired and within max_stale (> ttl): serve stale + single-flight background refresh;
/// - case beyond max_stale (or max_stale = None): blocking resolve
#[derive(Clone, Debug)]
pub struct CachingResolver {
    cache: &'static RwLock<HashMap<String, CachedAddrs>>,
    // Since the OS does not return the TTL as provided by DNS, the calling site
    // is responsible for providing one.
    ttl: Duration,
    // Upper limit for serve_stale DNS.
    max_stale: Option<Duration>,
}

impl CachingResolver {
    pub fn new(ttl: Duration, max_stale: Option<Duration>) -> Self {
        Self {
            cache: &DNS_CACHE,
            ttl,
            max_stale,
        }
    }
}

impl Resolve for CachingResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let cache = self.cache;
        let ttl = self.ttl;
        let max_stale = self.max_stale;
        let key = name.as_str().to_string();

        Box::pin(async move {
            {
                let read_guard = cache.read().await;

                if let Some(entry) = read_guard.get(&key) {
                    let age = entry.fetched_at.elapsed();
                    if age < ttl {
                        return Ok(shuffle_addrs(&entry.addrs));
                    }

                    if let Some(max_stale) = max_stale
                        && age < max_stale
                    {
                        // Expired: serve stale immediately, refresh in the background.
                        // CAS ensures a burst of stale hits spawns exactly one refresh.
                        if entry
                            .refreshing
                            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                            .is_ok()
                        {
                            let cache = cache.clone();
                            let key = key.clone();
                            let refreshing = entry.refreshing.clone();
                            ASYNC.spawn(async move {
                                let key_clone = key.clone();
                                let result = ASYNC
                                    .spawn_blocking(move || {
                                        (key_clone.as_str(), 0u16)
                                            .to_socket_addrs()
                                            .map(|it| it.collect::<Vec<_>>())
                                    })
                                    .await;

                                // Swap in the new set only on success; on failure keep
                                // serving the old addrs (next stale hit re-triggers).
                                if let Ok(Ok(addrs)) = result {
                                    let mut write_guard = cache.write().await;
                                    write_guard.insert(
                                        key,
                                        CachedAddrs {
                                            addrs: Arc::new(addrs),
                                            fetched_at: Instant::now(),
                                            refreshing: refreshing.clone(),
                                        },
                                    );
                                }
                                refreshing.store(false, Ordering::Release);
                            });
                        }

                        return Ok(shuffle_addrs(&entry.addrs));
                    }
                }
            }

            // Cache miss or expired
            let key_clone = key.clone();
            let mut write_guard = cache.write().await;

            // Re-check in case the cache has been populated in the meanwhile
            if let Some(entry) = write_guard.get(&key) {
                let age = entry.fetched_at.elapsed();
                if max_stale.is_some_and(|m| age < m) || age < ttl {
                    return Ok(shuffle_addrs(&entry.addrs));
                }
            }

            let t0 = Instant::now();
            let addrs = Arc::new(
                ASYNC
                    .spawn_blocking(move || {
                        (key_clone.as_str(), 0u16)
                            .to_socket_addrs()
                            .map(|it| it.collect::<Vec<_>>())
                    })
                    .await
                    .map_err(DynErr::from)??,
            );
            let elapsed = t0.elapsed();

            write_guard.insert(
                key.clone(),
                CachedAddrs {
                    addrs: addrs.clone(),
                    fetched_at: Instant::now(),
                    refreshing: Arc::new(AtomicBool::new(false)),
                },
            );
            drop(write_guard);

            if let Some(threshold) = polars_config::config().dns_log_threshold()
                && elapsed.gt(&threshold)
            {
                let display_key = if polars_config::config().verbose_sensitive() {
                    key.as_str()
                } else {
                    "<name suppressed>"
                };
                eprintln!(
                    "[dns_cache] dns lookup for {} took {:.1} ms, exceeded threshold of {} ms",
                    display_key,
                    elapsed.as_secs_f64() * 1000.0,
                    threshold.as_secs_f64() * 1000.0
                )
            }

            Ok(shuffle_addrs(&addrs))
        })
    }
}

fn shuffle_addrs(addrs: &Arc<Vec<SocketAddr>>) -> Addrs {
    let mut indices: Vec<usize> = (0..addrs.len()).collect();
    fastrand::shuffle(&mut indices);
    let addrs = addrs.clone();
    Box::new(indices.into_iter().map(move |i| addrs[i]))
}
