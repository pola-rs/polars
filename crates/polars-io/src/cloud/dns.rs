use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::Arc;
use std::time::{Duration, Instant};

use hashbrown::HashMap;
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

#[derive(Debug)]
struct CachedAddrs {
    addrs: Arc<Vec<SocketAddr>>,
    fetched_at: Instant,
}

/// Shuffle resolver with basic DNS cache. TTL is fixed and set by the calling site.
#[derive(Clone, Debug)]
pub struct CachingResolver {
    cache: &'static RwLock<HashMap<String, CachedAddrs>>,
    // Since the OS does not return the TTL as provided by DNS, the calling site
    // is responsible for providing one.
    ttl: Duration,
}

impl CachingResolver {
    pub fn new(ttl: Duration) -> Self {
        Self {
            cache: &DNS_CACHE,
            ttl,
        }
    }
}

impl Resolve for CachingResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let cache = self.cache;
        let ttl = self.ttl;
        let key = name.as_str().to_string();

        Box::pin(async move {
            {
                let read_guard = cache.read().await;

                if let Some(entry) = read_guard.get(&key) {
                    if entry.fetched_at.elapsed() < ttl {
                        return Ok(shuffle_addrs(&entry.addrs));
                    }
                }
            }

            // Cache miss or expired
            let key_clone = key.clone();
            let mut write_guard = cache.write().await;

            // Re-check in case the cache has been populated in the meanwhile
            if let Some(entry) = write_guard.get(&key) {
                if entry.fetched_at.elapsed() < ttl {
                    return Ok(shuffle_addrs(&entry.addrs));
                }
            }

            let addrs = Arc::new(
                polars_core::runtime::ASYNC
                    .spawn_blocking(move || {
                        (key_clone.as_str(), 0u16)
                            .to_socket_addrs()
                            .map(|it| it.collect::<Vec<_>>())
                    })
                    .await
                    .map_err(DynErr::from)??,
            );

            write_guard.insert(
                key,
                CachedAddrs {
                    addrs: addrs.clone(),
                    fetched_at: Instant::now(),
                },
            );
            drop(write_guard);

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
