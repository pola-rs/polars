use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use hashbrown::HashMap;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use tokio::sync::RwLock;

type DynErr = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_DNS_CACHE_TTL_SECS: u64 = 5;

static POLARS_DNS_LOG_THRESHOLD: LazyLock<Option<Duration>> = LazyLock::new(|| {
    let v: Option<Duration> =
        std::env::var("POLARS_DNS_LOG_THRESHOLD_MS")
            .ok()
            .and_then(|x| match x.trim().parse::<u64>() {
                Ok(ms) => Some(Duration::from_millis(ms)),
                Err(_) => {
                    if polars_config::config().verbose() {
                        eprintln!(
                            "[dns_cache] ignoring invalid POLARS_DNS_LOG_THRESHOLD_MS={x:?} \
                         (expected integer milliseconds)"
                        );
                    }
                    None
                },
            });

    if let Some(v) = v
        && polars_config::config().verbose()
    {
        eprintln!(
            "[dns_cache] dns log threshold: {} ms",
            v.as_secs_f64() * 1000.0
        )
    }

    v
});

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
    cache: Arc<RwLock<HashMap<String, CachedAddrs>>>,
    // Since the OS does not return the TTL as provided by DNS, the calling site
    // is responsible for providing one.
    ttl: Duration,
}

impl CachingResolver {
    pub fn new(ttl: Duration) -> Self {
        Self {
            cache: Arc::new(RwLock::default()),
            ttl,
        }
    }
}

impl Resolve for CachingResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let cache = self.cache.clone();
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

            let t0 = Instant::now();
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
            let elapsed = t0.elapsed();

            write_guard.insert(
                key.clone(),
                CachedAddrs {
                    addrs: addrs.clone(),
                    fetched_at: Instant::now(),
                },
            );
            drop(write_guard);

            if let Some(threshold) = POLARS_DNS_LOG_THRESHOLD.as_ref()
                && elapsed.gt(threshold)
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
