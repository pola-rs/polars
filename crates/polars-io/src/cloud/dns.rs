use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use futures::stream::{FuturesUnordered, StreamExt};
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

/// Hard-coded DNS TTL cache, as the operating system does not return it with the
/// calls used. Defaults to AWS TTL.
pub(crate) fn get_dns_cache_ttl() -> Duration {
    Duration::from_secs(
        std::env::var("POLARS_DNS_CACHE_TTL_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_DNS_CACHE_TTL_SECS),
    )
}

const DEFAULT_DNS_MAX_STALE_SECS: u64 = 300;

/// Upper limit for serving stale DNS while refresh is happening in the background.
pub(crate) fn get_dns_max_stale() -> Option<Duration> {
    let max_stale = Duration::from_secs(
        std::env::var("POLARS_DNS_MAX_STALE_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_DNS_MAX_STALE_SECS),
    );
    if max_stale.is_zero() {
        None
    } else {
        Some(max_stale)
    }
}

const DEFAULT_DNS_LOOKUP_ATTEMPTS: u64 = 3;

/// Total DNS lookup attempts (timeout-bounded retries + one final unbounded).
pub(crate) fn get_dns_lookup_attempts() -> u64 {
    std::env::var("POLARS_DNS_LOOKUP_ATTEMPTS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_DNS_LOOKUP_ATTEMPTS)
        .max(1)
}

const DEFAULT_DNS_ATTEMPT_TIMEOUT_MS: u64 = 500;

/// DNS lookup attempt timeout before hedging kicks in.
pub(crate) fn get_dns_attempt_timeout() -> Duration {
    let timeout_ms = std::env::var("POLARS_DNS_ATTEMPT_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_DNS_ATTEMPT_TIMEOUT_MS);

    Duration::from_millis(timeout_ms)
}

#[derive(Debug)]
struct CachedAddrs {
    addrs: Arc<Vec<SocketAddr>>,
    fetched_at: Instant,
    /// True while a background refresh for this host is in flight (single-flight gate).
    refreshing: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
pub struct DnsResolverConfig {
    pub ttl: Duration,
    pub max_stale: Option<Duration>,
    pub lookup_attempts: u64,
    pub attempt_timeout: Duration,
}

impl DnsResolverConfig {
    pub fn from_env() -> Self {
        Self {
            ttl: get_dns_cache_ttl(),
            max_stale: get_dns_max_stale(),
            lookup_attempts: get_dns_lookup_attempts(),
            attempt_timeout: get_dns_attempt_timeout(),
        }
    }
}

/// Shuffle resolver with basic DNS cache. TTL is fixed and set by the calling site.
/// The resolver serve policy:
/// - case fresh: serve from cache;
/// - case expired and within max_stale (> ttl): serve stale + single-flight background refresh;
/// - case beyond max_stale (or max_stale = None): blocking resolve
#[derive(Clone, Debug)]
pub struct CachingResolver {
    cache: &'static RwLock<HashMap<String, CachedAddrs>>,
    config: DnsResolverConfig,
}

impl CachingResolver {
    pub fn new(config: DnsResolverConfig) -> Self {
        if polars_config::config().verbose() {
            let max_stale = config
                .max_stale
                .map_or("disabled".to_string(), |m| format!("{}s", m.as_secs()));
            eprintln!(
                "[dns_cache] ttl: {}s, max_stale: {}, lookup_attempts: {}, attempt_timeout: {}ms",
                config.ttl.as_secs(),
                max_stale,
                config.lookup_attempts,
                config.attempt_timeout.as_millis()
            );
        }

        Self {
            cache: &DNS_CACHE,
            config,
        }
    }
}

impl Resolve for CachingResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let cache = self.cache;
        let DnsResolverConfig {
            ttl,
            max_stale,
            lookup_attempts,
            attempt_timeout,
        } = self.config;

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
                                let result =
                                    lookup_hedged(&key, lookup_attempts, attempt_timeout).await;

                                // Swap in the new set only on success; on failure keep
                                // serving the old addrs (next stale hit re-triggers).
                                if let Ok(addrs) = result {
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
            let mut write_guard = cache.write().await;

            // Re-check in case the cache has been populated in the meanwhile
            if let Some(entry) = write_guard.get(&key) {
                let age = entry.fetched_at.elapsed();
                if max_stale.is_some_and(|m| age < m) || age < ttl {
                    return Ok(shuffle_addrs(&entry.addrs));
                }
            }

            let addrs = Arc::new(lookup_hedged(&key, lookup_attempts, attempt_timeout).await?);

            write_guard.insert(
                key,
                CachedAddrs {
                    addrs: addrs.clone(),
                    fetched_at: Instant::now(),
                    refreshing: Arc::new(AtomicBool::new(false)),
                },
            );
            drop(write_guard);

            Ok(shuffle_addrs(&addrs))
        })
    }
}

/// DNS lookup with hedged attempts once the timeout has been exceeded.
async fn lookup_hedged(
    key: &str,
    lookup_attempts: u64,
    attempt_timeout: Duration,
) -> Result<Vec<SocketAddr>, DynErr> {
    let spawn_lookup = |key: String| {
        ASYNC.spawn_blocking(move || {
            (key.as_str(), 0u16)
                .to_socket_addrs()
                .map(|it| it.collect::<Vec<_>>())
        })
    };

    let mut in_flight = FuturesUnordered::new();
    in_flight.push(spawn_lookup(key.to_string()));
    let mut launched = 1;

    let t0 = Instant::now();

    loop {
        let can_hedge = launched < lookup_attempts;

        tokio::select! {
            biased;

            completed = in_flight.next() => {

                let completed: Option<Result<Vec<SocketAddr>, DynErr>> = completed.map(|joined| {
                    joined
                        .map_err(DynErr::from)
                        .and_then(|res| res.map_err(DynErr::from))
                });

                match completed {
                    // First success wins.
                    Some(Ok(addrs)) => {
                        let elapsed = t0.elapsed();

                        if let Some(threshold) = polars_config::config().dns_log_threshold()
                            && elapsed.gt(&threshold)
                        {
                            let display_key = if polars_config::config().verbose_sensitive() {
                                key
                            } else {
                                "<name suppressed>"
                            };
                            eprintln!(
                                "[dns_cache] dns lookup for {} launched {} attempt(s), took {:.1} ms, exceeded threshold of {} ms",
                                display_key,
                                launched,
                                elapsed.as_secs_f64() * 1000.0,
                                threshold.as_secs_f64() * 1000.0,
                            )
                        };

                        return Ok(addrs)},
                    Some(Err(err)) => {
                        if in_flight.is_empty() {
                            if can_hedge {
                                in_flight.push(spawn_lookup(key.to_string()));
                                launched += 1;
                            } else {
                                return Err(err);
                            }
                        }
                    },
                    None => unreachable!("in_flight drained while still looping"),
                }
            }

            _ = tokio::time::sleep(attempt_timeout), if can_hedge => {
                in_flight.push(spawn_lookup(key.to_string()));
                launched += 1;
            }
        }
    }
}

fn shuffle_addrs(addrs: &Arc<Vec<SocketAddr>>) -> Addrs {
    let mut indices: Vec<usize> = (0..addrs.len()).collect();
    fastrand::shuffle(&mut indices);
    let addrs = addrs.clone();
    Box::new(indices.into_iter().map(move |i| addrs[i]))
}
