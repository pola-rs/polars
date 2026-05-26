use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::Arc;
use std::time::{Duration, Instant};

use hashbrown::HashMap;
use rand::prelude::SliceRandom;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use tokio::sync::RwLock;

type DynErr = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_CLOUD_DNS_CACHE_TTL_SECS: u64 = 5;

pub(crate) fn get_cloud_dns_cache_ttl() -> Duration {
    Duration::from_secs(
        std::env::var("POLARS_CLOUD_DNS_CACHE_TTL_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_CLOUD_DNS_CACHE_TTL_SECS),
    )
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
                        let mut addrs = (*entry.addrs).clone();
                        addrs.shuffle(&mut rand::rng());
                        return Ok(Box::new(addrs.into_iter()) as Addrs);
                    }
                }
            }

            // Cache miss or expired
            let key_clone = key.clone();
            let mut write_guard = cache.write().await;

            // Re-check in case the cache has been populated in the meanwhile
            if let Some(entry) = write_guard.get(&key) {
                if entry.fetched_at.elapsed() < ttl {
                    let mut addrs = (*entry.addrs).clone();
                    addrs.shuffle(&mut rand::rng());
                    return Ok(Box::new(addrs.into_iter()) as Addrs);
                }
            }

            let addrs = Arc::new(
                tokio::task::spawn_blocking(move || {
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

            let mut addrs = (*addrs).clone();
            addrs.shuffle(&mut rand::rng());

            Ok(Box::new(addrs.into_iter()) as Addrs)
        })
    }
}
