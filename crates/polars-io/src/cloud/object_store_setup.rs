use std::sync::Arc;

use object_store::local::LocalFileSystem;
use object_store::ObjectStore;
use once_cell::sync::Lazy;
use polars_core::config::{self, verbose_print_sensitive};
use polars_error::{polars_bail, to_compute_err, PolarsError, PolarsResult};
use polars_utils::aliases::PlHashMap;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{format_pl_smallstr, pl_serialize};
use tokio::sync::RwLock;
use url::Url;

use super::{parse_url, CloudLocation, CloudOptions, CloudType, PolarsObjectStore};
use crate::cloud::CloudConfig;

/// Object stores must be cached. Every object-store will do DNS lookups and
/// get rate limited when querying the DNS (can take up to 5s).
/// Other reasons are connection pools that must be shared between as much as possible.
#[allow(clippy::type_complexity)]
static OBJECT_STORE_CACHE: Lazy<RwLock<PlHashMap<Vec<u8>, PolarsObjectStore>>> =
    Lazy::new(Default::default);

#[allow(dead_code)]
fn err_missing_feature(feature: &str, scheme: &str) -> PolarsResult<Arc<dyn ObjectStore>> {
    polars_bail!(
        ComputeError:
        "feature '{}' must be enabled in order to use '{}' cloud urls", feature, scheme,
    );
}

/// Get the key of a url for object store registration.
fn url_and_creds_to_key(url: &Url, options: Option<&CloudOptions>) -> Vec<u8> {
    // We include credentials as they can expire, so users will send new credentials for the same url.
    let cloud_options = options.map(
        |CloudOptions {
             // Destructure to ensure this breaks if anything changes.
             max_retries,
             #[cfg(feature = "file_cache")]
             file_cache_ttl,
             config,
             #[cfg(feature = "cloud")]
             credential_provider,
         }| {
            CloudOptions2 {
                max_retries: *max_retries,
                #[cfg(feature = "file_cache")]
                file_cache_ttl: *file_cache_ttl,
                config: config.clone(),
                #[cfg(feature = "cloud")]
                credential_provider: credential_provider.as_ref().map_or(0, |x| x.func_addr()),
            }
        },
    );

    let cache_key = CacheKey {
        url_base: format_pl_smallstr!(
            "{}",
            &url[url::Position::BeforeScheme..url::Position::AfterPort]
        ),
        cloud_options,
    };

    verbose_print_sensitive(|| format!("object store cache key: {} {:?}", url, &cache_key));

    return pl_serialize::serialize_to_bytes(&cache_key).unwrap();

    #[derive(Clone, Debug, PartialEq, Hash, Eq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize))]
    struct CacheKey {
        url_base: PlSmallStr,
        cloud_options: Option<CloudOptions2>,
    }

    /// Variant of CloudOptions for serializing to a cache key. The credential
    /// provider is replaced by the function address.
    #[derive(Clone, Debug, PartialEq, Hash, Eq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize))]
    struct CloudOptions2 {
        max_retries: usize,
        #[cfg(feature = "file_cache")]
        file_cache_ttl: u64,
        config: Option<CloudConfig>,
        #[cfg(feature = "cloud")]
        credential_provider: usize,
    }
}

/// Construct an object_store `Path` from a string without any encoding/decoding.
pub fn object_path_from_str(path: &str) -> PolarsResult<object_store::path::Path> {
    object_store::path::Path::parse(path).map_err(to_compute_err)
}

#[derive(Debug, Clone)]
pub(crate) struct PolarsObjectStoreBuilder {
    url: PlSmallStr,
    parsed_url: Url,
    #[allow(unused)]
    scheme: PlSmallStr,
    cloud_type: CloudType,
    options: Option<CloudOptions>,
}

impl PolarsObjectStoreBuilder {
    pub(super) async fn build_impl(&self) -> PolarsResult<Arc<dyn ObjectStore>> {
        let options = self
            .options
            .as_ref()
            .unwrap_or_else(|| CloudOptions::default_static_ref());

        let store = match self.cloud_type {
            CloudType::Aws => {
                #[cfg(feature = "aws")]
                {
                    let store = options.build_aws(&self.url).await?;
                    Ok::<_, PolarsError>(Arc::new(store) as Arc<dyn ObjectStore>)
                }
                #[cfg(not(feature = "aws"))]
                return err_missing_feature("aws", &self.scheme);
            },
            CloudType::Gcp => {
                #[cfg(feature = "gcp")]
                {
                    let store = options.build_gcp(&self.url)?;
                    Ok::<_, PolarsError>(Arc::new(store) as Arc<dyn ObjectStore>)
                }
                #[cfg(not(feature = "gcp"))]
                return err_missing_feature("gcp", &self.scheme);
            },
            CloudType::Azure => {
                {
                    #[cfg(feature = "azure")]
                    {
                        let store = options.build_azure(&self.url)?;
                        Ok::<_, PolarsError>(Arc::new(store) as Arc<dyn ObjectStore>)
                    }
                }
                #[cfg(not(feature = "azure"))]
                return err_missing_feature("azure", &self.scheme);
            },
            CloudType::File => {
                let local = LocalFileSystem::new();
                Ok::<_, PolarsError>(Arc::new(local) as Arc<dyn ObjectStore>)
            },
            CloudType::Http => {
                {
                    #[cfg(feature = "http")]
                    {
                        let store = options.build_http(&self.url)?;
                        PolarsResult::Ok(Arc::new(store) as Arc<dyn ObjectStore>)
                    }
                }
                #[cfg(not(feature = "http"))]
                return err_missing_feature("http", &cloud_location.scheme);
            },
            CloudType::Hf => panic!("impl error: unresolved hf:// path"),
        }?;

        Ok(store)
    }

    /// Note: Use `build_impl` for a non-caching version.
    pub(super) async fn build(self) -> PolarsResult<PolarsObjectStore> {
        let opt_cache_key = match &self.cloud_type {
            CloudType::Aws | CloudType::Gcp | CloudType::Azure => Some(url_and_creds_to_key(
                &self.parsed_url,
                self.options.as_ref(),
            )),
            CloudType::File | CloudType::Http | CloudType::Hf => None,
        };

        let opt_cache_write_guard = if let Some(cache_key) = opt_cache_key.as_deref() {
            let cache = OBJECT_STORE_CACHE.read().await;

            if let Some(store) = cache.get(cache_key) {
                return Ok(store.clone());
            }

            drop(cache);

            let cache = OBJECT_STORE_CACHE.write().await;

            if let Some(store) = cache.get(cache_key) {
                return Ok(store.clone());
            }

            Some(cache)
        } else {
            None
        };

        let store = self.build_impl().await?;
        let store = PolarsObjectStore::new_from_inner(store, self);

        if let Some(mut cache) = opt_cache_write_guard {
            // Clear the cache if we surpass a certain amount of buckets.
            if cache.len() >= 8 {
                if config::verbose() {
                    eprintln!(
                        "build_object_store: clearing store cache (cache.len(): {})",
                        cache.len()
                    );
                }
                cache.clear()
            }

            cache.insert(opt_cache_key.unwrap(), store.clone());
        }

        Ok(store)
    }

    pub(crate) fn is_azure(&self) -> bool {
        matches!(&self.cloud_type, CloudType::Azure)
    }
}

/// Build an [`ObjectStore`] based on the URL and passed in url. Return the cloud location and an implementation of the object store.
pub async fn build_object_store(
    url: &str,
    #[cfg_attr(
        not(any(feature = "aws", feature = "gcp", feature = "azure")),
        allow(unused_variables)
    )]
    options: Option<&CloudOptions>,
    glob: bool,
) -> PolarsResult<(CloudLocation, PolarsObjectStore)> {
    let parsed = parse_url(url).map_err(to_compute_err)?;
    let cloud_location = CloudLocation::from_url(&parsed, glob)?;
    let cloud_type = CloudType::from_url(&parsed)?;

    let store = PolarsObjectStoreBuilder {
        url: url.into(),
        parsed_url: parsed,
        scheme: cloud_location.scheme.as_str().into(),
        cloud_type,
        options: options.cloned(),
    }
    .build()
    .await?;

    Ok((cloud_location, store))
}

mod test {
    #[test]
    fn test_object_path_from_str() {
        use super::object_path_from_str;

        let path = "%25";
        let out = object_path_from_str(path).unwrap();

        assert_eq!(out.as_ref(), path);
    }
}
