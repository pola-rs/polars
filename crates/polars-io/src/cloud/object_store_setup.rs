use std::sync::{Arc, LazyLock};

use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use polars_core::config::{self, verbose, verbose_print_sensitive};
use polars_error::{PolarsError, PolarsResult, polars_bail, polars_err, to_compute_err};
use polars_utils::aliases::PlHashMap;
use polars_utils::pl_path::{CloudScheme, PlPath, PlRefPath};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{format_pl_smallstr, pl_serialize};
use tokio::sync::RwLock;

use super::{CloudLocation, CloudOptions, CloudType, PolarsObjectStore};
use crate::cloud::{CloudConfig, CloudRetryConfig};

/// Object stores must be cached. Every object-store will do DNS lookups and
/// get rate limited when querying the DNS (can take up to 5s).
/// Other reasons are connection pools that must be shared between as much as possible.
#[allow(clippy::type_complexity)]
static OBJECT_STORE_CACHE: LazyLock<RwLock<PlHashMap<Vec<u8>, PolarsObjectStore>>> =
    LazyLock::new(Default::default);

/// Trait for external ObjectStore builder (e.g., for HDFS). Unstable.
pub trait ExtObjectStoreBuilder {
    /// Build new object_store.
    fn build(
        &self,
        url: &PlRefPath,
        options: Option<&CloudOptions>,
    ) -> PolarsResult<Arc<dyn ObjectStore + Send + Sync>>;

    /// Return a stable cache key for this store.
    /// Defaults to `None`, which uses the default key (URL authority + serialised CloudOptions).
    fn stable_cache_key(
        &self,
        _url: &PlRefPath,
        _options: Option<&CloudOptions>,
    ) -> Option<Vec<u8>> {
        None
    }
}

static EXT_OBJECT_STORE_BUILDER_REGISTRY: LazyLock<
    RwLock<PlHashMap<String, Arc<dyn ExtObjectStoreBuilder + Send + Sync>>>,
> = LazyLock::new(Default::default);

/// Register custom object_store builder for a given cloud scheme.
/// Example: for 'hdfs://', the scheme is "hdfs".
/// Rejects native cloud schemes (e.g. "s3").
pub async fn register_object_store_builder(
    scheme: &str,
    builder: Arc<dyn ExtObjectStoreBuilder + Send + Sync>,
) -> PolarsResult<()> {

    // Reject schemes already handled natively.
    // TODO: allow shadowing of existing schemes.
    if CloudScheme::is_native_str(scheme) {
        polars_bail!(
            InvalidOperation:
            "cannot register object_store_builder for scheme '{}': \
             this scheme is handled natively",
            scheme
        );
    }

    if polars_config::config().verbose() {
        eprintln!(
            "[ObjectStoreBuilderRegistry]: register object_store_builder for scheme '{scheme}'"
        )
    }

    EXT_OBJECT_STORE_BUILDER_REGISTRY
        .write()
        .await
        .insert(scheme.to_string(), builder);
    Ok(())
}

pub async fn deregister_object_store_builder(scheme: &str) {
    if polars_config::config().verbose() {
        eprintln!(
            "[ObjectStoreBuilderRegistry]: deregister object_store_builder for scheme '{scheme}'"
        )
    }

    EXT_OBJECT_STORE_BUILDER_REGISTRY
        .write()
        .await
        .remove(scheme);
}

#[allow(dead_code)]
fn err_missing_feature(
    feature: &str,
    cloud_type: &CloudType,
) -> PolarsResult<Arc<dyn ObjectStore>> {
    polars_bail!(
        ComputeError:
        "feature '{}' must be enabled in order to use '{:?}' cloud urls",
        feature,
        cloud_type,
    );
}

/// Get the key of a url for object store registration.
fn path_and_creds_to_key(path: &PlPath, options: Option<&CloudOptions>) -> PolarsResult<Vec<u8>> {
    // We include credentials as they can expire, so users will send new credentials for the same url.

    #[cfg(feature = "cloud")]
    let credential_cache_key = CacheKeyBytes(
        options
            .and_then(|o| o.credential_provider.as_ref())
            .map(|x| x.stable_cache_key())
            .transpose()?
            .unwrap_or_default(),
    );

    let cloud_options = options
        .map(
            |CloudOptions {
                 // Destructure to ensure this breaks if anything changes.
                 #[cfg(feature = "file_cache")]
                 file_cache_ttl,
                 config,
                 retry_config,
                 #[cfg(feature = "cloud")]
                     credential_provider: _,
             }|
             -> PolarsResult<CloudOptionsKey> {
                Ok(CloudOptionsKey {
                    #[cfg(feature = "file_cache")]
                    file_cache_ttl: *file_cache_ttl,
                    config: config.clone(),
                    retry_config: *retry_config,
                    #[cfg(feature = "cloud")]
                    credential_provider: credential_cache_key,
                })
            },
        )
        .transpose()?;

    let cache_key = CacheKey {
        url_base: format_pl_smallstr!("{}", &path.as_str()[..path.authority_end_position()]),
        cloud_options,
    };

    verbose_print_sensitive(|| {
        format!(
            "object store cache key for path at '{}': {:?}",
            path, &cache_key
        )
    });

    return pl_serialize::serialize_to_bytes::<_, false>(&cache_key);

    #[derive(Clone, Debug, PartialEq, Hash, Eq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize))]
    struct CacheKey {
        url_base: PlSmallStr,
        cloud_options: Option<CloudOptionsKey>,
    }

    #[derive(Clone, PartialEq, Hash, Eq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize))]
    struct CacheKeyBytes(Vec<u8>);

    impl std::fmt::Debug for CacheKeyBytes {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            if self.0.is_empty() {
                write!(f, "None")
            } else {
                for b in &self.0 {
                    write!(f, "{:02x}", b)?;
                }
                Ok(())
            }
        }
    }

    /// Variant of CloudOptions for serializing to a cache key. The credential
    /// provider is replaced by the function address.
    #[derive(Clone, Debug, PartialEq, Hash, Eq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize))]
    struct CloudOptionsKey {
        #[cfg(feature = "file_cache")]
        file_cache_ttl: u64,
        config: Option<CloudConfig>,
        retry_config: CloudRetryConfig,
        #[cfg(feature = "cloud")]
        credential_provider: CacheKeyBytes,
    }
}

/// Construct an object_store `Path` from a string without any encoding/decoding.
pub fn object_path_from_str(path: &str) -> PolarsResult<object_store::path::Path> {
    object_store::path::Path::parse(path).map_err(to_compute_err)
}

#[derive(Debug, Clone)]
pub(crate) struct PolarsObjectStoreBuilder {
    path: PlRefPath,
    cloud_type: CloudType,
    options: Option<CloudOptions>,
}

impl PolarsObjectStoreBuilder {
    pub(super) fn path(&self) -> &PlRefPath {
        &self.path
    }

    pub(super) async fn build_impl(
        &self,
        // Whether to clear cached credentials for Python credential providers.
        clear_cached_credentials: bool,
    ) -> PolarsResult<Arc<dyn ObjectStore>> {
        let options = self
            .options
            .as_ref()
            .unwrap_or_else(|| CloudOptions::default_static_ref());

        if let Some(options) = &self.options
            && verbose()
        {
            eprintln!(
                "build object-store: file_cache_ttl: {}",
                options.file_cache_ttl
            )
        }

        let store = match &self.cloud_type {
            CloudType::Aws => {
                #[cfg(feature = "aws")]
                {
                    let store = options
                        .build_aws(self.path.clone(), clear_cached_credentials)
                        .await?;
                    Ok::<_, PolarsError>(Arc::new(store) as Arc<dyn ObjectStore>)
                }
                #[cfg(not(feature = "aws"))]
                return err_missing_feature("aws", &self.cloud_type);
            },
            CloudType::Gcp => {
                #[cfg(feature = "gcp")]
                {
                    let store = options.build_gcp(self.path.clone(), clear_cached_credentials)?;

                    Ok::<_, PolarsError>(Arc::new(store) as Arc<dyn ObjectStore>)
                }
                #[cfg(not(feature = "gcp"))]
                return err_missing_feature("gcp", &self.cloud_type);
            },
            CloudType::Azure => {
                {
                    #[cfg(feature = "azure")]
                    {
                        let store =
                            options.build_azure(self.path.clone(), clear_cached_credentials)?;
                        Ok::<_, PolarsError>(Arc::new(store) as Arc<dyn ObjectStore>)
                    }
                }
                #[cfg(not(feature = "azure"))]
                return err_missing_feature("azure", &self.cloud_type);
            },
            CloudType::File => {
                let local = LocalFileSystem::new();
                Ok::<_, PolarsError>(Arc::new(local) as Arc<dyn ObjectStore>)
            },
            CloudType::Http => {
                {
                    #[cfg(feature = "http")]
                    {
                        let store = options.build_http(self.path.clone())?;
                        PolarsResult::Ok(Arc::new(store) as Arc<dyn ObjectStore>)
                    }
                }
                #[cfg(not(feature = "http"))]
                return err_missing_feature("http", &cloud_location.scheme);
            },
            CloudType::Hf => panic!("impl error: unresolved hf:// path"),
            CloudType::Ext(scheme) => {
                let prefix = &self.path.as_str()[..self.path.authority_end_position()];

                let store = EXT_OBJECT_STORE_BUILDER_REGISTRY
                    .read()
                    .await
                    .get(scheme.as_str())
                    .ok_or_else(|| {
                        polars_err!(
                            ComputeError:
                            "no object_store_builder registered for prefix: {}; \
                             call register_object_store_builder() before executing queries \
                             against the scheme: {}",
                            prefix, scheme
                        )
                    })?
                    .build(&self.path, self.options.as_ref())?;

                return Ok(store);
            },
        }?;

        Ok(store)
    }

    /// Note: Use `build_impl` for a non-caching version.
    pub(super) async fn build(self) -> PolarsResult<PolarsObjectStore> {
        let opt_cache_key = match &self.cloud_type {
            CloudType::Aws | CloudType::Gcp | CloudType::Azure => {
                Some(path_and_creds_to_key(&self.path, self.options.as_ref())?)
            },
            CloudType::File | CloudType::Http | CloudType::Hf => None,
            CloudType::Ext(scheme) => {
                let registry = EXT_OBJECT_STORE_BUILDER_REGISTRY.read().await;
                let builder = registry.get(scheme.as_str()).ok_or_else(|| {
                    polars_err!(
                        ComputeError:
                        "no object_store_builder registered for scheme '{}'; \
                         call register_object_store_builder() before executing queries \
                         against this scheme",
                        scheme
                    )
                })?;

                let key = match builder.stable_cache_key(&self.path, self.options.as_ref()) {
                    Some(key) => key,
                    None => path_and_creds_to_key(&self.path, self.options.as_ref())?,
                };

                Some(key)
            },
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

        let store = self.build_impl(false).await?;
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
    path: PlRefPath,
    #[cfg_attr(
        not(any(feature = "aws", feature = "gcp", feature = "azure")),
        allow(unused_variables)
    )]
    options: Option<&CloudOptions>,
    glob: bool,
) -> PolarsResult<(CloudLocation, PolarsObjectStore)> {
    let path = path.to_absolute_path()?.into_owned();

    let cloud_type = path
        .scheme()
        .map_or(CloudType::File, CloudType::from_cloud_scheme);
    let cloud_location = CloudLocation::new(path.clone(), glob)?;

    let store = PolarsObjectStoreBuilder {
        path,
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

#[cfg(all(test, feature = "cloud"))]
mod ext_store_tests {
    use std::sync::Arc;

    use object_store::ObjectStore;
    use object_store::memory::InMemory;
    use polars_utils::pl_path::PlRefPath;
    use polars_utils::relaxed_cell::RelaxedCell;

    use super::*;

    struct TestBuilder {
        store: Arc<dyn ObjectStore + Send + Sync>,
        build_count: RelaxedCell<usize>,
    }

    impl TestBuilder {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                store: Arc::new(InMemory::new()),
                build_count: RelaxedCell::new_usize(0),
            })
        }

        fn build_count(&self) -> usize {
            self.build_count.load()
        }

        fn inc_build_count(&self) {
            self.build_count.fetch_add(1);
        }
    }

    impl ExtObjectStoreBuilder for TestBuilder {
        fn build(
            &self,
            _url: &PlRefPath,
            _options: Option<&CloudOptions>,
        ) -> PolarsResult<Arc<dyn ObjectStore + Send + Sync>> {
            self.inc_build_count();
            Ok(self.store.clone())
        }
    }

    #[tokio::test]
    async fn test_register_and_resolve() {
        let builder = TestBuilder::new();
        register_object_store_builder("test-scheme", builder.clone())
            .await
            .unwrap();

        let path = PlRefPath::new("test-scheme://host:1234/data/file.parquet");
        let result = build_object_store(path, None, false).await;
        assert!(result.is_ok());
        assert_eq!(builder.build_count(), 1);

        deregister_object_store_builder("test-scheme").await;
    }

    #[tokio::test]
    async fn test_cache_hit_after_first_build() {
        let builder = TestBuilder::new();
        register_object_store_builder("test-scheme2", builder.clone())
            .await
            .unwrap();

        let path = PlRefPath::new("test-scheme2://host:1234/data/file.parquet");

        // First call — cache miss, build_impl called
        build_object_store(path.clone(), None, false).await.unwrap();
        assert_eq!(builder.build_count(), 1);

        // Second call — cache hit, build_impl not called
        build_object_store(path.clone(), None, false).await.unwrap();
        assert_eq!(builder.build_count(), 1);

        deregister_object_store_builder("test-scheme2").await;
    }

    #[tokio::test]
    async fn test_unregistered_scheme_errors() {
        let path = PlRefPath::new("unknown-scheme://host:1234/data/file.parquet");
        let result = build_object_store(path, None, false).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("no object_store_builder registered")
        );
    }

    #[tokio::test]
    async fn test_native_scheme_rejected() {
        let builder = TestBuilder::new();
        let result = register_object_store_builder("s3", builder).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("handled natively"));
    }

    #[tokio::test]
    async fn test_stable_cache_key_override() {
        #[derive(Clone)]
        struct AuthorityOnlyBuilder {
            store: Arc<dyn ObjectStore + Send + Sync>,
            build_count: Arc<RelaxedCell<usize>>,
        }

        impl AuthorityOnlyBuilder {
            fn new() -> Self {
                Self {
                    store: Arc::new(InMemory::new()),
                    build_count: Arc::new(RelaxedCell::new_usize(0)),
                }
            }

            fn build_count(&self) -> usize {
                self.build_count.load()
            }

            fn inc_build_count(&self) -> usize {
                self.build_count.fetch_add(1)
            }
        }

        impl ExtObjectStoreBuilder for AuthorityOnlyBuilder {
            fn build(
                &self,
                _url: &PlRefPath,
                _options: Option<&CloudOptions>,
            ) -> PolarsResult<Arc<dyn ObjectStore + Send + Sync>> {
                self.inc_build_count();
                Ok(self.store.clone())
            }

            fn stable_cache_key(
                &self,
                url: &PlRefPath,
                _options: Option<&CloudOptions>,
            ) -> Option<Vec<u8>> {
                let authority = &url.as_str()[..url.authority_end_position()];
                Some(authority.as_bytes().to_vec())
            }
        }

        let builder = AuthorityOnlyBuilder::new();
        register_object_store_builder("test-scheme3", Arc::new(builder.clone()))
            .await
            .unwrap();

        use crate::cloud::{CloudConfig, CloudOptions};

        let options_a = CloudOptions {
            config: Some(CloudConfig::Ext {
                options: vec![("user".to_string(), "alice".to_string())],
            }),
            ..CloudOptions::default()
        };

        let options_b = CloudOptions {
            config: Some(CloudConfig::Ext {
                options: vec![("user".to_string(), "bob".to_string())],
            }),
            ..CloudOptions::default()
        };

        let path = PlRefPath::new("test-scheme3://host:1234/data/file.parquet");

        build_object_store(path.clone(), Some(&options_a), false)
            .await
            .unwrap();
        build_object_store(path.clone(), Some(&options_b), false)
            .await
            .unwrap();

        assert_eq!(builder.build_count(), 1);

        deregister_object_store_builder("test-scheme3").await;
    }

    #[tokio::test]
    async fn test_storage_options_passed_to_builder() {
        use crate::cloud::{CloudConfig, CloudOptions};

        struct CapturingBuilder {
            received_options: Arc<std::sync::Mutex<Option<Vec<(String, String)>>>>,
            store: Arc<dyn ObjectStore + Send + Sync>,
        }

        impl ExtObjectStoreBuilder for CapturingBuilder {
            fn build(
                &self,
                _url: &PlRefPath,
                options: Option<&CloudOptions>,
            ) -> PolarsResult<Arc<dyn ObjectStore + Send + Sync>> {
                let captured = match options {
                    Some(CloudOptions {
                        config: Some(CloudConfig::Ext { options }),
                        ..
                    }) => Some(options.clone()),
                    _ => None,
                };
                *self.received_options.lock().unwrap() = captured;
                Ok(self.store.clone())
            }
        }

        let received = Arc::new(std::sync::Mutex::new(None));

        let builder = Arc::new(CapturingBuilder {
            received_options: received.clone(),
            store: Arc::new(InMemory::new()),
        });

        register_object_store_builder("test-scheme4", builder)
            .await
            .unwrap();

        let options = CloudOptions {
            config: Some(CloudConfig::Ext {
                options: vec![
                    ("user".to_string(), "hadoop".to_string()),
                    ("token".to_string(), "abc123".to_string()),
                ],
            }),
            ..CloudOptions::default()
        };

        let path = PlRefPath::new("test-scheme4://host:1234/data/file.parquet");
        build_object_store(path, Some(&options), false)
            .await
            .unwrap();

        let captured = received.lock().unwrap().clone().unwrap();
        assert_eq!(captured.len(), 2);
        assert!(captured.iter().any(|(k, v)| k == "user" && v == "hadoop"));
        assert!(captured.iter().any(|(k, v)| k == "token" && v == "abc123"));

        deregister_object_store_builder("test-scheme4").await;
    }
}
