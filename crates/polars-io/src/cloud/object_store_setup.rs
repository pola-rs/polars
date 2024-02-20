use once_cell::sync::Lazy;
pub use options::*;
use polars_error::to_compute_err;
use polars_utils::aliases::PlHashMap;
use tokio::sync::RwLock;
use url::Url;

use super::*;

/// Object stores must be cached. Every object-store will do DNS lookups and
/// get rate limited when querying the DNS (can take up to 5s).
/// Other reasons are connection pools that must be shared between as much as possible.
#[allow(clippy::type_complexity)]
static OBJECT_STORE_CACHE: Lazy<RwLock<PlHashMap<String, Arc<dyn ObjectStore>>>> =
    Lazy::new(Default::default);

type BuildResult = PolarsResult<(CloudLocation, Arc<dyn ObjectStore>)>;

#[allow(dead_code)]
fn err_missing_feature(feature: &str, scheme: &str) -> BuildResult {
    polars_bail!(
        ComputeError:
        "feature '{}' must be enabled in order to use '{}' cloud urls", feature, scheme,
    );
}

/// Get the key of a url for object store registration.
/// The credential info will be removed
fn url_to_key(url: &Url) -> String {
    format!(
        "{}://{}",
        url.scheme(),
        &url[url::Position::BeforeHost..url::Position::AfterPort],
    )
}

/// Build an [`ObjectStore`] based on the URL and passed in url. Return the cloud location and an implementation of the object store.
pub async fn build_object_store(url: &str, options: Option<&CloudOptions>) -> BuildResult {
    let parsed = parse_url(url).map_err(to_compute_err)?;
    let cloud_location = CloudLocation::from_url(&parsed)?;

    let key = url_to_key(&parsed);

    {
        let cache = OBJECT_STORE_CACHE.read().await;
        if let Some(store) = cache.get(&key) {
            return Ok((cloud_location, store.clone()));
        }
    }

    let options = options
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(Default::default()));

    let cloud_type = CloudType::from_url(&parsed)?;
    let store = match cloud_type {
        CloudType::Aws => {
            #[cfg(feature = "aws")]
            {
                let store = options.build_aws(url).await?;
                Ok::<_, PolarsError>(Arc::new(store) as Arc<dyn ObjectStore>)
            }
            #[cfg(not(feature = "aws"))]
            return err_missing_feature("aws", &cloud_location.scheme);
        },
        CloudType::Gcp => {
            #[cfg(feature = "gcp")]
            {
                let store = options.build_gcp(url)?;
                Ok::<_, PolarsError>(Arc::new(store) as Arc<dyn ObjectStore>)
            }
            #[cfg(not(feature = "gcp"))]
            return err_missing_feature("gcp", &cloud_location.scheme);
        },
        CloudType::Azure => {
            {
                #[cfg(feature = "azure")]
                {
                    let store = options.build_azure(url)?;
                    Ok::<_, PolarsError>(Arc::new(store) as Arc<dyn ObjectStore>)
                }
            }
            #[cfg(not(feature = "azure"))]
            return err_missing_feature("azure", &cloud_location.scheme);
        },
        CloudType::File => {
            let local = LocalFileSystem::new();
            Ok::<_, PolarsError>(Arc::new(local) as Arc<dyn ObjectStore>)
        },
        CloudType::Http => {
            {
                #[cfg(feature = "http")]
                {
                    let store = object_store::http::HttpBuilder::new()
                        .with_url(url)
                        .with_client_options(get_client_options())
                        .build()?;
                    Ok::<_, PolarsError>(Arc::new(store) as Arc<dyn ObjectStore>)
                }
            }
            #[cfg(not(feature = "http"))]
            return err_missing_feature("http", &cloud_location.scheme);
        },
    }?;
    let mut cache = OBJECT_STORE_CACHE.write().await;
    // Clear the cache if we surpass a certain amount of buckets. Don't expect that to happen.
    if cache.len() > 512 {
        cache.clear()
    }
    cache.insert(key, store.clone());
    Ok((cloud_location, store))
}
