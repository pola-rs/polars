use once_cell::sync::Lazy;
pub use options::*;
use tokio::sync::RwLock;

use super::*;

type CacheKey = (String, Option<CloudOptions>);

/// A very simple cache that only stores a single object-store.
/// This greatly reduces the query times as multiple object stores (when reading many small files)
/// get rate limited when querying the DNS (can take up to 5s).
#[allow(clippy::type_complexity)]
static OBJECT_STORE_CACHE: Lazy<RwLock<Option<(CacheKey, Arc<dyn ObjectStore>)>>> =
    Lazy::new(Default::default);

type BuildResult = PolarsResult<(CloudLocation, Arc<dyn ObjectStore>)>;

#[allow(dead_code)]
fn err_missing_feature(feature: &str, scheme: &str) -> BuildResult {
    polars_bail!(
        ComputeError:
        "feature '{}' must be enabled in order to use '{}' cloud urls", feature, scheme,
    );
}

/// Build an [`ObjectStore`] based on the URL and passed in url. Return the cloud location and an implementation of the object store.
pub async fn build_object_store(url: &str, options: Option<&CloudOptions>) -> BuildResult {
    let cloud_location = CloudLocation::new(url)?;

    let options = options.cloned();
    let key = (url.to_string(), options);

    {
        let cache = OBJECT_STORE_CACHE.read().await;
        if let Some((stored_key, store)) = cache.as_ref() {
            if stored_key == &key {
                return Ok((cloud_location, store.clone()));
            }
        }
    }

    let cloud_type = CloudType::from_str(url)?;
    let options = key
        .1
        .as_ref()
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(Default::default()));
    let store = match cloud_type {
        CloudType::File => {
            let local = LocalFileSystem::new();
            Ok::<_, PolarsError>(Arc::new(local) as Arc<dyn ObjectStore>)
        },
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
    }?;
    let mut cache = OBJECT_STORE_CACHE.write().await;
    *cache = Some((key, store.clone()));
    Ok((cloud_location, store))
}
