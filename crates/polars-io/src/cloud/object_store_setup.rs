use std::sync::Arc;

use object_store::local::LocalFileSystem;
use object_store::ObjectStore;
use once_cell::sync::Lazy;
use polars_error::{polars_bail, to_compute_err, PolarsError, PolarsResult};
use polars_utils::aliases::PlHashMap;
use tokio::sync::RwLock;
use url::Url;

use super::{parse_url, CloudLocation, CloudOptions, CloudType};

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
fn url_and_creds_to_key(url: &Url, options: Option<&CloudOptions>) -> String {
    // We include credentials as they can expire, so users will send new credentials for the same url.
    let creds = serde_json::to_string(&options).unwrap_or_else(|_| "".into());
    format!(
        "{}://{}<\\creds\\>{}",
        url.scheme(),
        &url[url::Position::BeforeHost..url::Position::AfterPort],
        creds
    )
}

/// Construct an object_store `Path` from a string without any encoding/decoding.
pub fn object_path_from_str(path: &str) -> PolarsResult<object_store::path::Path> {
    object_store::path::Path::parse(path).map_err(to_compute_err)
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
) -> BuildResult {
    let parsed = parse_url(url).map_err(to_compute_err)?;
    let cloud_location = CloudLocation::from_url(&parsed, glob)?;

    let key = url_and_creds_to_key(&parsed, options);
    let mut allow_cache = true;

    {
        let cache = OBJECT_STORE_CACHE.read().await;
        if let Some(store) = cache.get(&key) {
            return Ok((cloud_location, store.clone()));
        }
    }

    let options = options.map(std::borrow::Cow::Borrowed).unwrap_or_default();

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
            allow_cache = false;
            let local = LocalFileSystem::new();
            Ok::<_, PolarsError>(Arc::new(local) as Arc<dyn ObjectStore>)
        },
        CloudType::Http => {
            {
                allow_cache = false;
                #[cfg(feature = "http")]
                {
                    let store = options.build_http(url)?;
                    PolarsResult::Ok(Arc::new(store) as Arc<dyn ObjectStore>)
                }
            }
            #[cfg(not(feature = "http"))]
            return err_missing_feature("http", &cloud_location.scheme);
        },
        CloudType::Hf => panic!("impl error: unresolved hf:// path"),
    }?;
    if allow_cache {
        let mut cache = OBJECT_STORE_CACHE.write().await;
        // Clear the cache if we surpass a certain amount of buckets.
        if cache.len() > 8 {
            cache.clear()
        }
        cache.insert(key, store.clone());
    }
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
