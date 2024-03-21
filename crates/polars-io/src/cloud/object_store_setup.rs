use std::sync::Arc;

use object_store::local::LocalFileSystem;
use object_store::ObjectStore;
use polars_error::{polars_bail, to_compute_err, PolarsError, PolarsResult};

use super::{parse_url, CloudLocation, CloudOptions, CloudType};

type BuildResult = PolarsResult<(CloudLocation, Arc<dyn ObjectStore>)>;

#[allow(dead_code)]
fn err_missing_feature(feature: &str, scheme: &str) -> BuildResult {
    polars_bail!(
        ComputeError:
        "feature '{}' must be enabled in order to use '{}' cloud urls", feature, scheme,
    );
}

/// Build an [`ObjectStore`] based on the URL and passed in url. Return the cloud location and an implementation of the object store.
pub async fn build_object_store(
    url: &str,
    #[cfg_attr(
        not(any(feature = "aws", feature = "gcp", feature = "azure")),
        allow(unused_variables)
    )]
    options: Option<&CloudOptions>,
) -> BuildResult {
    let parsed = parse_url(url).map_err(to_compute_err)?;
    let cloud_location = CloudLocation::from_url(&parsed)?;

    #[cfg(any(feature = "aws", feature = "gcp", feature = "azure"))]
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
                    let store = object_store::http::HttpBuilder::new()
                        .with_url(url)
                        .with_client_options(super::get_client_options())
                        .build()?;
                    Ok::<_, PolarsError>(Arc::new(store) as Arc<dyn ObjectStore>)
                }
            }
            #[cfg(not(feature = "http"))]
            return err_missing_feature("http", &cloud_location.scheme);
        },
    }?;
    Ok((cloud_location, store))
}
