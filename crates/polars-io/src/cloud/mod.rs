//! Interface with cloud storage through the object_store crate.

#[cfg(feature = "cloud")]
use std::borrow::Cow;
#[cfg(feature = "cloud")]
use std::str::FromStr;

#[cfg(feature = "cloud")]
use object_store::local::LocalFileSystem;
#[cfg(feature = "cloud")]
use object_store::ObjectStore;
#[cfg(feature = "cloud")]
use polars_core::prelude::{polars_bail, PolarsError, PolarsResult};

#[cfg(feature = "cloud")]
mod adaptors;
#[cfg(feature = "cloud")]
mod glob;
pub mod options;
#[cfg(feature = "cloud")]
pub use adaptors::*;
#[cfg(feature = "cloud")]
pub use glob::*;
pub use options::*;

#[cfg(feature = "cloud")]
type BuildResult = PolarsResult<(CloudLocation, Box<dyn ObjectStore>)>;

#[allow(dead_code)]
#[cfg(feature = "cloud")]
fn err_missing_feature(feature: &str, scheme: &str) -> BuildResult {
    polars_bail!(
        ComputeError:
        "feature '{}' must be enabled in order to use '{}' cloud urls", feature, scheme,
    );
}
#[cfg(any(feature = "azure", feature = "aws", feature = "gcp"))]
fn err_missing_configuration(feature: &str, scheme: &str) -> BuildResult {
    polars_bail!(
        ComputeError:
        "configuration '{}' must be provided in order to use '{}' cloud urls", feature, scheme,
    );
}

/// Build an [`ObjectStore`] based on the URL and passed in url. Return the cloud location and an implementation of the object store.
#[cfg(feature = "cloud")]
pub fn build_object_store(url: &str, _options: Option<&CloudOptions>) -> BuildResult {
    let cloud_location = CloudLocation::new(url)?;
    let store = match CloudType::from_str(url)? {
        CloudType::File => {
            let local = LocalFileSystem::new();
            Ok::<_, PolarsError>(Box::new(local) as Box<dyn ObjectStore>)
        },
        CloudType::Aws => {
            #[cfg(feature = "aws")]
            {
                let options = _options
                    .map(Cow::Borrowed)
                    .unwrap_or_else(|| Cow::Owned(Default::default()));
                let store = options.build_aws(&cloud_location.bucket)?;
                Ok::<_, PolarsError>(Box::new(store) as Box<dyn ObjectStore>)
            }
            #[cfg(not(feature = "aws"))]
            return err_missing_feature("aws", &cloud_location.scheme);
        },
        CloudType::Gcp => {
            #[cfg(feature = "gcp")]
            match _options {
                Some(options) => {
                    let store = options.build_gcp(&cloud_location.bucket)?;
                    Ok::<_, PolarsError>(Box::new(store) as Box<dyn ObjectStore>)
                },
                _ => return err_missing_configuration("gcp", &cloud_location.scheme),
            }
            #[cfg(not(feature = "gcp"))]
            return err_missing_feature("gcp", &cloud_location.scheme);
        },
        CloudType::Azure => {
            {
                #[cfg(feature = "azure")]
                match _options {
                    Some(options) => {
                        let store = options.build_azure(&cloud_location.bucket)?;
                        Ok::<_, PolarsError>(Box::new(store) as Box<dyn ObjectStore>)
                    },
                    _ => return err_missing_configuration("azure", &cloud_location.scheme),
                }
            }
            #[cfg(not(feature = "azure"))]
            return err_missing_feature("azure", &cloud_location.scheme);
        },
    }?;
    Ok((cloud_location, store))
}
