//! Interface with cloud storage through the object_store crate.

use std::str::FromStr;

use object_store::local::LocalFileSystem;
use object_store::ObjectStore;
use polars_core::cloud::{CloudOptions, CloudType};
use polars_core::prelude::{polars_bail, PolarsError, PolarsResult};

mod adaptors;
mod glob;
pub use adaptors::*;
pub use glob::*;

type BuildResult = PolarsResult<(CloudLocation, Box<dyn ObjectStore>)>;

#[allow(dead_code)]
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
/// Build an ObjectStore based on the URL and passed in url. Return the cloud location and an implementation of the object store.
pub fn build(url: &str, _options: Option<&CloudOptions>) -> BuildResult {
    let cloud_location = CloudLocation::new(url)?;
    let store = match CloudType::from_str(url)? {
        CloudType::File => {
            let local = LocalFileSystem::new();
            Ok::<_, PolarsError>(Box::new(local) as Box<dyn ObjectStore>)
        }
        CloudType::Aws => {
            #[cfg(feature = "aws")]
            match _options {
                Some(options) => {
                    let store = options.build_aws(&cloud_location.bucket)?;
                    Ok::<_, PolarsError>(Box::new(store) as Box<dyn ObjectStore>)
                }
                _ => return err_missing_configuration("aws", &cloud_location.scheme),
            }
            #[cfg(not(feature = "aws"))]
            return err_missing_feature("aws", &cloud_location.scheme);
        }
        CloudType::Gcp => {
            #[cfg(feature = "gcp")]
            match _options {
                Some(options) => {
                    let store = options.build_gcp(&cloud_location.bucket)?;
                    Ok::<_, PolarsError>(Box::new(store) as Box<dyn ObjectStore>)
                }
                _ => return err_missing_configuration("gcp", &cloud_location.scheme),
            }
            #[cfg(not(feature = "gcp"))]
            return err_missing_feature("gcp", &cloud_location.scheme);
        }
        CloudType::Azure => {
            {
                #[cfg(feature = "azure")]
                match _options {
                    Some(options) => {
                        let store = options.build_azure(&cloud_location.bucket)?;
                        Ok::<_, PolarsError>(Box::new(store) as Box<dyn ObjectStore>)
                    }
                    _ => return err_missing_configuration("azure", &cloud_location.scheme),
                }
            }
            #[cfg(not(feature = "azure"))]
            return err_missing_feature("azure", &cloud_location.scheme);
        }
    }?;
    Ok((cloud_location, store))
}
