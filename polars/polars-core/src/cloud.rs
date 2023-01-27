use std::str::FromStr;

#[cfg(feature = "aws")]
use object_store::aws::AmazonS3Builder;
#[cfg(feature = "aws")]
pub use object_store::aws::AmazonS3ConfigKey;
#[cfg(feature = "azure")]
pub use object_store::azure::AzureConfigKey;
#[cfg(feature = "azure")]
use object_store::azure::MicrosoftAzureBuilder;
#[cfg(feature = "gcp")]
use object_store::gcp::GoogleCloudStorageBuilder;
#[cfg(feature = "gcp")]
pub use object_store::gcp::GoogleConfigKey;
#[cfg(feature = "async")]
use object_store::ObjectStore;
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "async")]
use url::Url;

use crate::error::{PolarsError, PolarsResult};

/// The type of the config keys must satisfy the following requirements:
/// 1. must be easily collected into a HashMap, the type required by the object_crate API.
/// 2. be Serializable, required when the serde-lazy feature is defined.
/// 3. not actually use HashMap since that type is disallowed in Polars for performance reasons.
///
/// Currently this type is a vector of pairs config key - config value.
#[allow(dead_code)]
type Configs<T> = Vec<(T, String)>;

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
/// Options to conect to various cloud providers.
pub struct CloudOptions {
    #[cfg(feature = "aws")]
    aws: Option<Configs<AmazonS3ConfigKey>>,
    #[cfg(feature = "azure")]
    azure: Option<Configs<AzureConfigKey>>,
    #[cfg(feature = "gcp")]
    gcp: Option<Configs<GoogleConfigKey>>,
}

#[allow(dead_code)]
/// Parse an untype configuration hashmap to a typed configuration for the given configuration key type.
fn parsed_untyped_config<T, I: IntoIterator<Item = (impl AsRef<str>, impl Into<String>)>>(
    config: I,
) -> PolarsResult<Configs<T>>
where
    T: FromStr + std::cmp::Eq + std::hash::Hash,
{
    config
        .into_iter()
        .map(|(key, val)| {
            T::from_str(key.as_ref())
                .map_err(|_e| {
                    PolarsError::ComputeError(
                        format!("Unknown configuration key {}.", key.as_ref()).into(),
                    )
                })
                .map(|typed_key| (typed_key, val.into()))
        })
        .collect::<PolarsResult<Configs<T>>>()
}

pub enum CloudType {
    Aws,
    Azure,
    File,
    Gcp,
}

impl FromStr for CloudType {
    type Err = PolarsError;

    #[cfg(feature = "async")]
    fn from_str(url: &str) -> Result<Self, Self::Err> {
        let parsed = Url::parse(url).map_err(anyhow::Error::from)?;
        match parsed.scheme() {
            "s3" => Ok(Self::Aws),
            "az" | "adl" | "abfs" => Ok(Self::Azure),
            "gs" | "gcp" => Ok(Self::Gcp),
            "file" => Ok(Self::File),
            &_ => Err(PolarsError::ComputeError("Unknown url scheme.".into())),
        }
    }

    #[cfg(not(feature = "async"))]
    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        Err(PolarsError::ComputeError(
            "At least one of the cloud features must be enabled.".into(),
        ))
    }
}

impl CloudOptions {
    /// Set the configuration for AWS connections. This is the preferred API from rust.
    #[cfg(feature = "aws")]
    pub fn with_aws<I: IntoIterator<Item = (AmazonS3ConfigKey, impl Into<String>)>>(
        mut self,
        configs: I,
    ) -> Self {
        self.aws = Some(
            configs
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect::<Configs<AmazonS3ConfigKey>>(),
        );
        self
    }

    /// Build the ObjectStore implementation for AWS.
    #[cfg(feature = "aws")]
    pub fn build_aws(&self, bucket_name: &str) -> PolarsResult<impl ObjectStore> {
        let options = self.aws.as_ref().map(Ok).unwrap_or_else(|| {
            Err(PolarsError::ComputeError(
                "`aws` configuration missing.".into(),
            ))
        })?;
        AmazonS3Builder::new()
            .try_with_options(options.clone().into_iter())
            .and_then(|b| b.with_bucket_name(bucket_name).build())
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))
    }

    /// Set the configuration for Azure connections. This is the preferred API from rust.
    #[cfg(feature = "azure")]
    pub fn with_azure<I: IntoIterator<Item = (AzureConfigKey, impl Into<String>)>>(
        mut self,
        configs: I,
    ) -> Self {
        self.azure = Some(
            configs
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect::<Configs<AzureConfigKey>>(),
        );
        self
    }

    /// Build the ObjectStore implementation for Azure.
    #[cfg(feature = "azure")]
    pub fn build_azure(&self, container_name: &str) -> PolarsResult<impl ObjectStore> {
        let options = self.azure.as_ref().map(Ok).unwrap_or_else(|| {
            Err(PolarsError::ComputeError(
                "`azure` configuration missing.".into(),
            ))
        })?;
        MicrosoftAzureBuilder::new()
            .try_with_options(options.clone().into_iter())
            .and_then(|b| b.with_container_name(container_name).build())
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))
    }

    /// Set the configuration for GCP connections. This is the preferred API from rust.
    #[cfg(feature = "gcp")]
    pub fn with_gcp<I: IntoIterator<Item = (GoogleConfigKey, impl Into<String>)>>(
        mut self,
        configs: I,
    ) -> Self {
        self.gcp = Some(
            configs
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect::<Configs<GoogleConfigKey>>(),
        );
        self
    }

    /// Build the ObjectStore implementation for GCP.
    #[cfg(feature = "gcp")]
    pub fn build_gcp(&self, bucket_name: &str) -> PolarsResult<impl ObjectStore> {
        let options = self.gcp.as_ref().map(Ok).unwrap_or_else(|| {
            Err(PolarsError::ComputeError(
                "`gcp` configuration missing.".into(),
            ))
        })?;
        GoogleCloudStorageBuilder::new()
            .try_with_options(options.clone().into_iter())
            .and_then(|b| b.with_bucket_name(bucket_name).build())
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))
    }

    /// Parse a configuration from a Hashmap. This is the interface from Python.
    #[allow(unused_variables)]
    pub fn from_untyped_config<I: IntoIterator<Item = (impl AsRef<str>, impl Into<String>)>>(
        url: &str,
        config: I,
    ) -> PolarsResult<Self> {
        match CloudType::from_str(url)? {
            CloudType::Aws => {
                #[cfg(feature = "aws")]
                {
                    parsed_untyped_config::<AmazonS3ConfigKey, _>(config)
                        .map(|aws| Self::default().with_aws(aws))
                }
                #[cfg(not(feature = "aws"))]
                {
                    Err(PolarsError::ComputeError(
                        "Feature aws is not enabled.".into(),
                    ))
                }
            }
            CloudType::Azure => {
                #[cfg(feature = "azure")]
                {
                    parsed_untyped_config::<AzureConfigKey, _>(config)
                        .map(|azure| Self::default().with_azure(azure))
                }
                #[cfg(not(feature = "azure"))]
                {
                    Err(PolarsError::ComputeError(
                        "Feature gcp is not enabled.".into(),
                    ))
                }
            }
            CloudType::File => Ok(Self::default()),
            CloudType::Gcp => {
                #[cfg(feature = "gcp")]
                {
                    parsed_untyped_config::<GoogleConfigKey, _>(config)
                        .map(|gcp| Self::default().with_gcp(gcp))
                }
                #[cfg(not(feature = "gcp"))]
                {
                    Err(PolarsError::ComputeError(
                        "Feature gcp is not enabled.".into(),
                    ))
                }
            }
        }
    }
}
