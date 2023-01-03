#![allow(clippy::disallowed_types)]
use std::collections::HashMap;
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

/// The object_store crate API requires a HashMap.
#[allow(dead_code)]
#[allow(clippy::disallowed_types)]
type Configs<T> = HashMap<T, String>;

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
fn parsed_untyped_config<T>(config: HashMap<String, String>) -> PolarsResult<Configs<T>>
where
    T: FromStr + std::cmp::Eq + std::hash::Hash,
{
    config
        .into_iter()
        .map(|(key, val)| {
            T::from_str(&key)
                .map_err(|_e| {
                    PolarsError::ComputeError(format!("Unknown configuration key {key}.").into())
                })
                .map(|typed_key| (typed_key, val))
        })
        .collect::<PolarsResult<HashMap<T, String>>>()
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
    pub fn with_aws(mut self, aws: Configs<AmazonS3ConfigKey>) -> Self {
        self.aws = Some(aws);
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
            .try_with_options(options)
            .and_then(|b| b.with_bucket_name(bucket_name).build())
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))
    }

    /// Set the configuration for Azure connections. This is the preferred API from rust.
    #[cfg(feature = "azure")]
    pub fn with_azure(mut self, azure: Configs<AzureConfigKey>) -> Self {
        self.azure = Some(azure);
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
            .try_with_options(options)
            .and_then(|b| b.with_container_name(container_name).build())
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))
    }

    /// Set the configuration for GCP connections. This is the preferred API from rust.
    #[cfg(feature = "gcp")]
    pub fn with_gcp(mut self, gcp: Configs<GoogleConfigKey>) -> Self {
        self.gcp = Some(gcp);
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
            .try_with_options(options)
            .and_then(|b| b.with_bucket_name(bucket_name).build())
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))
    }

    /// Parse a configuration from a Hashmap. This is the interface from Python.
    #[allow(unused_variables)]
    pub fn from_untyped_config(url: &str, config: HashMap<String, String>) -> PolarsResult<Self> {
        match CloudType::from_str(url)? {
            CloudType::Aws => {
                #[cfg(feature = "aws")]
                {
                    parsed_untyped_config::<AmazonS3ConfigKey>(config)
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
                    parsed_untyped_config::<AzureConfigKey>(config)
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
                    parsed_untyped_config::<GoogleConfigKey>(config)
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
