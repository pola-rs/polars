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
use polars_error::{polars_bail, polars_err};
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
/// Options to connect to various cloud providers.
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
    T: FromStr + Eq + std::hash::Hash,
{
    config
        .into_iter()
        .map(|(key, val)| {
            T::from_str(key.as_ref())
                .map_err(
                    |_| polars_err!(ComputeError: "unknown configuration key: {}", key.as_ref()),
                )
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
        let parsed = Url::parse(url).map_err(polars_error::to_compute_err)?;
        Ok(match parsed.scheme() {
            "s3" => Self::Aws,
            "az" | "adl" | "abfs" => Self::Azure,
            "gs" | "gcp" => Self::Gcp,
            "file" => Self::File,
            _ => polars_bail!(ComputeError: "unknown url scheme"),
        })
    }

    #[cfg(not(feature = "async"))]
    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        polars_bail!(ComputeError: "at least one of the cloud features must be enabled");
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
        let options = self
            .aws
            .as_ref()
            .ok_or_else(|| polars_err!(ComputeError: "`aws` configuration missing"))?;

        let mut builder = AmazonS3Builder::new();
        for (key, value) in options.iter() {
            builder = builder.with_config(*key, value);
        }
        builder
            .with_bucket_name(bucket_name)
            .build()
            .map_err(polars_error::to_compute_err)
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
        let options = self
            .azure
            .as_ref()
            .ok_or_else(|| polars_err!(ComputeError: "`azure` configuration missing"))?;

        let mut builder = MicrosoftAzureBuilder::new();
        for (key, value) in options.iter() {
            builder = builder.with_config(*key, value);
        }
        builder
            .with_container_name(container_name)
            .build()
            .map_err(polars_error::to_compute_err)
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
        let options = self
            .gcp
            .as_ref()
            .ok_or_else(|| polars_err!(ComputeError: "`gcp` configuration missing"))?;

        let mut builder = GoogleCloudStorageBuilder::new();
        for (key, value) in options.iter() {
            builder = builder.with_config(*key, value);
        }
        builder
            .with_bucket_name(bucket_name)
            .build()
            .map_err(polars_error::to_compute_err)
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
                    polars_bail!(ComputeError: "'aws' feature is not enabled");
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
                    polars_bail!(ComputeError: "'azure' feature is not enabled");
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
                    polars_bail!(ComputeError: "'gcp' feature is not enabled");
                }
            }
        }
    }
}
