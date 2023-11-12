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
#[cfg(feature = "cloud")]
use object_store::ObjectStore;
#[cfg(any(feature = "aws", feature = "gcp", feature = "azure"))]
use object_store::{BackoffConfig, RetryConfig};
#[cfg(feature = "aws")]
use once_cell::sync::Lazy;
use polars_core::error::{PolarsError, PolarsResult};
use polars_error::*;
#[cfg(feature = "aws")]
use polars_utils::cache::FastFixedCache;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "aws")]
use smartstring::alias::String as SmartString;
#[cfg(feature = "cloud")]
use url::Url;

#[cfg(feature = "aws")]
static BUCKET_REGION: Lazy<std::sync::Mutex<FastFixedCache<SmartString, SmartString>>> =
    Lazy::new(|| std::sync::Mutex::new(FastFixedCache::new(32)));

/// The type of the config keys must satisfy the following requirements:
/// 1. must be easily collected into a HashMap, the type required by the object_crate API.
/// 2. be Serializable, required when the serde-lazy feature is defined.
/// 3. not actually use HashMap since that type is disallowed in Polars for performance reasons.
///
/// Currently this type is a vector of pairs config key - config value.
#[allow(dead_code)]
type Configs<T> = Vec<(T, String)>;

#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Options to connect to various cloud providers.
pub struct CloudOptions {
    #[cfg(feature = "aws")]
    aws: Option<Configs<AmazonS3ConfigKey>>,
    #[cfg(feature = "azure")]
    azure: Option<Configs<AzureConfigKey>>,
    #[cfg(feature = "gcp")]
    gcp: Option<Configs<GoogleConfigKey>>,
    pub max_retries: usize,
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

#[derive(PartialEq)]
pub enum CloudType {
    Aws,
    Azure,
    File,
    Gcp,
}

impl CloudType {
    #[cfg(feature = "cloud")]
    pub(crate) fn from_url(parsed: &Url) -> PolarsResult<Self> {
        Ok(match parsed.scheme() {
            "s3" | "s3a" => Self::Aws,
            "az" | "azure" | "adl" | "abfs" | "abfss" => Self::Azure,
            "gs" | "gcp" | "gcs" => Self::Gcp,
            "file" => Self::File,
            _ => polars_bail!(ComputeError: "unknown url scheme"),
        })
    }
}

#[cfg(feature = "cloud")]
pub(crate) fn parse_url(url: &str) -> std::result::Result<Url, url::ParseError> {
    match Url::parse(url) {
        Err(err) => match err {
            url::ParseError::RelativeUrlWithoutBase => {
                let parsed = Url::parse(&format!(
                    "file://{}/",
                    std::env::current_dir().unwrap().to_string_lossy()
                ))
                .unwrap();
                parsed.join(url)
            },
            err => Err(err),
        },
        parsed => parsed,
    }
}

impl FromStr for CloudType {
    type Err = PolarsError;

    #[cfg(feature = "cloud")]
    fn from_str(url: &str) -> Result<Self, Self::Err> {
        let parsed = parse_url(url).map_err(to_compute_err)?;
        Self::from_url(&parsed)
    }

    #[cfg(not(feature = "cloud"))]
    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        polars_bail!(ComputeError: "at least one of the cloud features must be enabled");
    }
}
#[cfg(any(feature = "aws", feature = "gcp", feature = "azure"))]
fn get_retry_config(max_retries: usize) -> RetryConfig {
    RetryConfig {
        backoff: BackoffConfig::default(),
        max_retries,
        retry_timeout: std::time::Duration::from_secs(10),
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

    /// Build the [`ObjectStore`] implementation for AWS.
    #[cfg(feature = "aws")]
    pub async fn build_aws(&self, url: &str) -> PolarsResult<impl ObjectStore> {
        let options = self.aws.as_ref();
        let mut builder = AmazonS3Builder::from_env().with_url(url);
        if let Some(options) = options {
            for (key, value) in options.iter() {
                builder = builder.with_config(*key, value);
            }
        }

        if builder
            .get_config_value(&AmazonS3ConfigKey::DefaultRegion)
            .is_none()
            && builder
                .get_config_value(&AmazonS3ConfigKey::Region)
                .is_none()
        {
            let bucket = crate::cloud::CloudLocation::new(url)?.bucket;
            let region = {
                let bucket_region = BUCKET_REGION.lock().unwrap();
                bucket_region.get(bucket.as_str()).cloned()
            };

            match region {
                Some(region) => {
                    builder = builder.with_config(AmazonS3ConfigKey::Region, region.as_str())
                },
                None => {
                    polars_warn!("'(default_)region' not set; polars will try to get it from bucket\n\nSet the region manually to silence this warning.");
                    let result = reqwest::Client::builder()
                        .build()
                        .unwrap()
                        .head(format!("https://{bucket}.s3.amazonaws.com"))
                        .send()
                        .await
                        .map_err(to_compute_err)?;
                    if let Some(region) = result.headers().get("x-amz-bucket-region") {
                        let region =
                            std::str::from_utf8(region.as_bytes()).map_err(to_compute_err)?;
                        let mut bucket_region = BUCKET_REGION.lock().unwrap();
                        bucket_region.insert(bucket.into(), region.into());
                        builder = builder.with_config(AmazonS3ConfigKey::Region, region)
                    }
                },
            };
        };

        builder
            .with_retry(get_retry_config(self.max_retries))
            .build()
            .map_err(to_compute_err)
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

    /// Build the [`ObjectStore`] implementation for Azure.
    #[cfg(feature = "azure")]
    pub fn build_azure(&self, url: &str) -> PolarsResult<impl ObjectStore> {
        let options = self.azure.as_ref();
        let mut builder = MicrosoftAzureBuilder::from_env();
        if let Some(options) = options {
            for (key, value) in options.iter() {
                builder = builder.with_config(*key, value);
            }
        }

        builder
            .with_url(url)
            .with_retry(get_retry_config(self.max_retries))
            .build()
            .map_err(to_compute_err)
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

    /// Build the [`ObjectStore`] implementation for GCP.
    #[cfg(feature = "gcp")]
    pub fn build_gcp(&self, url: &str) -> PolarsResult<impl ObjectStore> {
        let options = self.gcp.as_ref();
        let mut builder = GoogleCloudStorageBuilder::from_env();
        if let Some(options) = options {
            for (key, value) in options.iter() {
                builder = builder.with_config(*key, value);
            }
        }

        builder
            .with_url(url)
            .with_retry(get_retry_config(self.max_retries))
            .build()
            .map_err(to_compute_err)
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
            },
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
            },
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
            },
        }
    }
}
