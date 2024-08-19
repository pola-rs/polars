#[cfg(feature = "aws")]
use std::io::Read;
#[cfg(feature = "aws")]
use std::path::Path;
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
#[cfg(any(feature = "aws", feature = "gcp", feature = "azure", feature = "http"))]
use object_store::ClientOptions;
#[cfg(any(feature = "aws", feature = "gcp", feature = "azure"))]
use object_store::{BackoffConfig, RetryConfig};
#[cfg(feature = "aws")]
use once_cell::sync::Lazy;
use polars_error::*;
#[cfg(feature = "aws")]
use polars_utils::cache::FastFixedCache;
#[cfg(feature = "aws")]
use regex::Regex;
#[cfg(feature = "http")]
use reqwest::header::HeaderMap;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "aws")]
use smartstring::alias::String as SmartString;
#[cfg(feature = "cloud")]
use url::Url;

#[cfg(feature = "file_cache")]
use crate::file_cache::get_env_file_cache_ttl;
#[cfg(feature = "aws")]
use crate::pl_async::with_concurrency_budget;

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

#[derive(Clone, Debug, PartialEq, Hash, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) enum CloudConfig {
    #[cfg(feature = "aws")]
    Aws(Configs<AmazonS3ConfigKey>),
    #[cfg(feature = "azure")]
    Azure(Configs<AzureConfigKey>),
    #[cfg(feature = "gcp")]
    Gcp(Configs<GoogleConfigKey>),
    #[cfg(feature = "http")]
    Http { headers: Vec<(String, String)> },
}

#[derive(Clone, Debug, PartialEq, Hash, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Options to connect to various cloud providers.
pub struct CloudOptions {
    pub max_retries: usize,
    #[cfg(feature = "file_cache")]
    pub file_cache_ttl: u64,
    pub(crate) config: Option<CloudConfig>,
}

impl Default for CloudOptions {
    fn default() -> Self {
        Self {
            max_retries: 2,
            #[cfg(feature = "file_cache")]
            file_cache_ttl: get_env_file_cache_ttl(),
            config: None,
        }
    }
}

#[cfg(feature = "http")]
pub(crate) fn try_build_http_header_map_from_items_slice<S: AsRef<str>>(
    headers: &[(S, S)],
) -> PolarsResult<HeaderMap> {
    use reqwest::header::{HeaderName, HeaderValue};

    let mut map = HeaderMap::with_capacity(headers.len());
    for (k, v) in headers {
        let (k, v) = (k.as_ref(), v.as_ref());
        map.insert(
            HeaderName::from_str(k).map_err(to_compute_err)?,
            HeaderValue::from_str(v).map_err(to_compute_err)?,
        );
    }

    Ok(map)
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
    Http,
    Hf,
}

impl CloudType {
    #[cfg(feature = "cloud")]
    pub(crate) fn from_url(parsed: &Url) -> PolarsResult<Self> {
        Ok(match parsed.scheme() {
            "s3" | "s3a" => Self::Aws,
            "az" | "azure" | "adl" | "abfs" | "abfss" => Self::Azure,
            "gs" | "gcp" | "gcs" => Self::Gcp,
            "file" => Self::File,
            "http" | "https" => Self::Http,
            "hf" => Self::Hf,
            _ => polars_bail!(ComputeError: "unknown url scheme"),
        })
    }
}

#[cfg(feature = "cloud")]
pub(crate) fn parse_url(input: &str) -> std::result::Result<url::Url, url::ParseError> {
    Ok(if input.contains("://") {
        if input.starts_with("http://") || input.starts_with("https://") {
            url::Url::parse(input)
        } else {
            url::Url::parse(&input.replace("%", "%25"))
        }?
    } else {
        let path = std::path::Path::new(input);
        let mut tmp;
        url::Url::from_file_path(if path.is_relative() {
            tmp = std::env::current_dir().unwrap();
            tmp.push(path);
            tmp.as_path()
        } else {
            path
        })
        .unwrap()
    })
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

#[cfg(any(feature = "aws", feature = "gcp", feature = "azure", feature = "http"))]
pub(super) fn get_client_options() -> ClientOptions {
    ClientOptions::default()
        // We set request timeout super high as the timeout isn't reset at ACK,
        // but starts from the moment we start downloading a body.
        // https://docs.rs/reqwest/latest/reqwest/struct.ClientBuilder.html#method.timeout
        .with_timeout_disabled()
        // Concurrency can increase connection latency, so set to None, similar to default.
        .with_connect_timeout_disabled()
        .with_allow_http(true)
}

#[cfg(feature = "aws")]
fn read_config(
    builder: &mut AmazonS3Builder,
    items: &[(&Path, &[(&str, AmazonS3ConfigKey)])],
) -> Option<()> {
    use crate::path_utils::resolve_homedir;

    for (path, keys) in items {
        if keys
            .iter()
            .all(|(_, key)| builder.get_config_value(key).is_some())
        {
            continue;
        }

        let mut config = std::fs::File::open(resolve_homedir(path)).ok()?;
        let mut buf = vec![];
        config.read_to_end(&mut buf).ok()?;
        let content = std::str::from_utf8(buf.as_ref()).ok()?;

        for (pattern, key) in keys.iter() {
            if builder.get_config_value(key).is_none() {
                let reg = Regex::new(pattern).unwrap();
                let cap = reg.captures(content)?;
                let m = cap.get(1)?;
                let parsed = m.as_str();
                *builder = std::mem::take(builder).with_config(*key, parsed);
            }
        }
    }
    Some(())
}

impl CloudOptions {
    /// Set the maximum number of retries.
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set the configuration for AWS connections. This is the preferred API from rust.
    #[cfg(feature = "aws")]
    pub fn with_aws<I: IntoIterator<Item = (AmazonS3ConfigKey, impl Into<String>)>>(
        mut self,
        configs: I,
    ) -> Self {
        self.config = Some(CloudConfig::Aws(
            configs.into_iter().map(|(k, v)| (k, v.into())).collect(),
        ));
        self
    }

    /// Build the [`object_store::ObjectStore`] implementation for AWS.
    #[cfg(feature = "aws")]
    pub async fn build_aws(&self, url: &str) -> PolarsResult<impl object_store::ObjectStore> {
        let mut builder = AmazonS3Builder::from_env().with_url(url);
        if let Some(options) = &self.config {
            let CloudConfig::Aws(options) = options else {
                panic!("impl error: cloud type mismatch")
            };
            for (key, value) in options.iter() {
                builder = builder.with_config(*key, value);
            }
        }

        read_config(
            &mut builder,
            &[(
                Path::new("~/.aws/config"),
                &[("region\\s*=\\s*(.*)\n", AmazonS3ConfigKey::Region)],
            )],
        );
        read_config(
            &mut builder,
            &[(
                Path::new("~/.aws/credentials"),
                &[
                    (
                        "aws_access_key_id\\s*=\\s*(.*)\n",
                        AmazonS3ConfigKey::AccessKeyId,
                    ),
                    (
                        "aws_secret_access_key\\s*=\\s*(.*)\n",
                        AmazonS3ConfigKey::SecretAccessKey,
                    ),
                ],
            )],
        );

        if builder
            .get_config_value(&AmazonS3ConfigKey::DefaultRegion)
            .is_none()
            && builder
                .get_config_value(&AmazonS3ConfigKey::Region)
                .is_none()
        {
            let bucket = crate::cloud::CloudLocation::new(url, false)?.bucket;
            let region = {
                let bucket_region = BUCKET_REGION.lock().unwrap();
                bucket_region.get(bucket.as_str()).cloned()
            };

            match region {
                Some(region) => {
                    builder = builder.with_config(AmazonS3ConfigKey::Region, region.as_str())
                },
                None => {
                    if builder
                        .get_config_value(&AmazonS3ConfigKey::Endpoint)
                        .is_some()
                    {
                        // Set a default value if the endpoint is not aws.
                        // See: #13042
                        builder = builder.with_config(AmazonS3ConfigKey::Region, "us-east-1");
                    } else {
                        polars_warn!("'(default_)region' not set; polars will try to get it from bucket\n\nSet the region manually to silence this warning.");
                        let result = with_concurrency_budget(1, || async {
                            reqwest::Client::builder()
                                .build()
                                .unwrap()
                                .head(format!("https://{bucket}.s3.amazonaws.com"))
                                .send()
                                .await
                                .map_err(to_compute_err)
                        })
                        .await?;
                        if let Some(region) = result.headers().get("x-amz-bucket-region") {
                            let region =
                                std::str::from_utf8(region.as_bytes()).map_err(to_compute_err)?;
                            let mut bucket_region = BUCKET_REGION.lock().unwrap();
                            bucket_region.insert(bucket.into(), region.into());
                            builder = builder.with_config(AmazonS3ConfigKey::Region, region)
                        }
                    }
                },
            };
        };

        builder
            .with_client_options(get_client_options())
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
        self.config = Some(CloudConfig::Azure(
            configs.into_iter().map(|(k, v)| (k, v.into())).collect(),
        ));
        self
    }

    /// Build the [`object_store::ObjectStore`] implementation for Azure.
    #[cfg(feature = "azure")]
    pub fn build_azure(&self, url: &str) -> PolarsResult<impl object_store::ObjectStore> {
        let mut builder = MicrosoftAzureBuilder::from_env();
        if let Some(options) = &self.config {
            let CloudConfig::Azure(options) = options else {
                panic!("impl error: cloud type mismatch")
            };
            for (key, value) in options.iter() {
                builder = builder.with_config(*key, value);
            }
        }

        builder
            .with_client_options(get_client_options())
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
        self.config = Some(CloudConfig::Gcp(
            configs.into_iter().map(|(k, v)| (k, v.into())).collect(),
        ));
        self
    }

    /// Build the [`object_store::ObjectStore`] implementation for GCP.
    #[cfg(feature = "gcp")]
    pub fn build_gcp(&self, url: &str) -> PolarsResult<impl object_store::ObjectStore> {
        let mut builder = GoogleCloudStorageBuilder::from_env();
        if let Some(options) = &self.config {
            let CloudConfig::Gcp(options) = options else {
                panic!("impl error: cloud type mismatch")
            };
            for (key, value) in options.iter() {
                builder = builder.with_config(*key, value);
            }
        }

        builder
            .with_client_options(get_client_options())
            .with_url(url)
            .with_retry(get_retry_config(self.max_retries))
            .build()
            .map_err(to_compute_err)
    }

    #[cfg(feature = "http")]
    pub fn build_http(&self, url: &str) -> PolarsResult<impl object_store::ObjectStore> {
        object_store::http::HttpBuilder::new()
            .with_url(url)
            .with_client_options({
                let mut opts = super::get_client_options();
                if let Some(CloudConfig::Http { headers }) = &self.config {
                    opts = opts.with_default_headers(try_build_http_header_map_from_items_slice(
                        headers.as_slice(),
                    )?);
                }
                opts
            })
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
            CloudType::Http => Ok(Self::default()),
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
            CloudType::Hf => {
                #[cfg(feature = "http")]
                {
                    use polars_core::config;

                    use crate::path_utils::resolve_homedir;

                    let mut this = Self::default();
                    let mut token = None;
                    let verbose = config::verbose();

                    for (i, (k, v)) in config.into_iter().enumerate() {
                        let (k, v) = (k.as_ref(), v.into());

                        if i == 0 && k == "token" {
                            if verbose {
                                eprintln!("HF token sourced from storage_options");
                            }
                            token = Some(v);
                        } else {
                            polars_bail!(ComputeError: "unknown configuration key for HF: {}", k)
                        }
                    }

                    token = token
                        .or_else(|| {
                            let v = std::env::var("HF_TOKEN").ok();
                            if v.is_some() && verbose {
                                eprintln!("HF token sourced from HF_TOKEN env var");
                            }
                            v
                        })
                        .or_else(|| {
                            let hf_home = std::env::var("HF_HOME");
                            let hf_home = hf_home.as_deref();
                            let hf_home = hf_home.unwrap_or("~/.cache/huggingface");
                            let hf_home = resolve_homedir(std::path::Path::new(&hf_home));
                            let cached_token_path = hf_home.join("token");

                            let v = std::string::String::from_utf8(
                                std::fs::read(&cached_token_path).ok()?,
                            )
                            .ok()
                            .filter(|x| !x.is_empty());

                            if v.is_some() && verbose {
                                eprintln!(
                                    "HF token sourced from {}",
                                    cached_token_path.to_str().unwrap()
                                );
                            }

                            v
                        });

                    if let Some(v) = token {
                        this.config = Some(CloudConfig::Http {
                            headers: vec![("Authorization".into(), format!("Bearer {}", v))],
                        })
                    }

                    Ok(this)
                }
                #[cfg(not(feature = "http"))]
                {
                    polars_bail!(ComputeError: "'http' feature is not enabled");
                }
            },
        }
    }
}

#[cfg(feature = "cloud")]
#[cfg(test)]
mod tests {
    use super::parse_url;

    #[test]
    fn test_parse_url() {
        assert_eq!(
            parse_url(r"http://Users/Jane Doe/data.csv")
                .unwrap()
                .as_str(),
            "http://users/Jane%20Doe/data.csv"
        );
        assert_eq!(
            parse_url(r"http://Users/Jane Doe/data.csv")
                .unwrap()
                .as_str(),
            "http://users/Jane%20Doe/data.csv"
        );
        #[cfg(target_os = "windows")]
        {
            assert_eq!(
                parse_url(r"file:///c:/Users/Jane Doe/data.csv")
                    .unwrap()
                    .as_str(),
                "file:///c:/Users/Jane%20Doe/data.csv"
            );
            assert_eq!(
                parse_url(r"file://\c:\Users\Jane Doe\data.csv")
                    .unwrap()
                    .as_str(),
                "file:///c:/Users/Jane%20Doe/data.csv"
            );
            assert_eq!(
                parse_url(r"c:\Users\Jane Doe\data.csv").unwrap().as_str(),
                "file:///C:/Users/Jane%20Doe/data.csv"
            );
            assert_eq!(
                parse_url(r"data.csv").unwrap().as_str(),
                url::Url::from_file_path(
                    [
                        std::env::current_dir().unwrap().as_path(),
                        std::path::Path::new("data.csv")
                    ]
                    .into_iter()
                    .collect::<std::path::PathBuf>()
                )
                .unwrap()
                .as_str()
            );
        }
        #[cfg(not(target_os = "windows"))]
        {
            assert_eq!(
                parse_url(r"file:///home/Jane Doe/data.csv")
                    .unwrap()
                    .as_str(),
                "file:///home/Jane%20Doe/data.csv"
            );
            assert_eq!(
                parse_url(r"/home/Jane Doe/data.csv").unwrap().as_str(),
                "file:///home/Jane%20Doe/data.csv"
            );
            assert_eq!(
                parse_url(r"data.csv").unwrap().as_str(),
                url::Url::from_file_path(
                    [
                        std::env::current_dir().unwrap().as_path(),
                        std::path::Path::new("data.csv")
                    ]
                    .into_iter()
                    .collect::<std::path::PathBuf>()
                )
                .unwrap()
                .as_str()
            );
        }
    }
}
