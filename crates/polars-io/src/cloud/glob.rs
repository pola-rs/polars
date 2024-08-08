use futures::TryStreamExt;
use object_store::path::Path;
use polars_core::error::to_compute_err;
use polars_core::prelude::{polars_ensure, polars_err};
use polars_error::PolarsResult;
use regex::Regex;
use url::Url;

use super::{parse_url, CloudOptions};

const DELIMITER: char = '/';

/// Split the url in
/// 1. the prefix part (all path components until the first one with '*')
/// 2. a regular expression representation of the rest.
pub(crate) fn extract_prefix_expansion(url: &str) -> PolarsResult<(String, Option<String>)> {
    let splits = url.split(DELIMITER);
    let mut prefix = String::new();
    let mut expansion = String::new();
    let mut last_split_was_wildcard = false;
    for split in splits {
        if expansion.is_empty() && memchr::memchr2(b'*', b'[', split.as_bytes()).is_none() {
            // We are still gathering splits in the prefix.
            if !prefix.is_empty() {
                prefix.push(DELIMITER);
            }
            prefix.push_str(split);
            continue;
        }
        // We are gathering splits for the expansion.
        //
        // Handle '**', we expect them to be by themselves in a split.
        if split == "**" {
            last_split_was_wildcard = true;
            expansion.push_str(".*");
            continue;
        }
        polars_ensure!(
            !split.contains("**"),
            ComputeError: "expected '**' by itself in path component, got {}", url
        );
        if !last_split_was_wildcard && !expansion.is_empty() {
            expansion.push(DELIMITER);
        }
        // Handle '.' inside a split.
        if memchr::memchr2(b'.', b'*', split.as_bytes()).is_some() {
            let processed = split.replace('.', "\\.");
            expansion.push_str(&processed.replace('*', "([^/]*)"));
            continue;
        }
        last_split_was_wildcard = false;
        expansion.push_str(split);
    }
    // Prefix post-processing: when present, prefix should end with '/' in order to simplify matching.
    if !prefix.is_empty() && !expansion.is_empty() {
        prefix.push(DELIMITER);
    }
    // Expansion post-processing: when present, expansion should cover the whole input.
    if !expansion.is_empty() {
        expansion.insert(0, '^');
        expansion.push('$');
    }
    Ok((
        prefix,
        if !expansion.is_empty() {
            Some(expansion)
        } else {
            None
        },
    ))
}

/// A location on cloud storage, may have wildcards.
#[derive(PartialEq, Debug, Default)]
pub struct CloudLocation {
    /// The scheme (s3, ...).
    pub scheme: String,
    /// The bucket name.
    pub bucket: String,
    /// The prefix inside the bucket, this will be the full key when wildcards are not used.
    pub prefix: String,
    /// The path components that need to be expanded.
    pub expansion: Option<String>,
}

impl CloudLocation {
    pub fn from_url(parsed: &Url, glob: bool) -> PolarsResult<CloudLocation> {
        let is_local = parsed.scheme() == "file";
        let (bucket, key) = if is_local {
            ("".into(), parsed.path())
        } else {
            if parsed.scheme().starts_with("http") {
                return Ok(CloudLocation {
                    scheme: parsed.scheme().into(),
                    ..Default::default()
                });
            }

            let key = parsed.path();
            let bucket = parsed
                .host()
                .ok_or_else(
                    || polars_err!(ComputeError: "cannot parse bucket (host) from url: {}", parsed),
                )?
                .to_string();
            (bucket, key)
        };

        let key = percent_encoding::percent_decode_str(key)
            .decode_utf8()
            .map_err(to_compute_err)?;
        let (prefix, expansion) = if glob {
            let (mut prefix, expansion) = extract_prefix_expansion(&key)?;
            if is_local && key.starts_with(DELIMITER) {
                prefix.insert(0, DELIMITER);
            }
            (prefix, expansion)
        } else {
            (key.to_string(), None)
        };

        Ok(CloudLocation {
            scheme: parsed.scheme().into(),
            bucket,
            prefix,
            expansion,
        })
    }

    /// Parse a CloudLocation from an url.
    pub fn new(url: &str, glob: bool) -> PolarsResult<CloudLocation> {
        let parsed = parse_url(url).map_err(to_compute_err)?;
        Self::from_url(&parsed, glob)
    }
}

/// Return a full url from a key relative to the given location.
fn full_url(scheme: &str, bucket: &str, key: Path) -> String {
    format!("{scheme}://{bucket}/{key}")
}

/// A simple matcher, if more is required consider depending on https://crates.io/crates/globset.
/// The Cloud list api returns a list of all the file names under a prefix, there is no additional cost of `readdir`.
pub(crate) struct Matcher {
    prefix: String,
    re: Option<Regex>,
}

impl Matcher {
    /// Build a Matcher for the given prefix and expansion.
    pub(crate) fn new(prefix: String, expansion: Option<&str>) -> PolarsResult<Matcher> {
        // Cloud APIs accept a prefix without any expansion, extract it.
        let re = expansion.map(Regex::new).transpose()?;
        Ok(Matcher { prefix, re })
    }

    pub(crate) fn is_matching(&self, key: &str) -> bool {
        if !key.starts_with(&self.prefix) {
            // Prefix does not match, should not happen.
            return false;
        }
        if self.re.is_none() {
            return true;
        }
        let last = &key[self.prefix.len()..];
        return self.re.as_ref().unwrap().is_match(last.as_ref());
    }
}

/// List files with a prefix derived from the pattern.
pub async fn glob(url: &str, cloud_options: Option<&CloudOptions>) -> PolarsResult<Vec<String>> {
    // Find the fixed prefix, up to the first '*'.

    let (
        CloudLocation {
            scheme,
            bucket,
            prefix,
            expansion,
        },
        store,
    ) = super::build_object_store(url, cloud_options, true).await?;
    let matcher = &Matcher::new(
        if scheme == "file" {
            // For local paths the returned location has the leading slash stripped.
            prefix[1..].to_string()
        } else {
            prefix.clone()
        },
        expansion.as_deref(),
    )?;

    let mut locations = store
        .list(Some(&Path::from(prefix)))
        .try_filter_map(|x| async move {
            let out =
                (x.size > 0 && matcher.is_matching(x.location.as_ref())).then_some(x.location);
            Ok(out)
        })
        .try_collect::<Vec<_>>()
        .await
        .map_err(to_compute_err)?;

    locations.sort_unstable();
    Ok(locations
        .into_iter()
        .map(|l| full_url(&scheme, &bucket, l))
        .collect::<Vec<_>>())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cloud_location() {
        assert_eq!(
            CloudLocation::new("s3://a/b", true).unwrap(),
            CloudLocation {
                scheme: "s3".into(),
                bucket: "a".into(),
                prefix: "b".into(),
                expansion: None,
            }
        );
        assert_eq!(
            CloudLocation::new("s3://a/b/*.c", true).unwrap(),
            CloudLocation {
                scheme: "s3".into(),
                bucket: "a".into(),
                prefix: "b/".into(),
                expansion: Some("^([^/]*)\\.c$".into()),
            }
        );
        assert_eq!(
            CloudLocation::new("file:///a/b", true).unwrap(),
            CloudLocation {
                scheme: "file".into(),
                bucket: "".into(),
                prefix: "/a/b".into(),
                expansion: None,
            }
        );
    }

    #[test]
    fn test_extract_prefix_expansion() {
        assert!(extract_prefix_expansion("**url").is_err());
        assert_eq!(
            extract_prefix_expansion("a/b.c").unwrap(),
            ("a/b.c".into(), None)
        );
        assert_eq!(
            extract_prefix_expansion("a/**").unwrap(),
            ("a/".into(), Some("^.*$".into()))
        );
        assert_eq!(
            extract_prefix_expansion("a/**/b").unwrap(),
            ("a/".into(), Some("^.*b$".into()))
        );
        assert_eq!(
            extract_prefix_expansion("a/**/*b").unwrap(),
            ("a/".into(), Some("^.*([^/]*)b$".into()))
        );
        assert_eq!(
            extract_prefix_expansion("a/**/data/*b").unwrap(),
            ("a/".into(), Some("^.*data/([^/]*)b$".into()))
        );
        assert_eq!(
            extract_prefix_expansion("a/*b").unwrap(),
            ("a/".into(), Some("^([^/]*)b$".into()))
        );
    }

    #[test]
    fn test_matcher_file_name() {
        let cloud_location = CloudLocation::new("s3://bucket/folder/*.parquet", true).unwrap();
        let a = Matcher::new(cloud_location.prefix, cloud_location.expansion.as_deref()).unwrap();
        // Regular match.
        assert!(a.is_matching(Path::from("folder/1.parquet").as_ref()));
        // Require . in the file name.
        assert!(!a.is_matching(Path::from("folder/1parquet").as_ref()));
        // Intermediary folders are not allowed.
        assert!(!a.is_matching(Path::from("folder/other/1.parquet").as_ref()));
    }

    #[test]
    fn test_matcher_folders() {
        let cloud_location = CloudLocation::new("s3://bucket/folder/**/*.parquet", true).unwrap();
        let a = Matcher::new(cloud_location.prefix, cloud_location.expansion.as_deref()).unwrap();
        // Intermediary folders are optional.
        assert!(a.is_matching(Path::from("folder/1.parquet").as_ref()));
        // Intermediary folders are allowed.
        assert!(a.is_matching(Path::from("folder/other/1.parquet").as_ref()));
        let cloud_location =
            CloudLocation::new("s3://bucket/folder/**/data/*.parquet", true).unwrap();
        let a = Matcher::new(cloud_location.prefix, cloud_location.expansion.as_deref()).unwrap();
        // Required folder `data` is missing.
        assert!(!a.is_matching(Path::from("folder/1.parquet").as_ref()));
        // Required folder is present.
        assert!(a.is_matching(Path::from("folder/data/1.parquet").as_ref()));
        // Required folder is present and additional folders are allowed.
        assert!(a.is_matching(Path::from("folder/other/data/1.parquet").as_ref()));
    }

    #[test]
    fn test_cloud_location_no_glob() {
        let cloud_location = CloudLocation::new("s3://bucket/[*", false).unwrap();
        assert_eq!(
            cloud_location,
            CloudLocation {
                scheme: "s3".into(),
                bucket: "bucket".into(),
                prefix: "/[*".into(),
                expansion: None,
            },
        )
    }

    #[test]
    fn test_cloud_location_percentages() {
        use super::CloudLocation;

        let path = "s3://bucket/%25";
        let cloud_location = CloudLocation::new(path, true).unwrap();

        assert_eq!(
            cloud_location,
            CloudLocation {
                scheme: "s3".into(),
                bucket: "bucket".into(),
                prefix: "%25".into(),
                expansion: None,
            }
        );

        let path = "https://pola.rs/%25";
        let cloud_location = CloudLocation::new(path, true).unwrap();

        assert_eq!(
            cloud_location,
            CloudLocation {
                scheme: "https".into(),
                bucket: "".into(),
                prefix: "".into(),
                expansion: None,
            }
        );
    }
}
