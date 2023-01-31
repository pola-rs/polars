use futures::future::ready;
use futures::{StreamExt, TryStreamExt};
use object_store::path::Path;
use polars_core::cloud::CloudOptions;
use polars_core::prelude::{PolarsError, PolarsResult};
use regex::Regex;
use url::Url;

const DELIMITER: char = '/';

/// Split the url in
/// 1. the prefix part (all path components until the first one with '*')
/// 2. a regular expression representation of the rest.
fn extract_prefix_expansion(url: &str) -> PolarsResult<(String, Option<String>)> {
    let splits = url.split(DELIMITER);
    let mut prefix = String::new();
    let mut expansion = String::new();
    let mut last_split_was_wildcard = false;
    for split in splits {
        let has_star = split.contains('*');
        if expansion.is_empty() && !has_star {
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
        if split.contains("**") {
            return PolarsResult::Err(PolarsError::ComputeError(
                format!("Expected '**' by itself in path component, got {url}.").into(),
            ));
        }
        if !last_split_was_wildcard && !expansion.is_empty() {
            expansion.push(DELIMITER);
        }
        // Handle '.' inside a split.
        if split.contains('.') || split.contains('*') {
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
#[derive(PartialEq, Debug)]
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
    /// Parse a CloudLocation from an url.
    pub fn new(url: &str) -> PolarsResult<CloudLocation> {
        let parsed = Url::parse(url).map_err(anyhow::Error::from)?;
        let is_local = parsed.scheme() == "file";
        let (bucket, key) = if is_local {
            ("".into(), url[7..].into())
        } else {
            let key = parsed.path();
            let bucket = parsed
                .host()
                .ok_or(PolarsError::ComputeError(
                    format!("Cannot parse bucket (ie host) from {url}").into(),
                ))?
                .to_string();
            (bucket, key)
        };
        let (mut prefix, expansion) = extract_prefix_expansion(key)?;
        if is_local && key.starts_with(DELIMITER) {
            prefix.insert(0, DELIMITER);
        }
        Ok(CloudLocation {
            scheme: parsed.scheme().into(),
            bucket,
            prefix,
            expansion,
        })
    }
}

/// Return a full url from a key relative to the given location.
fn full_url(scheme: &str, bucket: &str, key: Path) -> String {
    format!("{scheme}://{bucket}/{key}")
}

/// A simple matcher, if more is required consider depending on https://crates.io/crates/globset.
/// The Cloud list api returns a list of all the file names under a prefix, there is no additional cost of `readdir`.
struct Matcher {
    prefix: String,
    re: Option<Regex>,
}

impl Matcher {
    /// Build a Matcher for the given prefix and expansion.
    fn new(prefix: String, expansion: Option<&str>) -> PolarsResult<Matcher> {
        // Cloud APIs accept a prefix without any expansion, extract it.
        let re = expansion
            .map(|ex| Regex::new(ex).map_err(anyhow::Error::from))
            .transpose()?;
        Ok(Matcher { prefix, re })
    }

    fn is_matching(&self, key: &Path) -> bool {
        let key: &str = key.as_ref();
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

#[tokio::main(flavor = "current_thread")]
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
    ) = super::build(url, cloud_options)?;
    let matcher = Matcher::new(prefix.clone(), expansion.as_deref())?;

    let list_stream = store
        .list(Some(&Path::from(prefix)))
        .await
        .map_err(anyhow::Error::from)?;
    let locations: Vec<Path> = list_stream
        .then(|entry| async {
            let entry = entry.map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
            Ok::<_, PolarsError>(entry.location)
        })
        .filter(|name| match name {
            PolarsResult::Ok(name) => ready(matcher.is_matching(name)),
            _ => ready(true),
        })
        .try_collect()
        .await?;
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
            CloudLocation::new("s3://a/b").unwrap(),
            CloudLocation {
                scheme: "s3".into(),
                bucket: "a".into(),
                prefix: "b".into(),
                expansion: None,
            }
        );
        assert_eq!(
            CloudLocation::new("s3://a/b/*.c").unwrap(),
            CloudLocation {
                scheme: "s3".into(),
                bucket: "a".into(),
                prefix: "b/".into(),
                expansion: Some("^([^/]*)\\.c$".into()),
            }
        );
        assert_eq!(
            CloudLocation::new("file:///a/b").unwrap(),
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
        let cloud_location = CloudLocation::new("s3://bucket/folder/*.parquet").unwrap();
        let a = Matcher::new(cloud_location.prefix, cloud_location.expansion.as_deref()).unwrap();
        // Regular match.
        assert!(a.is_matching(&Path::from("folder/1.parquet")));
        // Require . in the file name.
        assert!(!a.is_matching(&Path::from("folder/1parquet")));
        // Intermediary folders are not allowed.
        assert!(!a.is_matching(&Path::from("folder/other/1.parquet")));
    }

    #[test]
    fn test_matcher_folders() {
        let cloud_location = CloudLocation::new("s3://bucket/folder/**/*.parquet").unwrap();
        let a = Matcher::new(cloud_location.prefix, cloud_location.expansion.as_deref()).unwrap();
        // Intermediary folders are optional.
        assert!(a.is_matching(&Path::from("folder/1.parquet")));
        // Intermediary folders are allowed.
        assert!(a.is_matching(&Path::from("folder/other/1.parquet")));
        let cloud_location = CloudLocation::new("s3://bucket/folder/**/data/*.parquet").unwrap();
        let a = Matcher::new(cloud_location.prefix, cloud_location.expansion.as_deref()).unwrap();
        // Required folder `data` is missing.
        assert!(!a.is_matching(&Path::from("folder/1.parquet")));
        // Required folder is present.
        assert!(a.is_matching(&Path::from("folder/data/1.parquet")));
        // Required folder is present and additional folders are allowed.
        assert!(a.is_matching(&Path::from("folder/other/data/1.parquet")));
    }
}
