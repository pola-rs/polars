use std::borrow::Cow;

use futures::TryStreamExt;
use object_store::path::Path;
use polars_core::error::to_compute_err;
use polars_error::{PolarsResult, polars_bail};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;
use regex::Regex;
use url::Url;

use super::{CloudOptions, parse_url};

const DELIMITER: char = '/';

/// Converts a glob to regex form.
///
/// # Returns
/// 1. the prefix part (all path components until the first one with '*')
/// 2. a regular expression representation of the rest.
pub(crate) fn extract_prefix_expansion(url: &str) -> PolarsResult<(Cow<str>, Option<String>)> {
    let url = url.strip_prefix('/').unwrap_or(url);
    // (offset, len, replacement)
    let mut replacements: Vec<(usize, usize, &[u8])> = vec![];

    // The position after the last slash before glob characters begin.
    // `a/b/c*/`
    //      ^
    let mut pos: usize = if let Some(after_last_slash) = memchr::memchr2(b'*', b'[', url.as_bytes())
        .map(|i| {
            url.as_bytes()[..i]
                .iter()
                .rposition(|x| *x == b'/')
                .map_or(0, |x| 1 + x)
        }) {
        // First value is used as the starting point later.
        replacements.push((after_last_slash, 0, &[]));
        after_last_slash
    } else {
        usize::MAX
    };

    while pos < url.len() {
        match memchr::memchr2(b'*', b'.', &url.as_bytes()[pos..]) {
            None => break,
            Some(i) => pos += i,
        }

        let (len, replace): (usize, &[u8]) = match &url[pos..] {
            // Accept:
            // - `**/`
            // - `**` only if it is the end of the url
            v if v.starts_with("**") && (v.len() == 2 || v.as_bytes()[2] == b'/') => {
                // Wrapping in a capture group ensures we also match non-nested paths.
                (3, b"(.*/)?" as _)
            },
            v if v.starts_with("**") => {
                polars_bail!(ComputeError: "invalid ** glob pattern")
            },
            v if v.starts_with('*') => (1, b"[^/]*" as _),
            // Dots need to be escaped in regex.
            v if v.starts_with('.') => (1, b"\\." as _),
            _ => {
                pos += 1;
                continue;
            },
        };

        replacements.push((pos, len, replace));
        pos += len;
    }

    if replacements.is_empty() {
        return Ok((Cow::Borrowed(url), None));
    }

    let prefix = Cow::Borrowed(&url[..replacements[0].0]);

    let mut pos = replacements[0].0;
    let mut expansion = Vec::with_capacity(url.len() - pos);
    expansion.push(b'^');

    for (offset, len, replace) in replacements {
        expansion.extend_from_slice(&url.as_bytes()[pos..offset]);
        expansion.extend_from_slice(replace);
        pos = offset + len;
    }

    if pos < url.len() {
        expansion.extend_from_slice(&url.as_bytes()[pos..]);
    }

    expansion.push(b'$');

    Ok((prefix, Some(String::from_utf8(expansion).unwrap())))
}

/// A location on cloud storage, may have wildcards.
#[derive(PartialEq, Debug, Default)]
pub struct CloudLocation {
    /// The scheme (s3, ...).
    pub scheme: PlSmallStr,
    /// The bucket name.
    pub bucket: PlSmallStr,
    /// The prefix inside the bucket, this will be the full key when wildcards are not used.
    pub prefix: String,
    /// The path components that need to be expanded.
    pub expansion: Option<PlSmallStr>,
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

            let bucket = format_pl_smallstr!(
                "{}",
                &parsed[url::Position::BeforeUsername..url::Position::AfterPort]
            );

            if bucket.is_empty() {
                polars_bail!(ComputeError: "CloudLocation::from_url(): empty bucket: {}", parsed);
            }

            (bucket, key)
        };

        let key = percent_encoding::percent_decode_str(key)
            .decode_utf8()
            .map_err(to_compute_err)?;
        let (prefix, expansion) = if glob {
            let (prefix, expansion) = extract_prefix_expansion(&key)?;
            let mut prefix = prefix.into_owned();
            if is_local && key.starts_with(DELIMITER) && !prefix.starts_with(DELIMITER) {
                prefix.insert(0, DELIMITER);
            }
            (prefix, expansion.map(|x| x.into()))
        } else {
            (key.as_ref().into(), None)
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
        let re = expansion
            .map(polars_utils::regex_cache::compile_regex)
            .transpose()?;
        Ok(Matcher { prefix, re })
    }

    pub(crate) fn is_matching(&self, key: &str) -> bool {
        if !key.starts_with(self.prefix.as_str()) {
            // Prefix does not match, should not happen.
            return false;
        }
        if self.re.is_none() {
            return true;
        }
        let last = &key[self.prefix.len()..];
        self.re.as_ref().unwrap().is_match(last.as_ref())
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
            prefix[1..].into()
        } else {
            prefix.clone()
        },
        expansion.as_deref(),
    )?;

    let path = Path::from(prefix.as_str());
    let path = Some(&path);

    let mut locations = store
        .try_exec_rebuild_on_err(|store| {
            let st = store.clone();

            async {
                let store = st;
                store
                    .list(path)
                    .try_filter_map(|x| async move {
                        let out = (x.size > 0 && matcher.is_matching(x.location.as_ref()))
                            .then_some(x.location);
                        Ok(out)
                    })
                    .try_collect::<Vec<_>>()
                    .await
                    .map_err(to_compute_err)
            }
        })
        .await?;

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
                expansion: Some("^[^/]*\\.c$".into()),
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
            ("a/".into(), Some("^(.*/)?$".into()))
        );
        assert_eq!(
            extract_prefix_expansion("a/**/b").unwrap(),
            ("a/".into(), Some("^(.*/)?b$".into()))
        );
        assert_eq!(
            extract_prefix_expansion("a/**/*b").unwrap(),
            ("a/".into(), Some("^(.*/)?[^/]*b$".into()))
        );
        assert_eq!(
            extract_prefix_expansion("a/**/data/*b").unwrap(),
            ("a/".into(), Some("^(.*/)?data/[^/]*b$".into()))
        );
        assert_eq!(
            extract_prefix_expansion("a/*b").unwrap(),
            ("a/".into(), Some("^[^/]*b$".into()))
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

    #[test]
    fn test_glob_wildcard_21736() {
        let url = "s3://bucket/folder/**/data.parquet";
        let cloud_location = CloudLocation::new(url, true).unwrap();

        let a = Matcher::new(cloud_location.prefix, cloud_location.expansion.as_deref()).unwrap();

        assert!(!a.is_matching("folder/_data.parquet"));

        assert!(a.is_matching("folder/data.parquet"));
        assert!(a.is_matching("folder/abc/data.parquet"));
        assert!(a.is_matching("folder/abc/def/data.parquet"));

        let url = "s3://bucket/folder/data_*.parquet";
        let cloud_location = CloudLocation::new(url, true).unwrap();

        let a = Matcher::new(cloud_location.prefix, cloud_location.expansion.as_deref()).unwrap();

        assert!(!a.is_matching("folder/data_1.ipc"))
    }
}
