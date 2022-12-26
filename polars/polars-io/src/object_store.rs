//! Interface with the object_store crate and define AsyncSeek, AsyncRead.
//! This is used, for example, by the parquet2 crate.

use std::io::{self};
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;

use futures::executor::block_on;
use futures::future::{ready, BoxFuture};
use futures::lock::Mutex;
use futures::{AsyncRead, AsyncSeek, Future, StreamExt, TryFutureExt, TryStreamExt};
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::ObjectStore;
use polars_core::prelude::{PolarsError, PolarsResult};
use regex::Regex;
use url::Url;

type OptionalFuture = Arc<Mutex<Option<BoxFuture<'static, std::io::Result<Vec<u8>>>>>>;
const DELIMITER: char = '/';

/// Adaptor to translate from AsyncSeek and AsyncRead to the object_store get_range API.
pub struct CloudReader {
    // The current position in the stream, it is set by seeking and updated by reading bytes.
    pos: u64,
    // The total size of the object is required when seeking from the end of the file.
    length: Option<u64>,
    // Hold an reference to the store in a thread safe way.
    object_store: Arc<Mutex<Box<dyn ObjectStore>>>,
    // The path in the object_store of the current object being read.
    path: Path,
    // If a read is pending then `active` will point to its future.
    active: OptionalFuture,
}

impl CloudReader {
    pub fn new(
        length: Option<u64>,
        object_store: Arc<Mutex<Box<dyn ObjectStore>>>,
        path: Path,
    ) -> Self {
        Self {
            pos: 0,
            length,
            object_store,
            path,
            active: Arc::new(Mutex::new(None)),
        }
    }

    /// For each read request we create a new future.
    async fn read_operation(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        length: usize,
    ) -> std::task::Poll<std::io::Result<Vec<u8>>> {
        let start = self.pos as usize;

        // If we already have a future just poll it.
        if let Some(fut) = self.active.lock().await.as_mut() {
            return Future::poll(fut.as_mut(), cx);
        }

        // Create the future.
        let future = {
            let path = self.path.clone();
            let arc = self.object_store.clone();
            // Use an async move block to get our owned objects.
            async move {
                let object_store = arc.lock().await;
                object_store
                    .get_range(&path, start..start + length)
                    .map_ok(|r| r.to_vec())
                    .map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("object store error {e:?}"),
                        )
                    })
                    .await
            }
        };
        // Prepare for next read.
        self.pos += length as u64;

        let mut future = Box::pin(future);

        // Need to poll it once to get the pump going.
        let polled = Future::poll(future.as_mut(), cx);

        // Save for next time.
        let mut state = self.active.lock().await;
        *state = Some(future);
        polled
    }
}

impl AsyncRead for CloudReader {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut [u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        // Use block_on in order to get the future result in this thread and copy the data in the output buffer.
        // With this approach we keep ownership of the buffer and we don't have to pass it to the future runtime.
        match block_on(self.read_operation(cx, buf.len())) {
            Poll::Ready(Ok(bytes)) => {
                buf.copy_from_slice(&bytes);
                Poll::Ready(Ok(bytes.len()))
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl AsyncSeek for CloudReader {
    fn poll_seek(
        mut self: Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
        pos: io::SeekFrom,
    ) -> std::task::Poll<std::io::Result<u64>> {
        match pos {
            io::SeekFrom::Start(pos) => self.pos = pos,
            io::SeekFrom::End(pos) => {
                let length = self.length.ok_or::<io::Error>(io::Error::new(
                    std::io::ErrorKind::Other,
                    "Cannot seek from end of stream when length is unknown.",
                ))?;
                self.pos = (length as i64 + pos) as u64
            }
            io::SeekFrom::Current(pos) => self.pos = (self.pos as i64 + pos) as u64,
        };
        std::task::Poll::Ready(Ok(self.pos))
    }
}

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
    fn new(url: &str) -> PolarsResult<CloudLocation> {
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

/// Build an ObjectStore based on the URL and information from the environment. Return an object store and the path relative to the store.
pub fn build(url: &str) -> PolarsResult<(CloudLocation, Box<dyn ObjectStore>)> {
    let cloud_location = CloudLocation::new(url)?;
    let store = match cloud_location.scheme.as_str() {
        "s3" => {
            let s3 = AmazonS3Builder::from_env()
                .with_bucket_name(&cloud_location.bucket)
                .build()
                .map_err(anyhow::Error::from)?;
            Ok::<_, PolarsError>(Box::new(s3) as Box<dyn ObjectStore>)
        }
        "file" => {
            let local = LocalFileSystem::new();
            Ok::<_, PolarsError>(Box::new(local) as Box<dyn ObjectStore>)
        }
        _ => unimplemented!(),
    }?;
    Ok((cloud_location, store))
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
pub async fn glob(url: &str) -> PolarsResult<Vec<String>> {
    // Find the fixed prefix, up to the first '*'.

    let (
        CloudLocation {
            scheme,
            bucket,
            prefix,
            expansion,
        },
        store,
    ) = build(url)?;
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
