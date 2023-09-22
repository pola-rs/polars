//! This module is inspired from [arrow-datafusion](https://github.com/apache/arrow-datafusion/blob/f4c4ee1e7ffa97b089994162c3d754402f218503/datafusion/core/src/datasource/listing/url.rs) but decoupled from object_store crate in favour of opendal crate.

use std::collections::HashMap;
use std::path::{Component, PathBuf};

use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use glob::Pattern;
use itertools::Itertools;
use opendal::layers::RetryLayer;
use opendal::{
    Error as OpenDalError, ErrorKind as OpenDalErrorKind, Metadata, Metakey, Operator, Scheme,
};
use percent_encoding;
use polars_error::{to_compute_err, PolarsError, PolarsResult};
use url::Url;

pub(crate) const DELIMITER: &str = "/";

/// A parsed URL identifying files for a listing table, see [`ObjectListingUrl::parse`]
/// for more information on the supported expressions
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ObjectListingUrl {
    /// A URL that identifies a file or directory to list files from
    url: Url,
    /// The path prefix
    prefix: String,
    /// An optional glob expression used to filter files
    glob: Option<Pattern>,
}

impl ObjectListingUrl {
    /// Parse a provided string as a [[ObjectListingUrl]]
    ///
    /// # Paths without a Scheme
    ///
    /// If no scheme is provided, or the string is an absolute filesystem path
    /// as determined [`std::path::Path::is_absolute`], the string will be
    /// interpreted as a path on the local filesystem using the operating
    /// system's standard path delimiter, i.e. `\` on Windows, `/` on Unix.
    ///
    /// If the path contains any of `'?', '*', '['`, it will be considered
    /// a glob expression and resolved as described in the section below.
    ///
    /// Otherwise, the path will be resolved to an absolute path, returning
    /// an error if it does not exist, and converted to a [file URI]
    ///
    /// If you wish to specify a path that does not exist on the local
    /// machine you must provide it as a fully-qualified [file URI]
    /// e.g. `file:///myfile.txt`
    ///
    /// ## Glob File Paths
    ///
    /// If no scheme is provided, and the path contains a glob expression, it will
    /// be resolved as follows.
    ///
    /// The string up to the first path segment containing a glob expression will be extracted,
    /// and resolved in the same manner as a normal scheme-less path. That is, resolved to
    /// an absolute path on the local filesystem, returning an error if it does not exist,
    /// and converted to a [file URI]
    ///
    /// The remaining string will be interpreted as a [`Pattern`] and used as a
    /// filter when listing files from object storage
    ///
    /// [file URI]: https://en.wikipedia.org/wiki/File_URI_scheme
    pub fn parse(s: impl AsRef<str>) -> PolarsResult<Self> {
        let s = s.as_ref();

        // This is necessary to handle the case of a path starting with a drive letter
        if std::path::Path::new(s).is_absolute() {
            return Self::from_path(s);
        }

        match Url::parse(s) {
            Ok(mut url) => {
                let (prefix, glob) = Self::parse_to_prefix(url.path())?;
                url.set_path(prefix.as_str());
                Ok(Self { url, prefix, glob })
            },
            Err(url::ParseError::RelativeUrlWithoutBase) => Self::from_path(s),
            Err(e) => Err(PolarsError::Generic(Box::new(e))),
        }
    }

    fn from_path(url_path: &str) -> PolarsResult<Self> {
        let (prefix, glob) = Self::parse_to_prefix(url_path)?;
        let path = std::path::Path::new(prefix.as_str());
        if path.is_dir() {
            Url::from_directory_path(path)
        } else {
            Url::from_file_path(path)
        }
        .map(|mut url| {
            url.set_path(prefix.as_str());
            Self { url, prefix, glob }
        })
        .map_err(|_| to_compute_err(format!("Can not open path: {url_path}")))
    }

    /// Creates a new [`ObjectListingUrl`] interpreting `s` as a filesystem path
    fn parse_to_prefix(url_path: &str) -> PolarsResult<(String, Option<Pattern>)> {
        let (prefix, glob) = match split_glob_expression(url_path) {
            Some((prefix, glob)) => {
                let glob = Pattern::new(glob).map_err(|e| PolarsError::Generic(Box::new(e)))?;
                (prefix, Some(glob))
            },
            None => (url_path, None),
        };

        let prefix_path = std::path::Path::new(prefix);
        let normalized_prefix = normalize_path(prefix_path);

        let decoded_prefix =
            percent_encoding::percent_decode_str(normalized_prefix.to_string_lossy().as_ref())
                .decode_utf8_lossy()
                .to_string();

        let suffixed = if glob.is_some() || url_path.ends_with(DELIMITER) {
            decoded_prefix + DELIMITER
        } else {
            decoded_prefix
        };

        Ok((suffixed, glob))
    }

    /// Returns the URL scheme
    pub fn url(&self) -> Url {
        self.url.clone()
    }

    /// Returns the URL scheme
    pub fn scheme(&self) -> &str {
        self.url.scheme()
    }

    /// Return the prefix from which to list files
    pub fn prefix(&self) -> &str {
        self.prefix.as_str()
    }

    /// Return the prefix from which to list files
    pub fn prefix_as_path(&self) -> PathBuf {
        std::path::PathBuf::from(self.prefix.as_str())
    }

    /// Return the prefix from which to list files
    pub fn is_prefix_dir(&self) -> bool {
        self.prefix.ends_with(DELIMITER)
    }

    pub fn infer_operator(&self, opts: HashMap<String, String>) -> PolarsResult<Operator> {
        let mut _opts = opts;

        let scheme = match (self.url.scheme(), self.url.host_str()) {
            ("file", None) => Scheme::Fs,
            ("az", _) => Scheme::Azblob,
            ("adl" | "adls" | "abfs" | "abfss" | "azdfs" | "azdls", _) => Scheme::Azdls,
            ("s3" | "s3a", Some(bucket)) => {
                _opts.insert("bucket".to_string(), bucket.to_string());
                Scheme::S3
            },
            ("gs", Some(bucket)) => {
                _opts.insert("bucket".to_string(), bucket.to_string());
                Scheme::Gcs
            },
            (scheme, _) => {
                return Err(OpenDalError::new(
                    OpenDalErrorKind::Unsupported,
                    format!("Unable to recognise URL with scheme\"{scheme}\"",).as_str(),
                ))
                .map_err(|e| PolarsError::Generic(Box::new(e)))
            },
        };

        let root = if self.is_prefix_dir() {
            self.prefix().to_string()
        } else {
            let binding = self.prefix_as_path();
            let path_buf = binding.parent().and_then(|x| x.to_str());

            match path_buf {
                None => DELIMITER,
                Some("") => DELIMITER,
                Some(p) => p,
            }
            .to_string()
        };

        _opts.insert("root".to_string(), root);

        Operator::via_map(scheme, _opts)
            .map(|op| op.layer(RetryLayer::new()))
            .map_err(|e| PolarsError::Generic(Box::new(e)))
    }

    /// Returns `true` if `path` matches this [`ObjectListingUrl`]
    pub fn contains(&self, path: &str) -> bool {
        match self.strip_prefix(path) {
            Some(mut segments) => match &self.glob {
                Some(glob) => {
                    let stripped = segments.join("/");
                    glob.matches(&stripped)
                },
                None => true,
            },
            None => false,
        }
    }

    /// Strips the prefix of this [`ObjectListingUrl`] from the provided path, returning
    /// an iterator of the remaining path segments
    pub(crate) fn strip_prefix<'a, 'b: 'a>(
        &'a self,
        path: &'b str,
    ) -> Option<impl Iterator<Item = &'b str> + 'a> {
        Some(path.split_terminator(DELIMITER))
    }

    /// Streams all objects identified by this [`ObjectListingUrl`] for the provided options
    pub async fn glob_object_stream<'a>(
        &'a self,
        store: &'a Operator,
        file_extension: &'a str,
        exclude_empty: bool,
        recursive: bool,
    ) -> PolarsResult<BoxStream<'a, PolarsResult<(String, Metadata)>>> {
        let stream = if self.is_prefix_dir() {
            futures::stream::once(
                store
                    .lister_with(DELIMITER)
                    .delimiter(if recursive { "" } else { DELIMITER })
                    .metakey(Metakey::Mode | Metakey::ContentLength),
            )
            .try_flatten()
            .map_ok(|e| (e.path().to_string(), e.metadata().to_owned()))
            .map_err(|e| PolarsError::Generic(Box::new(e)))
            .boxed()
        } else {
            let binding = &self.prefix_as_path();
            let file_name = binding
                .file_name()
                .and_then(|x| x.to_str())
                .ok_or_else(|| {
                    to_compute_err(format!("cannot get file name from: {}", self.prefix()))
                })?;

            let metadata = store
                .stat(file_name)
                .await
                .map_err(|e| PolarsError::Generic(Box::new(e)))?;

            let object = (file_name.to_string(), metadata);
            futures::stream::once(async { Ok(object) }).boxed()
        }
        .try_filter(move |(path, metadata)| {
            let is_file = !path.ends_with(DELIMITER);
            let extension_match = path.ends_with(file_extension);
            let glob_match = self.contains(path);
            let is_empty = exclude_empty && metadata.content_length() == 0;

            futures::future::ready(is_file && extension_match && glob_match && !is_empty)
        })
        .boxed();

        Ok(stream)
    }

    /// Lists all objects identified by this [`ObjectListingUrl`] for the provided options
    pub async fn glob_object_list<'a>(
        &'a self,
        store: &'a Operator,
        file_extension: &'a str,
        exclude_empty: bool,
        recursive: bool,
    ) -> PolarsResult<Vec<(String, Metadata)>> {
        let stream = self
            .glob_object_stream(store, file_extension, exclude_empty, recursive)
            .await?;
        let list = stream.try_collect::<Vec<_>>().await?;

        Ok(list)
    }

    /// Returns this [`ObjectListingUrl`] as a string
    pub fn as_str(&self) -> &str {
        self.url.as_str()
    }
}

impl std::fmt::Display for ObjectListingUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}

const GLOB_START_CHARS: [char; 3] = ['?', '*', '['];

/// Splits `path` at the first path segment containing a glob expression, returning
/// `None` if no glob expression found.
///
/// Path delimiters are determined using [`std::path::is_separator`] which
/// permits `/` as a path delimiter even on Windows platforms.
///
fn split_glob_expression(path: &str) -> Option<(&str, &str)> {
    let mut last_separator = 0;

    for (byte_idx, char) in path.char_indices() {
        if GLOB_START_CHARS.contains(&char) {
            if last_separator == 0 {
                return Some((".", path));
            }
            return Some(path.split_at(last_separator));
        }

        if std::path::is_separator(char) {
            last_separator = byte_idx + char.len_utf8();
        }
    }
    None
}

/// Normalize a path, removing things like `.` and `..`.
///
/// CAUTION: This does not resolve symlinks (unlike
/// [`std::fs::canonicalize`]). This may cause incorrect or surprising
/// behavior at times. This should be used carefully. Unfortunately,
/// [`std::fs::canonicalize`] can be hard to use correctly, since it can often
/// fail, or on Windows returns annoying device paths.
fn normalize_path(path: &std::path::Path) -> PathBuf {
    let mut components = path.components().peekable();
    let mut ret = if let Some(c @ Component::Prefix(..)) = components.peek().cloned() {
        components.next();
        PathBuf::from(c.as_os_str())
    } else {
        PathBuf::new()
    };

    for component in components {
        match component {
            Component::Prefix(..) => unreachable!(),
            Component::RootDir => {
                ret.push(component.as_os_str());
            },
            Component::CurDir => {},
            Component::ParentDir => {
                ret.pop();
            },
            Component::Normal(c) => {
                ret.push(c);
            },
        }
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_path() {
        let root = std::env::current_dir().unwrap();
        let root = root.to_string_lossy();

        let url = ObjectListingUrl::parse(root).unwrap();
        let child = String::from("partition/file");
        let prefix: Vec<_> = url.strip_prefix(&child).unwrap().collect();
        assert_eq!(prefix, vec!["partition", "file"]);

        let url = ObjectListingUrl::parse("file:///").unwrap();
        let child = String::from("foo/bar");
        let prefix: Vec<_> = url.strip_prefix(&child).unwrap().collect();
        assert_eq!(prefix, vec!["foo", "bar"]);

        let url = ObjectListingUrl::parse("file:///foo").unwrap();
        let child = String::from("/foob/bar");
        assert!(url.strip_prefix(&child).is_some());

        let url = ObjectListingUrl::parse("file:///foo/file").unwrap();
        let child = String::from("foo/file");
        assert_eq!(url.strip_prefix(&child).unwrap().count(), 2);

        let url = ObjectListingUrl::parse("file:///foo/ bar").unwrap();
        assert_eq!(url.prefix, "/foo/ bar");

        let url = ObjectListingUrl::parse("file:///foo/bar?").unwrap();
        assert_eq!(url.prefix, "/foo/bar");
    }

    #[test]
    fn test_prefix_s3() {
        let url = ObjectListingUrl::parse("s3://bucket/foo/bar/").unwrap();
        assert_eq!(url.prefix(), "/foo/bar/");

        let child = String::from("partition/foo.parquet");
        let prefix: Vec<_> = url.strip_prefix(&child).unwrap().collect();
        assert_eq!(prefix, vec!["partition", "foo.parquet"]);
    }

    #[test]
    fn test_split_glob() {
        fn test(input: &str, expected: Option<(&str, &str)>) {
            assert_eq!(
                split_glob_expression(input),
                expected,
                "testing split_glob_expression with {input}"
            );
        }

        // no glob patterns
        test("/", None);
        test("/a.txt", None);
        test("/a", None);
        test("/a/", None);
        test("/a/b", None);
        test("/a/b/", None);
        test("/a/b.txt", None);
        test("/a/b/c.txt", None);
        // glob patterns, thus we build the longest path (os-specific)
        test("*.txt", Some((".", "*.txt")));
        test("/*.txt", Some(("/", "*.txt")));
        test("/a/*b.txt", Some(("/a/", "*b.txt")));
        test("/a/*/b.txt", Some(("/a/", "*/b.txt")));
        test("/a/b/[123]/file*.txt", Some(("/a/b/", "[123]/file*.txt")));
        test("/a/b*.txt", Some(("/a/", "b*.txt")));
        test("/a/b/**/c*.txt", Some(("/a/b/", "**/c*.txt")));

        // https://github.com/apache/arrow-datafusion/issues/2465
        test(
            "/a/b/c//alltypes_plain*.parquet",
            Some(("/a/b/c//", "alltypes_plain*.parquet")),
        );
    }
}
