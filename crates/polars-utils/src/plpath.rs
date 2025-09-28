use core::fmt;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A Path or URI
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum PlPath {
    Local(Arc<Path>),
    Cloud(PlCloudPath),
}

/// A reference to a Path or URI
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PlPathRef<'a> {
    Local(&'a Path),
    Cloud(PlCloudPathRef<'a>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct PlCloudPath {
    /// The scheme used in cloud e.g. `s3://` or `file://`.
    scheme: CloudScheme,
    /// The full URI e.g. `s3://path/to/bucket`.
    uri: Arc<str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlCloudPathRef<'a> {
    /// The scheme used in cloud e.g. `s3://` or `file://`.
    scheme: CloudScheme,
    /// The full URI e.g. `s3://path/to/bucket`.
    uri: &'a str,
}

impl<'a> fmt::Display for PlCloudPathRef<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.uri())
    }
}

impl fmt::Display for PlCloudPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl PlCloudPath {
    pub fn as_ref(&self) -> PlCloudPathRef<'_> {
        PlCloudPathRef {
            scheme: self.scheme,
            uri: self.uri.as_ref(),
        }
    }

    pub fn strip_scheme(&self) -> &str {
        self.scheme.strip_scheme_from_uri(&self.uri)
    }
}

impl PlCloudPathRef<'_> {
    pub fn new<'a>(uri: &'a str) -> Option<PlCloudPathRef<'a>> {
        CloudScheme::from_uri(uri).map(|scheme| PlCloudPathRef { scheme, uri })
    }

    pub fn into_owned(self) -> PlCloudPath {
        PlCloudPath {
            scheme: self.scheme,
            uri: self.uri.into(),
        }
    }

    pub fn scheme(&self) -> CloudScheme {
        self.scheme
    }

    pub fn uri(&self) -> &str {
        self.uri
    }

    pub fn strip_scheme(&self) -> &str {
        self.scheme.strip_scheme_from_uri(self.uri)
    }
}

pub struct PlPathDisplay<'a> {
    path: PlPathRef<'a>,
}

impl<'a> fmt::Display for PlPathDisplay<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.path {
            PlPathRef::Local(p) => p.display().fmt(f),
            PlPathRef::Cloud(p) => p.fmt(f),
        }
    }
}

macro_rules! impl_cloud_scheme {
    ($($t:ident = $n:literal,)+) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
        pub enum CloudScheme {
            $($t,)+
        }

        impl CloudScheme {
             /// Note, private function. Users should use [`CloudScheme::from_uri`], that will handle e.g.
            /// `file:/` without hostname properly.
            #[expect(unreachable_patterns)]
            fn from_scheme(s: &str) -> Option<Self> {
                Some(match s {
                    $($n => Self::$t,)+
                    _ => return None,
                })
            }

            pub const fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$t => $n,)+
                }
            }
        }
    };
}

impl_cloud_scheme! {
    Abfs = "abfs",
    Abfss = "abfss",
    Adl = "adl",
    Az = "az",
    Azure = "azure",
    File = "file",
    FileNoHostname = "file",
    Gcs = "gcs",
    Gs = "gs",
    Hf = "hf",
    Http = "http",
    Https = "https",
    S3 = "s3",
    S3a = "s3a",
}

impl CloudScheme {
    pub fn from_uri(path: &str) -> Option<Self> {
        if path.starts_with("file:/") {
            return Some(if path.as_bytes().get(6) != Some(&b'/') {
                Self::FileNoHostname
            } else {
                Self::File
            });
        }

        Self::from_scheme(&path[..path.find("://")?])
    }

    pub fn strip_scheme_from_uri<'a>(&self, uri: &'a str) -> &'a str {
        &uri[self.strip_scheme_index()..]
    }

    /// Returns `i` such that `&self.as_str()[i..]` strips the scheme, as well as the `://` if it
    /// exists.
    pub fn strip_scheme_index(&self) -> usize {
        if let Self::FileNoHostname = self {
            5
        } else {
            self.as_str().len() + 3
        }
    }
}

impl fmt::Display for CloudScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<'a> PlPathRef<'a> {
    pub fn scheme(&self) -> Option<CloudScheme> {
        match self {
            Self::Local(_) => None,
            Self::Cloud(p) => Some(p.scheme),
        }
    }

    pub fn is_local(&self) -> bool {
        matches!(self, Self::Local(_))
    }

    pub fn is_cloud_url(&self) -> bool {
        matches!(self, Self::Cloud(_))
    }

    pub fn as_local_path(&self) -> Option<&Path> {
        match self {
            Self::Local(p) => Some(p),
            Self::Cloud(_) => None,
        }
    }

    pub fn as_cloud_path(&'a self) -> Option<PlCloudPathRef<'a>> {
        match self {
            Self::Local(_) => None,
            Self::Cloud(p) => Some(*p),
        }
    }

    pub fn join(&self, other: impl AsRef<str>) -> PlPath {
        let other = other.as_ref();
        if other.is_empty() {
            return self.into_owned();
        }

        match self {
            Self::Local(p) => PlPath::Local(p.join(other).into()),
            Self::Cloud(p) => {
                if let Some(cloud_path) = PlCloudPathRef::new(other) {
                    return PlPath::Cloud(cloud_path.into_owned());
                }

                let needs_slash = !p.uri.ends_with('/') && !other.starts_with('/');

                let mut out =
                    String::with_capacity(p.uri.len() + usize::from(needs_slash) + other.len());

                out.push_str(p.uri);
                if needs_slash {
                    out.push('/');
                }
                // NOTE: This has as a consequence that pushing an absolute path into a URI
                // just pushes the slashes while for a path it will make that absolute path the new
                // path. I think this is acceptable as I don't really know what the alternative
                // would be.
                out.push_str(other);

                let uri = out.into();
                PlPath::Cloud(PlCloudPath {
                    scheme: p.scheme,
                    uri,
                })
            },
        }
    }

    pub fn display(&self) -> PlPathDisplay<'_> {
        PlPathDisplay { path: *self }
    }

    pub fn from_local_path(path: &'a Path) -> Self {
        Self::Local(path)
    }

    pub fn new(uri: &'a str) -> Self {
        if let Some(scheme) = CloudScheme::from_uri(uri) {
            Self::Cloud(PlCloudPathRef { scheme, uri })
        } else {
            Self::from_local_path(Path::new(uri))
        }
    }

    pub fn into_owned(self) -> PlPath {
        match self {
            Self::Local(p) => PlPath::Local(p.into()),
            Self::Cloud(p) => PlPath::Cloud(p.into_owned()),
        }
    }

    pub fn strip_scheme(&self) -> &str {
        match self {
            Self::Local(p) => p.to_str().unwrap(),
            Self::Cloud(p) => p.strip_scheme(),
        }
    }

    pub fn parent(&self) -> Option<Self> {
        Some(match self {
            Self::Local(p) => Self::Local(p.parent()?),
            Self::Cloud(p) => {
                let uri = p.uri;
                let offset_start = p.scheme.strip_scheme_index();
                let last_slash = uri[offset_start..]
                    .char_indices()
                    .rev()
                    .find(|(_, c)| *c == '/')?
                    .0;
                let uri = &uri[..offset_start + last_slash];

                Self::Cloud(PlCloudPathRef {
                    scheme: p.scheme,
                    uri,
                })
            },
        })
    }

    pub fn file_name(&self) -> Option<&OsStr> {
        match self {
            Self::Local(p) => {
                if p.is_dir() {
                    None
                } else {
                    p.file_name()
                }
            },
            Self::Cloud(p) => {
                if p.scheme() == CloudScheme::File
                    && std::fs::metadata(p.strip_scheme()).is_ok_and(|x| x.is_dir())
                {
                    return None;
                }

                let p = p.strip_scheme();
                let out = p.rfind('/').map_or(p, |i| &p[i + 1..]);
                (!out.is_empty()).then_some(out.as_ref())
            },
        }
    }

    pub fn extension(&self) -> Option<&str> {
        match self {
            Self::Local(path) => path.extension().and_then(|e| e.to_str()),
            Self::Cloud(_) => {
                let after_scheme = self.strip_scheme();

                after_scheme.rfind(['.', '/']).and_then(|i| {
                    after_scheme[i..]
                        .starts_with('.')
                        .then_some(&after_scheme[i + 1..])
                })
            },
        }
    }

    pub fn to_str(&self) -> &'a str {
        match self {
            Self::Local(p) => p.to_str().unwrap(),
            Self::Cloud(p) => p.uri,
        }
    }

    // It is up to the caller to ensure that the offset parameter 'n' matches
    // a valid path segment starting index
    pub fn offset_bytes(&'a self, n: usize) -> PathBuf {
        let s = self.to_str();
        if let Some(scheme) = self.scheme()
            && n > 0
        {
            debug_assert!(n >= scheme.as_str().len())
        }
        PathBuf::from(&s[n..])
    }

    /// Strips the scheme, then returns the authority component, and the remaining
    /// string after the authority component. This can be understood as extracting
    /// the bucket/prefix for cloud URIs.
    ///
    ///  E.g. `https://user@host:port/dir/file?param=value`
    /// * Authority: `user@host:port`
    /// * Remaining: `/dir/file?param=value`
    ///
    /// Note, for local / `file:` URIs, the returned authority will be empty, and
    /// the remainder will be the full URI.
    ///
    /// # Returns
    /// (authority, remaining).
    pub fn strip_scheme_split_authority(&self) -> Option<(&'_ str, &'_ str)> {
        match self.scheme() {
            None | Some(CloudScheme::File | CloudScheme::FileNoHostname) => {
                Some(("", self.strip_scheme()))
            },
            Some(scheme) => {
                let path_str = self.to_str();
                let position = self.authority_end_position();

                if position < path_str.len() {
                    assert!(path_str[position..].starts_with('/'));
                }

                (position < path_str.len()).then_some((
                    &path_str[scheme.strip_scheme_index()..position],
                    &path_str[position..],
                ))
            },
        }
    }

    /// Returns `i` such that `&self.to_str()[..i]` trims to the authority. If there is no '/'
    /// separator found, `i` will simply be the length of the string.
    pub fn authority_end_position(&self) -> usize {
        match self.scheme() {
            None | Some(CloudScheme::File | CloudScheme::FileNoHostname) => 0,
            Some(_) => {
                let after_scheme = self.strip_scheme();
                let offset = self.to_str().len() - after_scheme.len();

                offset + after_scheme.find('/').unwrap_or(after_scheme.len())
            },
        }
    }

    /// # Returns
    /// Returns an absolute local path if this path ref is a relative local path, otherwise returns None.
    pub fn to_absolute_path(&self) -> Option<PlPath> {
        if let Self::Local(p) = self
            && !p.is_absolute()
            && !p.to_str().unwrap().is_empty()
        {
            Some(PlPath::new(
                std::path::absolute(p).unwrap().to_str().unwrap(),
            ))
        } else {
            None
        }
    }
}

impl PlPath {
    pub fn new(uri: &str) -> Self {
        PlPathRef::new(uri).into_owned()
    }

    pub fn display(&self) -> PlPathDisplay<'_> {
        PlPathDisplay {
            path: match self {
                Self::Local(p) => PlPathRef::Local(p.as_ref()),
                Self::Cloud(p) => PlPathRef::Cloud(p.as_ref()),
            },
        }
    }

    pub fn is_local(&self) -> bool {
        self.as_ref().is_local()
    }

    pub fn is_cloud_url(&self) -> bool {
        self.as_ref().is_cloud_url()
    }

    // We don't want FromStr since we are infallible.
    #[expect(clippy::should_implement_trait)]
    pub fn from_str(uri: &str) -> Self {
        Self::new(uri)
    }

    pub fn from_string(uri: String) -> Self {
        Self::new(&uri)
    }

    pub fn as_ref(&self) -> PlPathRef<'_> {
        match self {
            Self::Local(p) => PlPathRef::Local(p.as_ref()),
            Self::Cloud(p) => PlPathRef::Cloud(p.as_ref()),
        }
    }

    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        match self {
            Self::Local(_) => None,
            Self::Cloud(p) => Some(p.scheme),
        }
    }

    pub fn to_str(&self) -> &str {
        match self {
            Self::Local(p) => p.to_str().unwrap(),
            Self::Cloud(p) => p.uri.as_ref(),
        }
    }

    pub fn into_local_path(self) -> Option<Arc<Path>> {
        match self {
            PlPath::Local(path) => Some(path),
            PlPath::Cloud(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plpath_file() {
        let p = PlPath::new("file:///home/user");
        assert_eq!(
            (
                p.cloud_scheme(),
                p.cloud_scheme().map(|x| x.as_str()),
                p.to_str(),
                p.as_ref().strip_scheme(),
            ),
            (
                Some(CloudScheme::File),
                Some("file"),
                "file:///home/user",
                "/home/user"
            )
        );

        let p = PlPath::new("file:/home/user");
        assert_eq!(
            (
                p.cloud_scheme(),
                p.cloud_scheme().map(|x| x.as_str()),
                p.to_str(),
                p.as_ref().strip_scheme(),
            ),
            (
                Some(CloudScheme::FileNoHostname),
                Some("file"),
                "file:/home/user",
                "/home/user"
            )
        );
    }

    #[test]
    fn plpath_join() {
        fn _assert_plpath_join(base: &str, added: &str, expect: &str, expect_uri: Option<&str>) {
            // Normal path test
            let path_base = base
                .chars()
                .map(|c| match c {
                    '/' => std::path::MAIN_SEPARATOR,
                    c => c,
                })
                .collect::<String>();
            let path_added = added
                .chars()
                .map(|c| match c {
                    '/' => std::path::MAIN_SEPARATOR,
                    c => c,
                })
                .collect::<String>();
            let path_result = expect
                .chars()
                .map(|c| match c {
                    '/' => std::path::MAIN_SEPARATOR,
                    c => c,
                })
                .collect::<String>();
            assert_eq!(
                PlPath::new(&path_base).as_ref().join(path_added).to_str(),
                path_result
            );

            if let Some(expect_uri) = expect_uri {
                // URI path test
                let uri_base = format!("file://{base}");

                let uri_result = format!("file://{expect_uri}");
                assert_eq!(
                    PlPath::new(uri_base.as_str()).as_ref().join(added).to_str(),
                    uri_result.as_str()
                );
            }
        }

        macro_rules! assert_plpath_join {
            ($base:literal + $added:literal => $expect:literal) => {
                _assert_plpath_join($base, $added, $expect, None)
            };
            ($base:literal + $added:literal => $expect:literal, $uri_result:literal) => {
                _assert_plpath_join($base, $added, $expect, Some($uri_result))
            };
        }

        assert_plpath_join!("a/b/c/" + "d/e" => "a/b/c/d/e");
        assert_plpath_join!("a/b/c" + "d/e" => "a/b/c/d/e");
        assert_plpath_join!("a/b/c" + "d/e/" => "a/b/c/d/e/");
        assert_plpath_join!("a/b/c" + "" => "a/b/c");
        assert_plpath_join!("a/b/c" + "/d" => "/d", "a/b/c/d");
        assert_plpath_join!("a/b/c" + "/d/" => "/d/", "a/b/c/d/");
        assert_plpath_join!("" + "/d/" => "/d/");
        assert_plpath_join!("/" + "/d/" => "/d/", "//d/");
        assert_plpath_join!("/x/y" + "/d/" => "/d/", "/x/y/d/");
        assert_plpath_join!("/x/y" + "/d" => "/d", "/x/y/d");
        assert_plpath_join!("/x/y" + "d" => "/x/y/d");

        assert_plpath_join!("/a/longer" + "path" => "/a/longer/path");
        assert_plpath_join!("/a/longer" + "/path" => "/path", "/a/longer/path");
        assert_plpath_join!("/a/longer" + "path/wow" => "/a/longer/path/wow");
        assert_plpath_join!("/a/longer" + "/path/wow" => "/path/wow", "/a/longer/path/wow");
        assert_plpath_join!("/an/even/longer" + "path" => "/an/even/longer/path");
        assert_plpath_join!("/an/even/longer" + "/path" => "/path", "/an/even/longer/path");
        assert_plpath_join!("/an/even/longer" + "path/wow" => "/an/even/longer/path/wow");
        assert_plpath_join!("/an/even/longer" + "/path/wow" => "/path/wow", "/an/even/longer/path/wow");
    }

    #[test]
    fn test_plpath_name() {
        assert_eq!(PlPathRef::new("s3://...").file_name(), Some("...".as_ref()));
        assert_eq!(
            PlPathRef::new("a/b/file.parquet").file_name(),
            Some("file.parquet".as_ref())
        );
        assert_eq!(
            PlPathRef::new("file.parquet").file_name(),
            Some("file.parquet".as_ref())
        );

        assert_eq!(PlPathRef::new("s3://").file_name(), None);
        assert_eq!(PlPathRef::new("").file_name(), None);
    }
}
