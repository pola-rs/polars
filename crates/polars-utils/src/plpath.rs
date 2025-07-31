use core::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
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
        &self.uri[self.scheme.as_str().len() + 3..]
    }
}

impl PlCloudPathRef<'_> {
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
        &self.uri[self.scheme.as_str().len() + "://".len()..]
    }
}

pub struct AddressDisplay<'a> {
    addr: PlPathRef<'a>,
}

impl<'a> fmt::Display for AddressDisplay<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.addr {
            PlPathRef::Local(p) => p.display().fmt(f),
            PlPathRef::Cloud(p) => p.fmt(f),
        }
    }
}

macro_rules! impl_scheme {
    ($($t:ident = $n:literal,)+) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
        pub enum CloudScheme {
            $($t,)+
        }

        impl FromStr for CloudScheme {
            type Err = ();

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($n => Ok(Self::$t),)+
                    _ => Err(()),
                }
            }
        }

        impl CloudScheme {
            pub fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$t => $n,)+
                }
            }
        }
    };
}

impl_scheme! {
    S3 = "s3",
    S3a = "s3a",
    Gs = "gs",
    Gcs = "gcs",
    File = "file",
    Abfs = "abfs",
    Abfss = "abfss",
    Azure = "azure",
    Az = "az",
    Adl = "adl",
    Http = "http",
    Https = "https",
    Hf = "hf",
}

impl fmt::Display for CloudScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

crate::regex_cache::cached_regex! {
    static CLOUD_SCHEME_REGEX = r"^(s3a?|gs|gcs|file|abfss?|azure|az|adl|https?|hf)$";
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

    pub fn as_cloud_addr(&self) -> Option<PlCloudPathRef<'_>> {
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

    pub fn display(&self) -> AddressDisplay<'_> {
        AddressDisplay { addr: *self }
    }

    pub fn from_local_path(path: &'a Path) -> Self {
        Self::Local(path)
    }

    pub fn new(uri: &'a str) -> Self {
        if let Some(i) = uri.find([':', '/']) {
            if uri[i..].starts_with("://") && CLOUD_SCHEME_REGEX.is_match(&uri[..i]) {
                let scheme = CloudScheme::from_str(&uri[..i]).unwrap();
                return Self::Cloud(PlCloudPathRef { scheme, uri });
            }
        }

        Self::from_local_path(Path::new(uri))
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
                let offset_start = p.scheme.as_str().len() + 3;
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

    pub fn extension(&self) -> Option<&str> {
        match self {
            Self::Local(path) => path.extension().and_then(|e| e.to_str()),
            Self::Cloud(_) => {
                let offset_path = self.strip_scheme();
                let separator = '/';

                let mut ext_start = None;
                for (i, c) in offset_path.char_indices() {
                    if c == separator {
                        ext_start = None;
                    }

                    if c == '.' && ext_start.is_none() {
                        ext_start = Some(i);
                    }
                }

                ext_start.map(|i| &offset_path[i + 1..])
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
}

impl PlPath {
    pub fn new(uri: &str) -> Self {
        PlPathRef::new(uri).into_owned()
    }

    pub fn display(&self) -> AddressDisplay<'_> {
        AddressDisplay {
            addr: match self {
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
    fn plpath_join() {
        macro_rules! assert_plpath_join {
            ($base:literal + $added:literal => $result:literal$(, $uri_result:literal)?) => {
                // Normal path test
                let path_base = $base.chars().map(|c| match c {
                    '/' => std::path::MAIN_SEPARATOR,
                    c => c,
                }).collect::<String>();
                let path_added = $added.chars().map(|c| match c {
                    '/' => std::path::MAIN_SEPARATOR,
                    c => c,
                }).collect::<String>();
                let path_result = $result.chars().map(|c| match c {
                    '/' => std::path::MAIN_SEPARATOR,
                    c => c,
                }).collect::<String>();
                assert_eq!(PlPath::new(&path_base).as_ref().join(path_added).to_str(), path_result);

                // URI path test
                let uri_base = format!("file://{}", $base);
                #[allow(unused_variables)]
                let result = {
                    let x = $result;
                    $(let x = $uri_result;)?
                    x
                };
                let uri_result = format!("file://{result}");
                assert_eq!(
                    PlPath::new(uri_base.as_str())
                        .as_ref()
                        .join($added)
                        .to_str(),
                    uri_result.as_str()
                );
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
}
