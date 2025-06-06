use core::fmt;
use std::path::Path;
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
    pub fn as_ref(&self) -> PlCloudPathRef {
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

    pub fn as_cloud_addr(&self) -> Option<PlCloudPathRef> {
        match self {
            Self::Local(_) => None,
            Self::Cloud(p) => Some(*p),
        }
    }

    pub fn join(&self, other: impl AsRef<Path>) -> PlPath {
        match self {
            Self::Local(p) => PlPath::Local(p.join(other).into()),
            Self::Cloud(p) => {
                let other = other.as_ref();
                let mut out = String::with_capacity(
                    p.uri.len() + usize::from(other.is_relative()) + other.as_os_str().len(),
                );
                out.push_str(p.uri);
                if other.is_relative() {
                    out.push('/');
                }
                for c in other.components() {
                    use std::path::Component as C;
                    match c {
                        C::RootDir => {},
                        C::Normal(s) => out.push_str(s.to_str().unwrap()),
                        C::CurDir | C::ParentDir | C::Prefix(_) => unreachable!(),
                    }
                    out.push('/');
                }
                let uri = out.into();
                PlPath::Cloud(PlCloudPath {
                    scheme: p.scheme,
                    uri,
                })
            },
        }
    }

    pub fn display(&self) -> AddressDisplay {
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
        let offset_path = self.strip_scheme();
        let separator = match self {
            Self::Local(_) => std::path::MAIN_SEPARATOR,
            Self::Cloud(_) => '/',
        };

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
    }

    pub fn to_str(&self) -> &'a str {
        match self {
            Self::Local(p) => p.to_str().unwrap(),
            Self::Cloud(p) => p.uri,
        }
    }

    pub fn separator(&self) -> char {
        match self {
            Self::Local(_) => std::path::MAIN_SEPARATOR,
            Self::Cloud(_) => '/',
        }
    }
}

impl PlPath {
    pub fn new(uri: &str) -> Self {
        PlPathRef::new(uri).into_owned()
    }

    pub fn display(&self) -> AddressDisplay {
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

    pub fn from_string(uri: String) -> Self {
        Self::new(&uri)
    }

    pub fn as_ref(&self) -> PlPathRef {
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
