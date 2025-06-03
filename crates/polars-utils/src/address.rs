use core::fmt;
use std::borrow::Cow;
use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::str::FromStr;

/// A Path or URI
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum Address {
    Local(Box<Path>),
    Cloud(CloudAddress),
}

/// A reference to a Path or URI
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AddressRef<'a> {
    Local(&'a Path),
    Cloud(CloudAddressRef<'a>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct CloudAddress {
    /// The scheme used in cloud e.g. `s3://` or `file://`.
    scheme: CloudScheme,

    /// The path to the specific file or directory. If the scheme is `None`, this is a `Path`
    /// otherwise this is a `/` delimited string.
    path: Box<OsStr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CloudAddressRef<'a> {
    /// The scheme used in cloud e.g. `s3://` or `file://`.
    scheme: CloudScheme,

    /// The path to the specific file or directory. If the scheme is `None`, this is a `Path`
    /// otherwise this is a `/` delimited string.
    path: &'a OsStr,
}

impl<'a> fmt::Display for CloudAddressRef<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { scheme, path } = self;
        write!(f, "{scheme}://{}", path.display())
    }
}

impl fmt::Display for CloudAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl CloudAddress {
    pub fn as_ref(&self) -> CloudAddressRef {
        CloudAddressRef {
            scheme: self.scheme,
            path: self.path.as_ref(),
        }
    }
}

impl CloudAddressRef<'_> {
    pub fn into_owned(self) -> CloudAddress {
        CloudAddress {
            scheme: self.scheme,
            path: self.path.into(),
        }
    }

    pub fn scheme(&self) -> CloudScheme {

        self.scheme
    }

    pub fn path(&self) -> &OsStr {
        self.path
    }
}

pub struct AddressDisplay<'a> {
    addr: AddressRef<'a>,
}

impl<'a> fmt::Display for AddressDisplay<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.addr {
            AddressRef::Local(p) => p.display().fmt(f),
            AddressRef::Cloud(p) => p.fmt(f),
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
    static CLOUD_URL = r"^(s3a?|gs|gcs|file|abfss?|azure|az|adl|https?|hf)://";
}

impl<'a> AddressRef<'a> {
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

    pub fn as_cloud_addr(&self) -> Option<CloudAddressRef> {
        match self {
            Self::Local(_) => None,
            Self::Cloud(p) => Some(*p),
        }
    }

    pub fn join(&self, other: impl AsRef<Path>) -> Address {
        match self {
            Self::Local(p) => Address::Local(p.join(other).into()),
            Self::Cloud(p) => Address::Cloud(CloudAddress {
                scheme: p.scheme,
                path: if std::path::MAIN_SEPARATOR == '\\' {
                    let other = other.as_ref();
                    let mut out = OsString::with_capacity(
                        p.path.len() + usize::from(other.is_relative()) + other.as_os_str().len(),
                    );
                    if other.is_relative() {
                        out.push(OsStr::new("/"));
                    }
                    for c in other.components() {
                        use std::path::Component as C;
                        match c {
                            C::RootDir => {},
                            C::Normal(s) => out.push(&s),
                            C::CurDir | C::ParentDir | C::Prefix(_) => unreachable!(),
                        }
                        out.push(OsStr::new("/"));
                    }
                    out
                } else {
                    Path::new(p.path).join(other).into_os_string()
                }
                .into(),
            }),
        }
    }

    pub fn display(&self) -> AddressDisplay {
        AddressDisplay { addr: *self }
    }

    pub fn from_local_path(path: &'a Path) -> Self {
        Self::Local(path)
    }

    pub fn from_str(uri: &'a str) -> Self {
        if let Some(i) = uri.find(|c| matches!(c, ':' | '/')) {
            if uri.as_bytes().len() >= i + 3 && &uri.as_bytes()[i..i + 3] != b"://" {
                if CLOUD_URL.is_match(&uri[..i]) {
                    let scheme = CloudScheme::from_str(&uri[..i]).unwrap();
                    return Self::Cloud(CloudAddressRef {
                        scheme,
                        path: (&uri[i + 3..]).as_ref(),
                    });
                }
            }
        }

        Self::from_local_path(Path::new(uri))
    }

    pub fn into_owned(self) -> Address {
        match self {
            Self::Local(p) => Address::Local(p.into()),
            Self::Cloud(p) => Address::Cloud(p.into_owned()),
        }
    }

    pub fn offset_path(&self) -> &OsStr {
        match self {
            Self::Local(p) => p.as_os_str(),
            Self::Cloud(p) => p.path,
        }
    }

    pub fn parent(&self) -> Option<Self> {
        Some(match self {
            Self::Local(p) => Self::Local(p.parent()?),
            Self::Cloud(p) => {
                let path = p.path.to_str().unwrap();
                let last_slash = path.char_indices().rev().find(|(_, c)| *c == '/')?.0;
                let path = OsStr::new(&path[..last_slash]);

                Self::Cloud(CloudAddressRef {
                    scheme: p.scheme,
                    path,
                })
            },
        })
    }

    pub fn extension(&self) -> Option<&OsStr> {
        let offset_path = self.offset_path().to_str().unwrap();
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

        ext_start.map(|i| OsStr::new(&offset_path[i + 1..]))
    }

    pub fn to_str(&self) -> Cow<str> {
        match self {
            Self::Local(p) => Cow::Borrowed(p.to_str().unwrap()),
            Self::Cloud(p) => Cow::Owned(p.to_string()),
        }
    }

    pub fn separator(&self) -> char {
        match self {
            Self::Local(path) => std::path::MAIN_SEPARATOR,
            Self::Cloud(_) => '/',
        }
    }
}

impl Address {
    pub fn display(&self) -> AddressDisplay {
        AddressDisplay {
            addr: match self {
                Self::Local(p) => AddressRef::Local(p.as_ref()),
                Self::Cloud(p) => AddressRef::Cloud(p.as_ref()),
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
        AddressRef::from_str(&uri).into_owned()
    }

    pub fn as_ref(&self) -> AddressRef {
        match self {
            Self::Local(p) => AddressRef::Local(p.as_ref()),
            Self::Cloud(p) => AddressRef::Cloud(p.as_ref()),
        }
    }

    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        match self {
            Self::Local(_) => None,
            Self::Cloud(p) => Some(p.scheme),
        }
    }

    pub fn to_str(&self) -> Cow<str> {
        match self {
            Self::Local(p) => Cow::Borrowed(p.to_str().unwrap()),
            Self::Cloud(p) => Cow::Owned(p.to_string()),
        }
    }

    pub fn into_local_path(self) -> Option<Box<Path>> {
        match self {
            Address::Local(path) => Some(path),
            Address::Cloud(_) => None,
        }
    }
}
