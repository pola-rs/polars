use std::borrow::{Borrow, Cow};
use std::ffi::OsStr;
use std::fmt::Display;
use std::ops::{Deref, Range};
use std::path::{Path, PathBuf};

use polars_error::{PolarsResult, polars_err};

use crate::format_pl_refstr;
use crate::pl_str::PlRefStr;

/// Windows paths can be prefixed with this.
/// <https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry>
pub const WINDOWS_EXTPATH_PREFIX: &str = r#"\\?\"#;

/// Path represented as a UTF-8 string.
///
/// Equality and ordering are based on the string value, which can be sensitive to duplicate
/// separators. `as_std_path()` can be used to return a `&std::path::Path` for comparisons / API
/// access.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct PlPath {
    inner: str,
}

#[derive(Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
/// Reference-counted [`PlPath`].
///
/// # Windows paths invariant
/// Windows paths will have leading `\\?\` prefix stripped, and all backslashes normalized to
/// forward slashes.
pub struct PlRefPath {
    inner: PlRefStr,
}

impl PlPath {
    // Note: Do not expose the following constructors, they do not normalize paths.
    fn _new<S: AsRef<str> + ?Sized>(s: &S) -> &PlPath {
        let s: &str = s.as_ref();
        // Safety: `PlPath` is `repr(transparent)` on `str`.
        unsafe { &*(s as *const str as *const PlPath) }
    }

    fn _try_from_path(path: &Path) -> PolarsResult<&PlPath> {
        path.to_str()
            .ok_or_else(|| polars_err!(non_utf8_path))
            .map(Self::_new)
    }

    pub fn as_str(&self) -> &str {
        unsafe { &*(self as *const PlPath as *const str) }
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.as_str().as_bytes()
    }

    pub fn as_os_str(&self) -> &OsStr {
        OsStr::new(self)
    }

    pub fn as_std_path(&self) -> &Path {
        Path::new(self)
    }

    pub fn to_ref_path(&self) -> PlRefPath {
        PlRefPath::_new_no_normalize(self.as_str().into())
    }

    pub fn scheme(&self) -> Option<CloudScheme> {
        CloudScheme::from_path(self.as_str())
    }

    /// Shorthand for `self.scheme().is_some()`.
    pub fn has_scheme(&self) -> bool {
        self.scheme().is_some()
    }

    /// Return a string with the scheme prefix removed (if any).
    pub fn strip_scheme(&self) -> &str {
        &self.as_str()[self.scheme().map_or(0, |x| x.strip_scheme_index())..self.inner.len()]
    }

    pub fn file_name(&self) -> Option<&OsStr> {
        Path::new(self.strip_scheme()).file_name()
    }

    pub fn extension(&self) -> Option<&str> {
        Path::new(self.strip_scheme())
            .extension()
            .map(|x| x.to_str().unwrap())
    }

    pub fn parent(&self) -> Option<&str> {
        Path::new(self.strip_scheme())
            .parent()
            .map(|x| x.to_str().unwrap())
    }

    /// Slices the path.
    pub fn sliced(&self, range: Range<usize>) -> &PlPath {
        Self::_new(&self.as_str()[range])
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
            None | Some(CloudScheme::FileNoHostname) => Some(("", self.strip_scheme())),
            Some(scheme) => {
                let path_str = self.as_str();
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

    /// Returns 0 if `self.scheme()` is `None`. Otherwise, returns `i` such that
    /// `&self.to_str()[..i]` trims to the authority.
    /// * If there is no '/', separator found, `i` will simply be the length of the string.
    ///   * This is except if the scheme is `FileNoHostname`, where instead `i` will be "file:".len()
    /// * If `self` has no `CloudScheme`, returns 0
    pub fn authority_end_position(&self) -> usize {
        match self.scheme() {
            None => 0,
            Some(scheme @ CloudScheme::FileNoHostname) => scheme.strip_scheme_index(),
            Some(_) => {
                let after_scheme = self.strip_scheme();
                let offset = self.as_str().len() - after_scheme.len();

                offset + after_scheme.find('/').unwrap_or(after_scheme.len())
            },
        }
    }

    pub fn to_absolute_path(&self) -> PolarsResult<PlRefPath> {
        PlRefPath::try_from_pathbuf(std::path::absolute(Path::new(self.strip_scheme()))?)
    }

    pub fn join(&self, other: impl AsRef<str>) -> PlRefPath {
        let other = other.as_ref();

        if CloudScheme::from_path(other).is_some() {
            PlRefPath::new(other)
        } else {
            PlRefPath::try_from_pathbuf(self.as_std_path().join(other)).unwrap()
        }
    }

    /// Converts backslashes to forward-slashes, and removes `\\?\` prefix.
    pub fn normalize_windows_path(path_str: &str) -> Option<PlRefPath> {
        let has_extpath_prefix = path_str.starts_with(WINDOWS_EXTPATH_PREFIX);

        if has_extpath_prefix || cfg!(target_family = "windows") {
            let path_str = path_str
                .strip_prefix(WINDOWS_EXTPATH_PREFIX)
                .unwrap_or(path_str);

            if matches!(
                CloudScheme::from_path(path_str),
                None | Some(CloudScheme::File | CloudScheme::FileNoHostname)
            ) && path_str.contains('\\')
            {
                let new_path = path_str.replace('\\', "/");
                let inner = PlRefStr::from_string(new_path);
                return Some(PlRefPath { inner });
            }
        }

        None
    }
}

impl AsRef<str> for PlPath {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<OsStr> for PlPath {
    fn as_ref(&self) -> &OsStr {
        OsStr::new(self.as_str())
    }
}

impl AsRef<Path> for PlPath {
    fn as_ref(&self) -> &Path {
        self.as_std_path()
    }
}

impl From<&PlPath> for Box<PlPath> {
    fn from(value: &PlPath) -> Self {
        let s: &str = value.as_str();
        let s: Box<str> = s.into();
        // Safety: `PlPath` is `repr(transparent)` on `str`.
        let out: Box<PlPath> = unsafe { std::mem::transmute(s) };
        out
    }
}

impl Display for PlPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.as_str(), f)
    }
}

impl PlRefPath {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Normalizes Windows paths.
    pub fn new(path: impl AsRef<str> + Into<PlRefStr>) -> Self {
        if let Some(path) = PlPath::normalize_windows_path(path.as_ref()) {
            return path;
        }

        Self::_new_no_normalize(path.into())
    }

    const fn _new_no_normalize(path: PlRefStr) -> Self {
        Self { inner: path }
    }

    pub fn try_from_path(path: &Path) -> PolarsResult<PlRefPath> {
        Ok(Self::new(PlPath::_try_from_path(path)?.as_str()))
    }

    pub fn try_from_pathbuf(path: PathBuf) -> PolarsResult<PlRefPath> {
        Self::try_from_path(&path)
    }

    pub fn as_str(&self) -> &str {
        &self.inner
    }

    pub fn as_ref_str(&self) -> &PlRefStr {
        &self.inner
    }

    pub fn into_ref_str(self) -> PlRefStr {
        self.inner
    }

    /// Slices the path.
    pub fn sliced(&self, range: Range<usize>) -> PlRefPath {
        if range == (0..self.as_str().len()) {
            self.clone()
        } else {
            Self::_new_no_normalize(PlPath::sliced(self, range).as_str().into())
        }
    }

    /// # Returns
    /// Returns an absolute local path if this path ref is a relative local path, otherwise returns None.
    pub fn to_absolute_path(&self) -> PolarsResult<Cow<'_, PlRefPath>> {
        Ok(if self.has_scheme() || self.as_std_path().is_absolute() {
            Cow::Borrowed(self)
        } else {
            Cow::Owned(PlPath::to_absolute_path(self)?)
        })
    }

    /// Checks if references point to the same allocation.
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        PlRefStr::ptr_eq(this.as_ref_str(), other.as_ref_str())
    }
}

impl AsRef<str> for PlRefPath {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<OsStr> for PlRefPath {
    fn as_ref(&self) -> &OsStr {
        self.as_os_str()
    }
}

impl AsRef<Path> for PlRefPath {
    fn as_ref(&self) -> &Path {
        self.as_std_path()
    }
}

impl Deref for PlRefPath {
    type Target = PlPath;

    fn deref(&self) -> &Self::Target {
        PlPath::_new(self)
    }
}

impl Display for PlRefPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.as_str(), f)
    }
}

impl ToOwned for PlPath {
    type Owned = PlRefPath;

    fn to_owned(&self) -> Self::Owned {
        self.to_ref_path()
    }
}

impl Borrow<PlPath> for PlRefPath {
    fn borrow(&self) -> &PlPath {
        self
    }
}

impl From<&str> for PlRefPath {
    fn from(value: &str) -> Self {
        Self::new(value)
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
            /// Note, private function. Users should use [`CloudScheme::from_path`], that will handle e.g.
            /// `file:/` without hostname properly.
            #[expect(unreachable_patterns)]
            fn from_scheme_str(s: &str) -> Option<Self> {
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
    pub fn from_path(path: &str) -> Option<Self> {
        if let Some(stripped) = path.strip_prefix("file:") {
            return Some(if stripped.starts_with("//") {
                Self::File
            } else {
                Self::FileNoHostname
            });
        }

        Self::from_scheme_str(&path[..path.find("://")?])
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

impl Display for CloudScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.as_str(), f)
    }
}

/// Formats a local path to begin with `file:///`.
///
/// # Panics
/// May panic if `absolute_local_path` is not an absolute local path.
pub fn format_file_uri(absolute_local_path: &str) -> PlRefPath {
    // Windows needs an extra slash, i.e.:
    //
    // # Windows
    // Absolute path: "C:/Windows/system32"
    // Formatted: "file:///C:/Windows/system32"
    //
    // # Unix
    // Absolute path: "/root/.vimrc"
    // Formatted: "file:///root/.vimrc"
    if cfg!(target_family = "windows") || absolute_local_path.starts_with(WINDOWS_EXTPATH_PREFIX) {
        if let Some(path) = PlPath::normalize_windows_path(absolute_local_path) {
            PlRefPath::new(format_pl_refstr!("file:///{path}"))
        } else {
            PlRefPath::new(format_pl_refstr!("file:///{absolute_local_path}"))
        }
    } else {
        PlRefPath::new(format_pl_refstr!("file://{absolute_local_path}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plpath_file() {
        let p = PlRefPath::new("file:///home/user");
        assert_eq!(
            (
                p.scheme(),
                p.scheme().map(|x| x.as_str()),
                p.as_str(),
                p.strip_scheme(),
            ),
            (
                Some(CloudScheme::File),
                Some("file"),
                "file:///home/user",
                "/home/user"
            )
        );

        let p = PlRefPath::new("file:/home/user");
        assert_eq!(
            (
                p.scheme(),
                p.scheme().map(|x| x.as_str()),
                p.as_str(),
                p.strip_scheme(),
            ),
            (
                Some(CloudScheme::FileNoHostname),
                Some("file"),
                "file:/home/user",
                "/home/user"
            )
        );

        assert_eq!(PlRefPath::new("file://").scheme(), Some(CloudScheme::File));

        assert_eq!(
            PlRefPath::new("file://").strip_scheme_split_authority(),
            None
        );

        assert_eq!(
            PlRefPath::new("file:///").strip_scheme_split_authority(),
            Some(("", "/"))
        );

        assert_eq!(
            PlRefPath::new("file:///path").strip_scheme_split_authority(),
            Some(("", "/path"))
        );

        assert_eq!(
            PlRefPath::new("file://hostname:80/path").strip_scheme_split_authority(),
            Some(("hostname:80", "/path"))
        );

        assert_eq!(
            PlRefPath::new("file:").scheme(),
            Some(CloudScheme::FileNoHostname)
        );
        assert_eq!(
            PlRefPath::new("file:/").scheme(),
            Some(CloudScheme::FileNoHostname)
        );
        assert_eq!(
            PlRefPath::new("file:").strip_scheme_split_authority(),
            Some(("", ""))
        );
        assert_eq!(
            PlRefPath::new("file:/Local/path").strip_scheme_split_authority(),
            Some(("", "/Local/path"))
        );

        assert_eq!(
            PlRefPath::new(r#"\\?\C:\Windows\system32"#).as_str(),
            "C:/Windows/system32"
        );
    }

    #[test]
    fn test_plpath_join() {
        assert_eq!(
            PlRefPath::new("s3://.../...").join("az://.../...").as_str(),
            "az://.../..."
        );

        fn _assert_plpath_join(base: &str, added: &str, expect: &str) {
            // Normal path test
            let expect = PlRefPath::new(expect);
            let base = base.replace('/', std::path::MAIN_SEPARATOR_STR);
            let added = added.replace('/', std::path::MAIN_SEPARATOR_STR);

            assert_eq!(PlRefPath::new(&base).join(&added), expect);

            // URI path test
            let uri_base = format_file_uri(&base);
            let expect_uri = if added.starts_with(std::path::MAIN_SEPARATOR_STR) {
                expect.clone()
            } else {
                format_file_uri(expect.as_str())
            };

            assert_eq!(PlRefPath::new(uri_base.as_str()).join(added), expect_uri);
        }

        macro_rules! assert_plpath_join {
            ($base:literal + $added:literal => $expect:literal) => {
                _assert_plpath_join($base, $added, $expect)
            };
        }

        assert_plpath_join!("a/b/c/" + "d/e" => "a/b/c/d/e");
        assert_plpath_join!("a/b/c" + "d/e" => "a/b/c/d/e");
        assert_plpath_join!("a/b/c" + "d/e/" => "a/b/c/d/e/");
        assert_plpath_join!("a/b/c" + "/d" => "/d");
        assert_plpath_join!("a/b/c" + "/d/" => "/d/");
        assert_plpath_join!("" + "/d/" => "/d/");
        assert_plpath_join!("/" + "/d/" => "/d/");
        assert_plpath_join!("/x/y" + "/d/" => "/d/");
        assert_plpath_join!("/x/y" + "/d" => "/d");
        assert_plpath_join!("/x/y" + "d" => "/x/y/d");

        assert_plpath_join!("/a/longer" + "path" => "/a/longer/path");
        assert_plpath_join!("/a/longer" + "/path" => "/path");
        assert_plpath_join!("/a/longer" + "path/test" => "/a/longer/path/test");
        assert_plpath_join!("/a/longer" + "/path/test" => "/path/test");
    }

    #[test]
    fn test_plpath_name() {
        assert_eq!(PlRefPath::new("s3://...").file_name(), Some("...".as_ref()));
        assert_eq!(
            PlRefPath::new("a/b/file.parquet").file_name(),
            Some("file.parquet".as_ref())
        );
        assert_eq!(
            PlRefPath::new("file.parquet").file_name(),
            Some("file.parquet".as_ref())
        );

        assert_eq!(PlRefPath::new("s3://").file_name(), None);
        assert_eq!(PlRefPath::new("").file_name(), None);
    }
}
