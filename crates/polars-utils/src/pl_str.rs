#[macro_export]
macro_rules! format_pl_smallstr {
    ($($arg:tt)*) => {{
        use std::fmt::Write;

        let mut string = String::new();
        write!(string, $($arg)*).unwrap();
        PlSmallStr::from_string(string)
    }}
}

type Inner = compact_str::CompactString;

/// String type that inlines small strings.
#[derive(Clone, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlSmallStr(Inner);

impl PlSmallStr {
    pub const EMPTY: Self = Self::from_static("");
    pub const EMPTY_REF: &'static Self = &Self::from_static("");

    #[inline(always)]
    pub const fn from_static(s: &'static str) -> Self {
        Self(Inner::const_new(s))
    }

    #[inline(always)]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        Self(Inner::from(s))
    }

    #[inline(always)]
    pub fn from_string(s: String) -> Self {
        Self(Inner::from(s))
    }

    #[inline(always)]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    #[inline(always)]
    pub fn into_string(self) -> String {
        self.0.into_string()
    }
}

impl Default for PlSmallStr {
    #[inline(always)]
    fn default() -> Self {
        Self::EMPTY
    }
}

/// AsRef, Deref and Borrow impls to &str

impl AsRef<str> for PlSmallStr {
    #[inline(always)]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl core::ops::Deref for PlSmallStr {
    type Target = str;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl core::borrow::Borrow<str> for PlSmallStr {
    #[inline(always)]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

/// AsRef impls for other types

impl AsRef<std::path::Path> for PlSmallStr {
    #[inline(always)]
    fn as_ref(&self) -> &std::path::Path {
        self.as_str().as_ref()
    }
}

impl AsRef<[u8]> for PlSmallStr {
    #[inline(always)]
    fn as_ref(&self) -> &[u8] {
        self.as_str().as_bytes()
    }
}

impl AsRef<std::ffi::OsStr> for PlSmallStr {
    #[inline(always)]
    fn as_ref(&self) -> &std::ffi::OsStr {
        self.as_str().as_ref()
    }
}

/// From impls

impl From<&str> for PlSmallStr {
    #[inline(always)]
    fn from(value: &str) -> Self {
        Self::from_str(value)
    }
}

impl From<String> for PlSmallStr {
    #[inline(always)]
    fn from(value: String) -> Self {
        Self::from_string(value)
    }
}

impl From<&String> for PlSmallStr {
    #[inline(always)]
    fn from(value: &String) -> Self {
        Self::from_str(value.as_str())
    }
}

/// FromIterator impls (TODO)

/// PartialEq impls

impl<T> PartialEq<T> for PlSmallStr
where
    T: AsRef<str> + ?Sized,
{
    #[inline(always)]
    fn eq(&self, other: &T) -> bool {
        self.as_str() == other.as_ref()
    }
}

impl PartialEq<PlSmallStr> for &str {
    #[inline(always)]
    fn eq(&self, other: &PlSmallStr) -> bool {
        *self == other.as_str()
    }
}

impl PartialEq<PlSmallStr> for String {
    #[inline(always)]
    fn eq(&self, other: &PlSmallStr) -> bool {
        self.as_str() == other.as_str()
    }
}

/// Debug, Display

impl core::fmt::Debug for PlSmallStr {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}

impl core::fmt::Display for PlSmallStr {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}
