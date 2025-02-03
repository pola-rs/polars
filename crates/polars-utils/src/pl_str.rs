#[macro_export]
macro_rules! format_pl_smallstr {
    ($($arg:tt)*) => {{
        use std::fmt::Write;

        let mut string = $crate::pl_str::PlSmallStr::EMPTY;
        write!(string, $($arg)*).unwrap();
        string
    }}
}

type Inner = compact_str::CompactString;

/// String type that inlines small strings.
#[derive(Clone, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(transparent)
)]
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
    pub fn as_mut_str(&mut self) -> &mut str {
        self.0.as_mut_str()
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

// AsRef, Deref and Borrow impls to &str

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

impl core::ops::DerefMut for PlSmallStr {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_str()
    }
}

impl core::borrow::Borrow<str> for PlSmallStr {
    #[inline(always)]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

// AsRef impls for other types

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

// From impls

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

impl From<Inner> for PlSmallStr {
    #[inline(always)]
    fn from(value: Inner) -> Self {
        Self(value)
    }
}

// FromIterator impls

impl FromIterator<PlSmallStr> for PlSmallStr {
    #[inline(always)]
    fn from_iter<T: IntoIterator<Item = PlSmallStr>>(iter: T) -> Self {
        Self(Inner::from_iter(iter.into_iter().map(|x| x.0)))
    }
}

impl<'a> FromIterator<&'a PlSmallStr> for PlSmallStr {
    #[inline(always)]
    fn from_iter<T: IntoIterator<Item = &'a PlSmallStr>>(iter: T) -> Self {
        Self(Inner::from_iter(iter.into_iter().map(|x| x.as_str())))
    }
}

impl FromIterator<char> for PlSmallStr {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = char>>(iter: I) -> PlSmallStr {
        Self(Inner::from_iter(iter))
    }
}

impl<'a> FromIterator<&'a char> for PlSmallStr {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = &'a char>>(iter: I) -> PlSmallStr {
        Self(Inner::from_iter(iter))
    }
}

impl<'a> FromIterator<&'a str> for PlSmallStr {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> PlSmallStr {
        Self(Inner::from_iter(iter))
    }
}

impl FromIterator<String> for PlSmallStr {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> PlSmallStr {
        Self(Inner::from_iter(iter))
    }
}

impl FromIterator<Box<str>> for PlSmallStr {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = Box<str>>>(iter: I) -> PlSmallStr {
        Self(Inner::from_iter(iter))
    }
}

impl<'a> FromIterator<std::borrow::Cow<'a, str>> for PlSmallStr {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = std::borrow::Cow<'a, str>>>(iter: I) -> PlSmallStr {
        Self(Inner::from_iter(iter))
    }
}

// PartialEq impls

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

// Write

impl core::fmt::Write for PlSmallStr {
    #[inline(always)]
    fn write_char(&mut self, c: char) -> std::fmt::Result {
        self.0.write_char(c)
    }

    #[inline(always)]
    fn write_fmt(&mut self, args: std::fmt::Arguments<'_>) -> std::fmt::Result {
        self.0.write_fmt(args)
    }

    #[inline(always)]
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write_str(s)
    }
}

// Debug, Display

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
