use std::borrow::{Borrow, Cow};
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Deref;
use std::sync::{Arc, LazyLock};

#[macro_export]
macro_rules! format_pl_refstr {
    ($($arg:tt)*) => {{
        use std::fmt::Write;

        let mut string = String::new();
        write!(string, $($arg)*).unwrap();
        $crate::pl_str::PlRefStr::from_string(string)
    }}
}

type Inner = Arc<str>;

/// Reference-counted string.
#[derive(Clone, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlRefStr(Inner);

impl PlRefStr {
    pub fn empty() -> Self {
        return EMPTY.clone();
        static EMPTY: LazyLock<PlRefStr> = LazyLock::new(|| PlRefStr::from_str(""));
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
    pub fn from_arc_str(arc: Arc<str>) -> Self {
        Self(arc)
    }

    #[inline(always)]
    pub fn as_str(&self) -> &str {
        self.0.as_ref()
    }

    #[inline(always)]
    #[allow(clippy::inherent_to_string_shadow_display)] // This is faster.
    pub fn to_string(&self) -> String {
        self.as_str().to_string()
    }

    #[inline(always)]
    pub fn into_string(self) -> String {
        self.as_str().to_string()
    }

    #[inline(always)]
    pub fn into_arc_str(self) -> Arc<str> {
        self.0
    }

    /// Checks if references point to the same allocation.
    #[inline(always)]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        Inner::ptr_eq(&this.0, &other.0)
    }

    #[inline(always)]
    pub fn get_mut(&mut self) -> Option<&mut str> {
        Inner::get_mut(&mut self.0)
    }

    #[inline(always)]
    pub fn make_mut(&mut self) -> &mut str {
        Inner::make_mut(&mut self.0)
    }
}

impl Default for PlRefStr {
    #[inline(always)]
    fn default() -> Self {
        Self::empty()
    }
}

// AsRef, Deref and Borrow impls to &str

impl AsRef<str> for PlRefStr {
    #[inline(always)]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Deref for PlRefStr {
    type Target = str;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl Borrow<str> for PlRefStr {
    #[inline(always)]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

// AsRef impls for other types

impl AsRef<std::path::Path> for PlRefStr {
    #[inline(always)]
    fn as_ref(&self) -> &std::path::Path {
        self.as_str().as_ref()
    }
}

impl AsRef<[u8]> for PlRefStr {
    #[inline(always)]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl AsRef<std::ffi::OsStr> for PlRefStr {
    #[inline(always)]
    fn as_ref(&self) -> &std::ffi::OsStr {
        self.as_str().as_ref()
    }
}

// From impls

impl From<&str> for PlRefStr {
    #[inline(always)]
    fn from(value: &str) -> Self {
        Self::from_str(value)
    }
}

impl From<String> for PlRefStr {
    #[inline(always)]
    fn from(value: String) -> Self {
        Self::from_string(value)
    }
}

impl From<PlRefStr> for String {
    #[inline(always)]
    fn from(value: PlRefStr) -> Self {
        value.into_string()
    }
}

impl From<Cow<'_, str>> for PlRefStr {
    #[inline(always)]
    fn from(value: Cow<str>) -> Self {
        match value {
            Cow::Owned(s) => Self::from_string(s),
            Cow::Borrowed(s) => Self::from_str(s),
        }
    }
}

impl From<&String> for PlRefStr {
    #[inline(always)]
    fn from(value: &String) -> Self {
        Self::from_str(value.as_str())
    }
}

impl From<Arc<str>> for PlRefStr {
    #[inline(always)]
    fn from(value: Arc<str>) -> Self {
        Self::from_arc_str(value)
    }
}

// FromIterator impls
impl FromIterator<char> for PlRefStr {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = char>>(iter: I) -> PlRefStr {
        Self::from_string(String::from_iter(iter))
    }
}

impl<'a> FromIterator<&'a char> for PlRefStr {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = &'a char>>(iter: I) -> PlRefStr {
        Self::from_string(String::from_iter(iter))
    }
}

impl<'a> FromIterator<&'a str> for PlRefStr {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> PlRefStr {
        Self::from_string(String::from_iter(iter))
    }
}

impl FromIterator<String> for PlRefStr {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> PlRefStr {
        Self::from_string(String::from_iter(iter))
    }
}

impl FromIterator<Box<str>> for PlRefStr {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = Box<str>>>(iter: I) -> PlRefStr {
        Self::from_string(String::from_iter(iter))
    }
}

impl<'a> FromIterator<std::borrow::Cow<'a, str>> for PlRefStr {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = std::borrow::Cow<'a, str>>>(iter: I) -> PlRefStr {
        Self::from_string(String::from_iter(iter))
    }
}

impl<T> PartialEq<T> for PlRefStr
where
    T: AsRef<str> + ?Sized,
{
    #[inline(always)]
    fn eq(&self, other: &T) -> bool {
        self.as_str() == other.as_ref()
    }
}

impl PartialEq<PlRefStr> for &str {
    #[inline(always)]
    fn eq(&self, other: &PlRefStr) -> bool {
        *self == other.as_str()
    }
}

impl PartialEq<PlRefStr> for String {
    #[inline(always)]
    fn eq(&self, other: &PlRefStr) -> bool {
        self.as_str() == other.as_str()
    }
}

// Debug, Display

impl Debug for PlRefStr {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.as_str(), f)
    }
}

impl core::fmt::Display for PlRefStr {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        core::fmt::Display::fmt(self.as_str(), f)
    }
}

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for PlRefStr {
    fn inline_schema() -> bool {
        str::inline_schema()
    }
    fn schema_name() -> std::borrow::Cow<'static, str> {
        str::schema_name()
    }
    fn schema_id() -> std::borrow::Cow<'static, str> {
        str::schema_id()
    }
    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        str::json_schema(generator)
    }
}
