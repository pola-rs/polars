pub mod compression;
mod other;

pub use other::*;
#[cfg(any(feature = "async", feature = "cloud"))]
pub mod byte_source;
pub mod file;
pub mod mkdir;
pub mod slice;
pub mod sync_on_close;

/// Excludes only the unreserved URI characters in RFC-3986:
///
/// <https://datatracker.ietf.org/doc/html/rfc3986#section-2.3>
///
/// Characters that are allowed in a URI but do not have a reserved
/// purpose are called unreserved.  These include uppercase and lowercase
/// letters, decimal digits, hyphen, period, underscore, and tilde.
///
/// unreserved  = ALPHA / DIGIT / "-" / "." / "_" / "~"
pub const URL_ENCODE_CHARSET: &percent_encoding::AsciiSet = &percent_encoding::NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~');

/// Characters to percent-encode for hive values such that they round-trip from bucket storage.
///
/// This is much more relaxed than the RFC-3986 URI spec as bucket storage is more permissive of allowed
/// characters.
pub const HIVE_VALUE_ENCODE_CHARSET: &percent_encoding::AsciiSet = &percent_encoding::CONTROLS
    .add(b'/') // Exclude path separator
    .add(b'=') // Exclude hive `key=value` separator
    .add(b'%') // Percent itself.
    // Colon and space are supported by object storage, but are encoded to mimic
    // the datetime output format from pyarrow:
    // * i.e. 'date2=2023-01-01 00:00:00.000000' becomes 'date2=2023-01-01%2000%3A00%3A00.000000'
    .add(b':')
    .add(b' ');
