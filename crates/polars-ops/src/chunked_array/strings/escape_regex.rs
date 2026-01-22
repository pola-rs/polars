use std::borrow::Cow;

use polars_core::prelude::{StringChunked, StringChunkedBuilder};

#[inline]
pub fn escape_regex_str(s: &str) -> Cow<'_, str> {
    if s.contains(|c: char| regex_syntax::is_meta_character(c)) {
        Cow::Owned(regex_syntax::escape(s))
    } else {
        Cow::Borrowed(s)
    }
}

/// An example of regex escaping
/// ```rust
/// # use polars_core::prelude::*;
/// # use polars_ops::chunked_array::strings::escape_regex;
/// let ca = StringChunked::new(PlSmallStr::from_static("s"), &["hello", ".*"]);
/// let out = escape_regex(&ca);
/// assert_eq!(out.get(1), Some("\\.\\*"));
/// ```
pub fn escape_regex(ca: &StringChunked) -> StringChunked {
    let mut builder = StringChunkedBuilder::new(ca.name().clone(), ca.len());
    let mut buffer = String::new();

    for opt_s in ca.iter() {
        match opt_s {
            Some(s) => {
                // Check metacharacters to avoid unnecessary buffer operations
                if s.contains(regex_syntax::is_meta_character) {
                    buffer.clear();
                    regex_syntax::escape_into(s, &mut buffer);
                    builder.append_value(&buffer);
                } else {
                    builder.append_value(s);
                }
            },
            None => builder.append_null(),
        }
    }
    builder.finish()
}
