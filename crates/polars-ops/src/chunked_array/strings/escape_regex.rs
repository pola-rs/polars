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
    // When we use StringChunkedBuilder, it will still copy the data into its buffer.
    // But we can avoid unnecessary formation using regex_syntax.
    let mut builder = StringChunkedBuilder::new(ca.name().clone(), ca.len());

    for opt_s in ca.iter() {
        match opt_s {
            Some(s) => {
                let escaped = escape_regex_str(s);
                builder.append_value(&escaped);
            },
            None => builder.append_null(),
        }
    }
    builder.finish()
}
