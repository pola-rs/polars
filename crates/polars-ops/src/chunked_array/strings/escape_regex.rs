use polars_core::prelude::StringChunked;

#[inline]
pub fn escape_regex_str(s: &str) -> String {
    regex_syntax::escape(s)
}

pub fn escape_regex(ca: &StringChunked) -> StringChunked {
    // Through the shared apply, which asks how each chunk stores its values once rather than
    // once per element: a chunk that repeats one string is escaped by a single call.
    ca.apply_into_string_amortized(regex_syntax::escape_into)
}
