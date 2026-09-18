use polars_core::prelude::StringChunked;

#[inline]
pub fn escape_regex_str(s: &str) -> String {
    regex_syntax::escape(s)
}

pub fn escape_regex(ca: &StringChunked) -> StringChunked {
    ca.apply_into_string_amortized(regex_syntax::escape_into)
}
