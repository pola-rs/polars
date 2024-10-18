use polars_core::prelude::arity::unary_elementwise;
use polars_core::prelude::StringChunked;
use regex::escape;

#[inline]
pub fn escape_regex_str(s: &str) -> String {
    escape(s)
}

fn escape_regex_helper(s: Option<&str>) -> Option<String> {
    s.map(escape_regex_str)
}

pub fn escape_regex(ca: &StringChunked) -> StringChunked {
    unary_elementwise(ca, escape_regex_helper)
}
