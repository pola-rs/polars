use polars_core::prelude::Utf8Chunked;

fn to_reverse_helper(s: Option<&str>) -> Option<String> {
    s.map(|v| v.chars().rev().collect::<String>())
}

pub fn reverse(ca: &Utf8Chunked) -> Utf8Chunked {
    ca.apply_generic(to_reverse_helper)
}
