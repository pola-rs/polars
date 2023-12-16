use polars_core::prelude::Utf8Chunked;
use unicode_reverse::reverse_grapheme_clusters_in_place;

fn to_reverse_helper(s: Option<&str>) -> Option<String> {
    s.map(|v| {
        let mut text = v.to_string();
        reverse_grapheme_clusters_in_place(&mut text);
        text
    })
}

pub fn reverse(ca: &Utf8Chunked) -> Utf8Chunked {
    ca.apply_generic(to_reverse_helper)
}
