use polars_core::prelude::arity::unary_elementwise;
use polars_core::prelude::StringChunked;
use unicode_reverse::reverse_grapheme_clusters_in_place;

fn to_reverse_helper(s: Option<&str>) -> Option<String> {
    s.map(|v| {
        let mut text = v.to_string();
        reverse_grapheme_clusters_in_place(&mut text);
        text
    })
}

pub fn reverse(ca: &StringChunked) -> StringChunked {
    unary_elementwise(ca, to_reverse_helper)
}
