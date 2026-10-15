use polars_core::prelude::StringChunked;
use unicode_reverse::reverse_grapheme_clusters_in_place;

pub fn reverse(ca: &StringChunked) -> StringChunked {
    ca.apply_into_string_amortized(|s, buf| {
        buf.push_str(s);
        reverse_grapheme_clusters_in_place(buf);
    })
}
