use polars_core::prelude::arity::binary_elementwise;

use super::*;

fn strip_chars_binary<'a>(opt_s: Option<&'a str>, opt_pat: Option<&str>) -> Option<&'a str> {
    match (opt_s, opt_pat) {
        (Some(s), Some(pat)) => {
            if pat.chars().count() == 1 {
                Some(s.trim_matches(pat.chars().next().unwrap()))
            } else {
                Some(s.trim_matches(|c| pat.contains(c)))
            }
        },
        (Some(s), _) => Some(s.trim()),
        _ => None,
    }
}

fn strip_chars_start_binary<'a>(opt_s: Option<&'a str>, opt_pat: Option<&str>) -> Option<&'a str> {
    match (opt_s, opt_pat) {
        (Some(s), Some(pat)) => {
            if pat.chars().count() == 1 {
                Some(s.trim_start_matches(pat.chars().next().unwrap()))
            } else {
                Some(s.trim_start_matches(|c| pat.contains(c)))
            }
        },
        (Some(s), _) => Some(s.trim_start()),
        _ => None,
    }
}

fn strip_chars_end_binary<'a>(opt_s: Option<&'a str>, opt_pat: Option<&str>) -> Option<&'a str> {
    match (opt_s, opt_pat) {
        (Some(s), Some(pat)) => {
            if pat.chars().count() == 1 {
                Some(s.trim_end_matches(pat.chars().next().unwrap()))
            } else {
                Some(s.trim_end_matches(|c| pat.contains(c)))
            }
        },
        (Some(s), _) => Some(s.trim_end()),
        _ => None,
    }
}

fn strip_prefix_binary<'a>(s: Option<&'a str>, prefix: Option<&str>) -> Option<&'a str> {
    Some(s?.strip_prefix(prefix?).unwrap_or(s?))
}

fn strip_suffix_binary<'a>(s: Option<&'a str>, suffix: Option<&str>) -> Option<&'a str> {
    Some(s?.strip_suffix(suffix?).unwrap_or(s?))
}

pub fn strip_chars(ca: &Utf8Chunked, pat: &Utf8Chunked) -> Utf8Chunked {
    match pat.len() {
        1 => {
            if let Some(pat) = pat.get(0) {
                if pat.chars().count() == 1 {
                    // Fast path for when a single character is passed
                    ca.apply_generic(|opt_s| {
                        opt_s.map(|s| s.trim_matches(pat.chars().next().unwrap()))
                    })
                } else {
                    ca.apply_generic(|opt_s| opt_s.map(|s| s.trim_matches(|c| pat.contains(c))))
                }
            } else {
                ca.apply_generic(|opt_s| opt_s.map(|s| s.trim()))
            }
        },
        _ => binary_elementwise(ca, pat, strip_chars_binary),
    }
}

pub fn strip_chars_start(ca: &Utf8Chunked, pat: &Utf8Chunked) -> Utf8Chunked {
    match pat.len() {
        1 => {
            if let Some(pat) = pat.get(0) {
                if pat.chars().count() == 1 {
                    // Fast path for when a single character is passed
                    ca.apply_generic(|opt_s| {
                        opt_s.map(|s| s.trim_start_matches(pat.chars().next().unwrap()))
                    })
                } else {
                    ca.apply_generic(|opt_s| {
                        opt_s.map(|s| s.trim_start_matches(|c| pat.contains(c)))
                    })
                }
            } else {
                ca.apply_generic(|opt_s| opt_s.map(|s| s.trim_start()))
            }
        },
        _ => binary_elementwise(ca, pat, strip_chars_start_binary),
    }
}

pub fn strip_chars_end(ca: &Utf8Chunked, pat: &Utf8Chunked) -> Utf8Chunked {
    match pat.len() {
        1 => {
            if let Some(pat) = pat.get(0) {
                if pat.chars().count() == 1 {
                    // Fast path for when a single character is passed
                    ca.apply_generic(|opt_s| {
                        opt_s.map(|s| s.trim_end_matches(pat.chars().next().unwrap()))
                    })
                } else {
                    ca.apply_generic(|opt_s| opt_s.map(|s| s.trim_end_matches(|c| pat.contains(c))))
                }
            } else {
                ca.apply_generic(|opt_s| opt_s.map(|s| s.trim_end()))
            }
        },
        _ => binary_elementwise(ca, pat, strip_chars_end_binary),
    }
}

pub fn strip_prefix(ca: &Utf8Chunked, prefix: &Utf8Chunked) -> Utf8Chunked {
    match prefix.len() {
        1 => match prefix.get(0) {
            Some(prefix) => {
                ca.apply_generic(|opt_s| opt_s.map(|s| s.strip_prefix(prefix).unwrap_or(s)))
            },
            _ => Utf8Chunked::full_null(ca.name(), ca.len()),
        },
        _ => binary_elementwise(ca, prefix, strip_prefix_binary),
    }
}

pub fn strip_suffix(ca: &Utf8Chunked, suffix: &Utf8Chunked) -> Utf8Chunked {
    match suffix.len() {
        1 => match suffix.get(0) {
            Some(suffix) => {
                ca.apply_generic(|opt_s| opt_s.map(|s| s.strip_suffix(suffix).unwrap_or(s)))
            },
            _ => Utf8Chunked::full_null(ca.name(), ca.len()),
        },
        _ => binary_elementwise(ca, suffix, strip_suffix_binary),
    }
}
