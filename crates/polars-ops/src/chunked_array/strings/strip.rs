use polars_core::prelude::arity::{broadcast_binary_elementwise, unary_elementwise};

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

pub fn strip_chars(ca: &StringChunked, pat: &StringChunked) -> StringChunked {
    match pat.len() {
        1 => {
            if let Some(pat) = pat.get(0) {
                if pat.chars().count() == 1 {
                    // Fast path for when a single character is passed
                    unary_elementwise(ca, |opt_s| {
                        opt_s.map(|s| s.trim_matches(pat.chars().next().unwrap()))
                    })
                } else {
                    unary_elementwise(ca, |opt_s| {
                        opt_s.map(|s| s.trim_matches(|c| pat.contains(c)))
                    })
                }
            } else {
                unary_elementwise(ca, |opt_s| opt_s.map(|s| s.trim()))
            }
        },
        _ => broadcast_binary_elementwise(ca, pat, strip_chars_binary),
    }
}

pub fn strip_chars_start(ca: &StringChunked, pat: &StringChunked) -> StringChunked {
    match pat.len() {
        1 => {
            if let Some(pat) = pat.get(0) {
                if pat.chars().count() == 1 {
                    // Fast path for when a single character is passed
                    unary_elementwise(ca, |opt_s| {
                        opt_s.map(|s| s.trim_start_matches(pat.chars().next().unwrap()))
                    })
                } else {
                    unary_elementwise(ca, |opt_s| {
                        opt_s.map(|s| s.trim_start_matches(|c| pat.contains(c)))
                    })
                }
            } else {
                unary_elementwise(ca, |opt_s| opt_s.map(|s| s.trim_start()))
            }
        },
        _ => broadcast_binary_elementwise(ca, pat, strip_chars_start_binary),
    }
}

pub fn strip_chars_end(ca: &StringChunked, pat: &StringChunked) -> StringChunked {
    match pat.len() {
        1 => {
            if let Some(pat) = pat.get(0) {
                if pat.chars().count() == 1 {
                    // Fast path for when a single character is passed
                    unary_elementwise(ca, |opt_s| {
                        opt_s.map(|s| s.trim_end_matches(pat.chars().next().unwrap()))
                    })
                } else {
                    unary_elementwise(ca, |opt_s| {
                        opt_s.map(|s| s.trim_end_matches(|c| pat.contains(c)))
                    })
                }
            } else {
                unary_elementwise(ca, |opt_s| opt_s.map(|s| s.trim_end()))
            }
        },
        _ => broadcast_binary_elementwise(ca, pat, strip_chars_end_binary),
    }
}

pub fn strip_prefix(ca: &StringChunked, prefix: &StringChunked) -> StringChunked {
    match prefix.len() {
        1 => match prefix.get(0) {
            Some(prefix) => unary_elementwise(ca, |opt_s| {
                opt_s.map(|s| s.strip_prefix(prefix).unwrap_or(s))
            }),
            _ => StringChunked::full_null(ca.name(), ca.len()),
        },
        _ => broadcast_binary_elementwise(ca, prefix, strip_prefix_binary),
    }
}

pub fn strip_suffix(ca: &StringChunked, suffix: &StringChunked) -> StringChunked {
    match suffix.len() {
        1 => match suffix.get(0) {
            Some(suffix) => unary_elementwise(ca, |opt_s| {
                opt_s.map(|s| s.strip_suffix(suffix).unwrap_or(s))
            }),
            _ => StringChunked::full_null(ca.name(), ca.len()),
        },
        _ => broadcast_binary_elementwise(ca, suffix, strip_suffix_binary),
    }
}
