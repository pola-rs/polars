use std::iter::repeat;

use super::utils::{iter_with_view_and_buffers, subview_from_str};
use super::*;

macro_rules! impl_strip_chars_binary {
    ($name:ident, $trimmer:ident, $alt_trimmer:ident) => {
        fn $name<'a>(opt_s: Option<&'a str>, opt_pat: Option<&str>) -> Option<&'a str> {
            match (opt_s, opt_pat) {
                (Some(s), Some(pat)) => {
                    let mut chars = pat.chars();
                    let Some(c) = chars.next() else {
                        return Some(s);
                    };
                    if chars.next().is_none() {
                        Some(s.$trimmer(c))
                    } else {
                        Some(s.$trimmer(|c| pat.contains(c)))
                    }
                },
                (Some(s), _) => Some(s.$alt_trimmer()),
                _ => None,
            }
        }
    };
}

impl_strip_chars_binary!(strip_chars_binary, trim_matches, trim);
impl_strip_chars_binary!(strip_chars_start_binary, trim_start_matches, trim_start);
impl_strip_chars_binary!(strip_chars_end_binary, trim_end_matches, trim_end);

fn strip_prefix_binary<'a>(s: Option<&'a str>, prefix: Option<&str>) -> Option<&'a str> {
    Some(s?.strip_prefix(prefix?).unwrap_or(s?))
}

fn strip_suffix_binary<'a>(s: Option<&'a str>, suffix: Option<&str>) -> Option<&'a str> {
    Some(s?.strip_suffix(suffix?).unwrap_or(s?))
}

pub fn strip_chars(ca: &StringChunked, pat: &StringChunked) -> PolarsResult<StringChunked> {
    let result_len = match (ca.len(), pat.len()) {
        (1, len) => len,
        (len, 1) => len,
        (len_a, len_b) => {
            polars_ensure!(len_a == len_b, ComputeError: "pat's length: {} does not match that of the argument series: {}", len_b, len_a);
            len_a
        },
    };
    let mut builder = StringChunkedBuilder::new(ca.name(), result_len);
    for ((opt_s, (&view, buffers)), opt_pat) in repeat(ca)
        .flat_map(iter_with_view_and_buffers)
        .zip(repeat(pat).flatten())
        .take(result_len)
    {
        if let Some(subs) = strip_chars_binary(opt_s, opt_pat) {
            builder.append_view(subview_from_str(opt_s.unwrap(), subs, view), buffers);
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish())
}

pub fn strip_chars_start(ca: &StringChunked, pat: &StringChunked) -> PolarsResult<StringChunked> {
    let result_len = match (ca.len(), pat.len()) {
        (1, len) => len,
        (len, 1) => len,
        (len_a, len_b) => {
            polars_ensure!(len_a == len_b, ComputeError: "pat's length: {} does not match that of the argument series: {}", len_b, len_a);
            len_a
        },
    };
    let mut builder = StringChunkedBuilder::new(ca.name(), result_len);
    for ((opt_s, (&view, buffers)), opt_pat) in repeat(ca)
        .flat_map(iter_with_view_and_buffers)
        .zip(repeat(pat).flatten())
        .take(result_len)
    {
        if let Some(subs) = strip_chars_start_binary(opt_s, opt_pat) {
            builder.append_view(subview_from_str(opt_s.unwrap(), subs, view), buffers);
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish())
}

pub fn strip_chars_end(ca: &StringChunked, pat: &StringChunked) -> PolarsResult<StringChunked> {
    let result_len = match (ca.len(), pat.len()) {
        (1, len) => len,
        (len, 1) => len,
        (len_a, len_b) => {
            polars_ensure!(len_a == len_b, ComputeError: "pat's length: {} does not match that of the argument series: {}", len_b, len_a);
            len_a
        },
    };
    let mut builder = StringChunkedBuilder::new(ca.name(), result_len);
    for ((opt_s, (&view, buffers)), opt_pat) in repeat(ca)
        .flat_map(iter_with_view_and_buffers)
        .zip(repeat(pat).flatten())
        .take(result_len)
    {
        if let Some(subs) = strip_chars_end_binary(opt_s, opt_pat) {
            builder.append_view(subview_from_str(opt_s.unwrap(), subs, view), buffers);
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish())
}

pub fn strip_prefix(ca: &StringChunked, prefix: &StringChunked) -> PolarsResult<StringChunked> {
    let result_len = match (ca.len(), prefix.len()) {
        (1, len) => {
            if ca.has_nulls() {
                return Ok(StringChunked::full_null(ca.name(), ca.len()));
            }
            len
        },
        (len, 1) => {
            if prefix.has_nulls() {
                return Ok(StringChunked::full_null(ca.name(), ca.len()));
            }
            len
        },
        (len_a, len_b) => {
            polars_ensure!(len_a == len_b, ComputeError: "prefix's length: {} does not match that of the argument series: {}", len_b, len_a);
            len_a
        },
    };
    let mut builder = StringChunkedBuilder::new(ca.name(), result_len);
    for ((opt_s, (&view, buffers)), opt_prefix) in repeat(ca)
        .flat_map(iter_with_view_and_buffers)
        .zip(repeat(prefix).flatten())
        .take(result_len)
    {
        if let Some(subs) = strip_prefix_binary(opt_s, opt_prefix) {
            builder.append_view(subview_from_str(opt_s.unwrap(), subs, view), buffers);
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish())
}

pub fn strip_suffix(ca: &StringChunked, suffix: &StringChunked) -> PolarsResult<StringChunked> {
    let result_len = match (ca.len(), suffix.len()) {
        (1, len) => {
            if ca.has_nulls() {
                return Ok(StringChunked::full_null(ca.name(), ca.len()));
            }
            len
        },
        (len, 1) => {
            if suffix.has_nulls() {
                return Ok(StringChunked::full_null(ca.name(), ca.len()));
            }
            len
        },
        (len_a, len_b) => {
            polars_ensure!(len_a == len_b, ComputeError: "suffix's length: {} does not match that of the argument series: {}", len_b, len_a);
            len_a
        },
    };
    let mut builder = StringChunkedBuilder::new(ca.name(), result_len);
    for ((opt_s, (&view, buffers)), opt_suffix) in repeat(ca)
        .flat_map(iter_with_view_and_buffers)
        .zip(repeat(suffix).flatten())
        .take(result_len)
    {
        if let Some(subs) = strip_suffix_binary(opt_s, opt_suffix) {
            builder.append_view(subview_from_str(opt_s.unwrap(), subs, view), buffers);
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish())
}
