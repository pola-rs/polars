use std::fmt::Write;

use polars_core::prelude::arity::broadcast_binary_elementwise;
use polars_core::prelude::{StringChunked, UInt64Chunked};

fn pad_fn<'a>(
    s: Option<&'a str>,
    length: Option<u64>,
    buf: &mut String,
    fill_char: char,
    pad_start: bool,
) -> Option<&'a str> {
    if let (Some(s), Some(length)) = (s, length) {
        let length = length as usize;
        let n_chars = s.chars().count();
        if length <= n_chars {
            return Some(s);
        }
        let padding = length - n_chars;
        buf.clear();
        if !pad_start {
            buf.push_str(s);
        }
        for _ in 0..padding {
            buf.push(fill_char)
        }
        if pad_start {
            buf.push_str(s);
        }
        // extend lifetime
        // lifetime is bound to 'a
        let slice = buf.as_str();
        Some(unsafe { std::mem::transmute::<&str, &'a str>(slice) })
    } else {
        None
    }
}

fn zfill_fn<'a>(s: Option<&'a str>, length: Option<u64>, buf: &mut String) -> Option<&'a str> {
    if let (Some(s), Some(length)) = (s, length) {
        let s_len = s.len();
        let length = length as usize;
        if length <= s_len {
            return Some(s);
        }
        buf.clear();
        let length = length - s_len;
        if let Some(stripped) = s.strip_prefix('-') {
            write!(buf, "-{:0length$}{stripped}", 0,).unwrap();
        } else if let Some(stripped) = s.strip_prefix('+') {
            write!(buf, "+{:0length$}{stripped}", 0,).unwrap();
        } else {
            write!(buf, "{:0length$}{s}", 0,).unwrap();
        };
        // extend lifetime
        // lifetime is bound to 'a
        let slice = buf.as_str();
        Some(unsafe { std::mem::transmute::<&str, &'a str>(slice) })
    } else {
        None
    }
}

pub(super) fn zfill<'a>(ca: &'a StringChunked, length: &'a UInt64Chunked) -> StringChunked {
    // amortize allocation
    let mut buf = String::new();
    fn infer<F: for<'a> FnMut(Option<&'a str>, Option<u64>) -> Option<&'a str>>(f: F) -> F where {
        f
    }
    broadcast_binary_elementwise(
        ca,
        length,
        infer(|opt_s, opt_len| zfill_fn(opt_s, opt_len, &mut buf)),
    )
}

pub(super) fn pad_start<'a>(
    ca: &'a StringChunked,
    length: &'a UInt64Chunked,
    fill_char: char,
) -> StringChunked {
    // amortize allocation
    let mut buf = String::new();
    fn infer<F: for<'a> FnMut(Option<&'a str>, Option<u64>) -> Option<&'a str>>(f: F) -> F where {
        f
    }
    broadcast_binary_elementwise(
        ca,
        length,
        infer(|opt_s, opt_len| pad_fn(opt_s, opt_len, &mut buf, fill_char, true)),
    )
}

pub(super) fn pad_end<'a>(
    ca: &'a StringChunked,
    length: &'a UInt64Chunked,
    fill_char: char,
) -> StringChunked {
    // amortize allocation
    let mut buf = String::new();
    fn infer<F: for<'a> FnMut(Option<&'a str>, Option<u64>) -> Option<&'a str>>(f: F) -> F where {
        f
    }
    broadcast_binary_elementwise(
        ca,
        length,
        infer(|opt_s, opt_len| pad_fn(opt_s, opt_len, &mut buf, fill_char, false)),
    )
}
