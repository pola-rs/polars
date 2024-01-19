use std::fmt::Write;

use polars_core::prelude::arity::broadcast_binary_elementwise;
use polars_core::prelude::{StringChunked, UInt64Chunked};

pub(super) fn pad_end<'a>(ca: &'a StringChunked, length: usize, fill_char: char) -> StringChunked {
    // amortize allocation
    let mut buf = String::new();
    let f = |s: &'a str| {
        let padding = length.saturating_sub(s.chars().count());
        if padding == 0 {
            s
        } else {
            buf.clear();
            buf.push_str(s);
            for _ in 0..padding {
                buf.push(fill_char)
            }
            // extend lifetime
            // lifetime is bound to 'a
            let slice = buf.as_str();
            unsafe { std::mem::transmute::<&str, &'a str>(slice) }
        }
    };
    ca.apply_mut(f)
}

pub(super) fn pad_start<'a>(
    ca: &'a StringChunked,
    length: usize,
    fill_char: char,
) -> StringChunked {
    // amortize allocation
    let mut buf = String::new();
    let f = |s: &'a str| {
        let padding = length.saturating_sub(s.chars().count());
        if padding == 0 {
            s
        } else {
            buf.clear();
            for _ in 0..padding {
                buf.push(fill_char)
            }
            buf.push_str(s);
            // extend lifetime
            // lifetime is bound to 'a
            let slice = buf.as_str();
            unsafe { std::mem::transmute::<&str, &'a str>(slice) }
        }
    };
    ca.apply_mut(f)
}

fn zfill_fn<'a>(s: Option<&'a str>, len: Option<u64>, buf: &mut String) -> Option<&'a str> {
    match (s, len) {
        (Some(s), Some(length)) => {
            let length = length.saturating_sub(s.len() as u64);
            if length == 0 {
                return Some(s);
            }
            buf.clear();
            if let Some(stripped) = s.strip_prefix('-') {
                write!(
                    buf,
                    "-{:0length$}{value}",
                    0,
                    length = length as usize,
                    value = stripped
                )
                .unwrap();
            } else {
                write!(
                    buf,
                    "{:0length$}{value}",
                    0,
                    length = length as usize,
                    value = s
                )
                .unwrap();
            };
            // extend lifetime
            // lifetime is bound to 'a
            let slice = buf.as_str();
            Some(unsafe { std::mem::transmute::<&str, &'a str>(slice) })
        },
        _ => None,
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
