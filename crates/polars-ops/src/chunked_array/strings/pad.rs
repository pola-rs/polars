use std::fmt::Write;

use polars_core::prelude::Utf8Chunked;

pub(super) fn pad_end<'a>(ca: &'a Utf8Chunked, length: usize, fill_char: char) -> Utf8Chunked {
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

pub(super) fn pad_start<'a>(ca: &'a Utf8Chunked, length: usize, fill_char: char) -> Utf8Chunked {
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

pub(super) fn zfill<'a>(ca: &'a Utf8Chunked, length: usize) -> Utf8Chunked {
    // amortize allocation
    let mut buf = String::new();
    let f = |s: &'a str| {
        let length = length.saturating_sub(s.len());
        if length == 0 {
            return s;
        }
        buf.clear();
        if let Some(stripped) = s.strip_prefix('-') {
            write!(
                &mut buf,
                "-{:0length$}{value}",
                0,
                length = length,
                value = stripped
            )
            .unwrap();
        } else {
            write!(
                &mut buf,
                "{:0length$}{value}",
                0,
                length = length,
                value = s
            )
            .unwrap();
        };
        // extend lifetime
        // lifetime is bound to 'a
        let slice = buf.as_str();
        unsafe { std::mem::transmute::<&str, &'a str>(slice) }
    };
    ca.apply_mut(f)
}
