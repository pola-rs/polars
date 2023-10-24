use std::fmt::Write;

use polars_core::prelude::Utf8Chunked;

pub(super) fn pad_end<'a>(ca: &'a Utf8Chunked, width: usize, fillchar: char) -> Utf8Chunked {
    // amortize allocation
    let mut buf = String::new();
    let f = |s: &'a str| {
        let padding = width.saturating_sub(s.len());
        if padding == 0 {
            s
        } else {
            buf.clear();
            buf.push_str(s);
            for _ in 0..padding {
                buf.push(fillchar)
            }
            // extend lifetime
            // lifetime is bound to 'a
            let slice = buf.as_str();
            unsafe { std::mem::transmute::<&str, &'a str>(slice) }
        }
    };
    ca.apply_mut(f)
}

pub(super) fn pad_start<'a>(ca: &'a Utf8Chunked, width: usize, fillchar: char) -> Utf8Chunked {
    // amortize allocation
    let mut buf = String::new();
    let f = |s: &'a str| {
        let padding = width.saturating_sub(s.len());
        if padding == 0 {
            s
        } else {
            buf.clear();
            for _ in 0..padding {
                buf.push(fillchar)
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

pub(super) fn zfill<'a>(ca: &'a Utf8Chunked, alignment: usize) -> Utf8Chunked {
    // amortize allocation
    let mut buf = String::new();
    let f = |s: &'a str| {
        let alignment = alignment.saturating_sub(s.len());
        if alignment == 0 {
            return s;
        }
        buf.clear();
        if let Some(stripped) = s.strip_prefix('-') {
            write!(
                &mut buf,
                "-{:0alignment$}{value}",
                0,
                alignment = alignment,
                value = stripped
            )
            .unwrap();
        } else {
            write!(
                &mut buf,
                "{:0alignment$}{value}",
                0,
                alignment = alignment,
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
