use polars_core::prelude::arity::{binary_elementwise, ternary_elementwise, unary_elementwise};
use polars_core::prelude::{Int64Chunked, StringChunked, UInt64Chunked};

fn head_binary(opt_str_val: Option<&str>, opt_n: Option<i64>) -> Option<&str> {
    if let (Some(str_val), Some(mut n)) = (opt_str_val, opt_n) {
        let str_len = str_val.len() as i64;
        if n >= str_len {
            Some(str_val)
        } else if (n == 0) | (str_len == 0) | (n <= -str_len) {
            Some("")
        } else {
            if n < 0 {
                // If `n` is negative, it counts from the end of the string.
                n += str_len; // adding negative value
            }
            Some(&str_val[0..n as usize])
        }
    } else {
        None
    }
}

fn tail_binary(opt_str_val: Option<&str>, opt_n: Option<i64>) -> Option<&str> {
    if let (Some(str_val), Some(mut n)) = (opt_str_val, opt_n) {
        let str_len = str_val.len() as i64;
        if n >= str_len {
            Some(str_val)
        } else if (n == 0) | (str_len == 0) | (n <= -str_len) {
            Some("")
        } else {
            // We re-assign `n` to be the start of the slice.
            // The end of the slice is always the end of the string.
            if n < 0 {
                // If `n` is negative, we count from the beginning.
                n = -n;
            } else {
                // If `n` is positive, we count from the end.
                n = str_len - n;
            }
            Some(&str_val[n as usize..str_len as usize])
        }
    } else {
        None
    }
}

fn substring_ternary(
    opt_str_val: Option<&str>,
    opt_offset: Option<i64>,
    opt_length: Option<u64>,
) -> Option<&str> {
    match (opt_str_val, opt_offset) {
        (Some(str_val), Some(offset)) => {
            // If `offset` is negative, it counts from the end of the string.
            let offset = if offset >= 0 {
                offset as usize
            } else {
                let offset = (0i64 - offset) as usize;
                str_val
                    .char_indices()
                    .rev()
                    .nth(offset)
                    .map(|(idx, _)| idx + 1)
                    .unwrap_or(0)
            };

            let mut iter_chars = str_val.char_indices();
            if let Some((offset_idx, _)) = iter_chars.nth(offset) {
                let len_end = str_val.len() - offset_idx;

                // Slice to end of str if no length given.
                let length = if let Some(length) = opt_length {
                    length as usize
                } else {
                    len_end
                };

                if length == 0 {
                    return Some("");
                }

                let end_idx = iter_chars
                    .nth(length.saturating_sub(1))
                    .map(|(idx, _)| idx)
                    .unwrap_or(str_val.len());

                Some(&str_val[offset_idx..end_idx])
            } else {
                Some("")
            }
        },
        _ => None,
    }
}

pub(super) fn substring(
    ca: &StringChunked,
    offset: &Int64Chunked,
    length: &UInt64Chunked,
) -> StringChunked {
    match (ca.len(), offset.len(), length.len()) {
        (1, 1, _) => {
            // SAFETY: `ca` was verified to have least 1 element.
            let str_val = unsafe { ca.get_unchecked(0) };
            // SAFETY: `offset` was verified to have at least 1 element.
            let offset = unsafe { offset.get_unchecked(0) };
            unary_elementwise(length, |length| substring_ternary(str_val, offset, length))
                .with_name(ca.name())
        },
        (_, 1, 1) => {
            // SAFETY: `offset` was verified to have at least 1 element.
            let offset = unsafe { offset.get_unchecked(0) };
            // SAFETY: `length` was verified to have at least 1 element.
            let length = unsafe { length.get_unchecked(0) };
            unary_elementwise(ca, |str_val| substring_ternary(str_val, offset, length))
        },
        (1, _, 1) => {
            // SAFETY: `ca` was verified to have at least 1 element.
            let str_val = unsafe { ca.get_unchecked(0) };
            // SAFETY: `length` was verified to have at least 1 element.
            let length = unsafe { length.get_unchecked(0) };
            unary_elementwise(offset, |offset| substring_ternary(str_val, offset, length))
                .with_name(ca.name())
        },
        (1, len_b, len_c) if len_b == len_c => {
            // SAFETY: `ca` was verified to have at least 1 element.
            let str_val = unsafe { ca.get_unchecked(0) };
            binary_elementwise(offset, length, |offset, length| {
                substring_ternary(str_val, offset, length)
            })
        },
        (len_a, 1, len_c) if len_a == len_c => {
            fn infer<F: for<'a> FnMut(Option<&'a str>, Option<u64>) -> Option<&'a str>>(f: F) -> F where
            {
                f
            }
            // SAFETY: index `0` is in bound.
            let offset = unsafe { offset.get_unchecked(0) };
            binary_elementwise(
                ca,
                length,
                infer(|str_val, length| substring_ternary(str_val, offset, length)),
            )
        },
        (len_a, len_b, 1) if len_a == len_b => {
            fn infer<F: for<'a> FnMut(Option<&'a str>, Option<i64>) -> Option<&'a str>>(f: F) -> F where
            {
                f
            }
            // SAFETY: index `0` is in bound.
            let length = unsafe { length.get_unchecked(0) };
            binary_elementwise(
                ca,
                offset,
                infer(|str_val, offset| substring_ternary(str_val, offset, length)),
            )
        },
        _ => ternary_elementwise(ca, offset, length, substring_ternary),
    }
}

pub(super) fn head(ca: &StringChunked, n: &Int64Chunked) -> StringChunked {
    match (ca.len(), n.len()) {
        (_, 1) => {
            // SAFETY: `n` was verified to have at least 1 element.
            let n = unsafe { n.get_unchecked(0) };
            unary_elementwise(ca, |str_val| head_binary(str_val, n)).with_name(ca.name())
        },
        (1, _) => {
            // SAFETY: `ca` was verified to have at least 1 element.
            let str_val = unsafe { ca.get_unchecked(0) };
            unary_elementwise(n, |n| head_binary(str_val, n)).with_name(ca.name())
        },
        _ => binary_elementwise(ca, n, head_binary),
    }
}

pub(super) fn tail(ca: &StringChunked, n: &Int64Chunked) -> StringChunked {
    match (ca.len(), n.len()) {
        (_, 1) => {
            // SAFETY: `n` was verified to have at least 1 element.
            let n = unsafe { n.get_unchecked(0) };
            unary_elementwise(ca, |str_val| tail_binary(str_val, n)).with_name(ca.name())
        },
        (1, _) => {
            // SAFETY: `ca` was verified to have at least 1 element.
            let str_val = unsafe { ca.get_unchecked(0) };
            unary_elementwise(n, |n| tail_binary(str_val, n)).with_name(ca.name())
        },
        _ => binary_elementwise(ca, n, tail_binary),
    }
}
