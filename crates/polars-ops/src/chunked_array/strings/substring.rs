use polars_core::prelude::arity::{binary_elementwise, ternary_elementwise, unary_elementwise};
use polars_core::prelude::{Int64Chunked, StringChunked, UInt64Chunked};

fn head_binary(opt_str_val: Option<&str>, opt_n: Option<i64>) -> Option<&str> {
    if let (Some(str_val), Some(n)) = (opt_str_val, opt_n) {
        // `max_len` is guaranteed to be at least the total number of characters.
        let max_len = str_val.len();
        if n == 0 {
            Some("")
        } else {
            let end_idx = if n > 0 {
                if n as usize >= max_len {
                    return opt_str_val;
                }
                // End after the nth codepoint.
                str_val
                    .char_indices()
                    .nth(n as usize)
                    .map(|(idx, _)| idx)
                    .unwrap_or(max_len)
            } else {
                // End after the nth codepoint from the end.
                str_val
                    .char_indices()
                    .rev()
                    .nth((-n - 1) as usize)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            };
            Some(&str_val[..end_idx])
        }
    } else {
        None
    }
}

fn tail_binary(opt_str_val: Option<&str>, opt_n: Option<i64>) -> Option<&str> {
    if let (Some(str_val), Some(n)) = (opt_str_val, opt_n) {
        // `max_len` is guaranteed to be at least the total number of characters.
        let max_len = str_val.len();
        if n == 0 {
            Some("")
        } else {
            let start_idx = if n > 0 {
                if n as usize >= max_len {
                    return opt_str_val;
                }
                // Start from nth codepoint from the end
                str_val
                    .char_indices()
                    .rev()
                    .nth((n - 1) as usize)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            } else {
                // Start after the nth codepoint
                str_val
                    .char_indices()
                    .nth((-n) as usize)
                    .map(|(idx, _)| idx)
                    .unwrap_or(max_len)
            };
            Some(&str_val[start_idx..])
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
    let str_val = opt_str_val?;
    let offset = opt_offset?;

    // Fast-path: always empty string.
    if opt_length == Some(0) || offset >= str_val.len() as i64 {
        return Some("");
    }

    let mut indices = str_val.char_indices().map(|(o, _)| o);
    let mut length_reduction = 0;
    let start_byte_offset = if offset >= 0 {
        indices.nth(offset as usize).unwrap_or(str_val.len())
    } else {
        // If `offset` is negative, it counts from the end of the string.
        let mut chars_skipped = 0;
        let found = indices
            .inspect(|_| chars_skipped += 1)
            .nth_back((-offset - 1) as usize);

        // If we didn't find our char that means our offset was so negative it
        // is before the start of our string. This means our length must be
        // reduced, assuming it is finite.
        if let Some(off) = found {
            off
        } else {
            length_reduction = (-offset) as usize - chars_skipped;
            0
        }
    };

    let str_val = &str_val[start_byte_offset..];
    let mut indices = str_val.char_indices().map(|(o, _)| o);
    let stop_byte_offset = opt_length
        .and_then(|l| indices.nth((l as usize).saturating_sub(length_reduction)))
        .unwrap_or(str_val.len());
    Some(&str_val[..stop_byte_offset])
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
