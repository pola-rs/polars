use polars_core::prelude::arity::{binary_elementwise, ternary_elementwise};

use crate::chunked_array::{Int64Chunked, UInt64Chunked, Utf8Chunked};

fn utf8_substring_ternary<'a>(
    opt_str_val: Option<&'a str>,
    opt_offset: Option<i64>,
    opt_length: Option<u64>,
) -> Option<&'a str> {
    match (opt_str_val, opt_offset) {
        (Some(str_val), Some(offset)) => {
            // if `offset` is negative, it counts from the end of the string
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

                // slice to end of str if no length given
                let length = match opt_length {
                    Some(length) => length as usize,
                    _ => len_end,
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

pub(super) fn utf8_substring(
    ca: &Utf8Chunked,
    offset: &Int64Chunked,
    length: &UInt64Chunked,
) -> Utf8Chunked {
    match (offset.len(), length.len()) {
        (1, 1) => ca.apply_generic(|opt_str_val| {
            utf8_substring_ternary(opt_str_val, offset.get(0), length.get(0))
        }),
        (1, _) => {
            fn infer<F: for<'a> FnMut(Option<&'a str>, Option<u64>) -> Option<&'a str>>(f: F) -> F where
            {
                f
            }

            let off = offset.get(0);
            binary_elementwise(
                ca,
                length,
                infer(|val, len| utf8_substring_ternary(val, off, len)),
            )
        },
        (_, 1) => {
            fn infer<F: for<'a> FnMut(Option<&'a str>, Option<i64>) -> Option<&'a str>>(f: F) -> F where
            {
                f
            }

            let len = length.get(0);
            binary_elementwise(
                ca,
                offset,
                infer(|val, off| utf8_substring_ternary(val, off, len)),
            )
        },
        _ => ternary_elementwise(ca, offset, length, utf8_substring_ternary),
    }
}
