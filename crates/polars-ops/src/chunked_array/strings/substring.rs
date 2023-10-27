use polars_core::prelude::arity::ternary_elementwise;

use crate::chunked_array::{Int64Chunked, UInt64Chunked, Utf8Chunked};

/// Returns a Utf8Array<O> with a substring starting from `start` and with optional length `length` of each of the elements in `array`.
/// `offset` can be negative, in which case the offset counts from the end of the string.
fn utf8_substring_ternary<'a>(
    opt_str_val: Option<&'a str>,
    opt_offset: Option<i64>,
    opt_length: Option<u64>,
) -> Option<&'a str> {
    match (opt_str_val, opt_offset) {
        (Some(str_val), Some(offset)) => {
            // compute where we should offset slicing this entry.
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
                // length of the str
                let len_end = str_val.len() - offset_idx;

                // slice to end of str if no length given
                let length = match opt_length {
                    Some(length) => length as usize,
                    _ => len_end,
                };

                if length == 0 {
                    return Some("");
                }
                // compute
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
    ternary_elementwise(ca, offset, length, utf8_substring_ternary)
}
