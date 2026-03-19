use arrow::array::View;
use polars_core::prelude::arity::{binary_elementwise, ternary_elementwise, unary_elementwise};
use polars_core::prelude::{ChunkFullNull, Int64Chunked, StringChunked, UInt64Chunked};
use polars_error::{PolarsResult, polars_ensure};

fn is_utf8_codepoint_start(b: u8) -> bool {
    // The top two bits of a continuation byte are 10. Any other value is a
    // starting byte. We can use signed comparison to test for this in one
    // instruction, as the top bits 11, 00 and 01 are all more positive and thus
    // larger in signed comparison.
    (b as i8) >= (0b1100_0000_u8 as i8)
}

/// Similar to char_to_byte_idx but if `char_idx` would be out-of-bounds the
/// number of codepoints in s is returned as an error.
pub fn char_to_byte_idx_or_cp_count(s: &str, char_idx: usize) -> Result<usize, usize> {
    let bytes = s.as_bytes();
    if char_idx == 0 {
        return Ok(0);
    }

    let mut offset = 0;
    let mut num_chars_seen = 0;

    // Auto-vectorized bulk processing, but skip if index is small.
    if char_idx >= 16 {
        while let Some(chunk) = bytes.get(offset..offset + 16) {
            let chunk_seen: usize = chunk
                .iter()
                .map(|b| is_utf8_codepoint_start(*b) as usize)
                .sum();
            if num_chars_seen + chunk_seen > char_idx {
                break;
            }
            offset += 16;
            num_chars_seen += chunk_seen;
        }
    }

    while let Some(b) = bytes.get(offset) {
        num_chars_seen += is_utf8_codepoint_start(*b) as usize;
        if num_chars_seen > char_idx {
            return Ok(offset);
        }
        offset += 1;
    }

    debug_assert!(offset == bytes.len());
    Err(num_chars_seen)
}

/// Given an offset to the start of the `char_idx`th codepoint, returns the
/// equivalent offset in bytes.
///
/// If `char_idx` would be out-of-bounds s.len() is returned.
pub fn char_to_byte_idx(s: &str, char_idx: usize) -> usize {
    if char_idx >= s.len() {
        // No need to even count.
        s.len()
    } else {
        char_to_byte_idx_or_cp_count(s, char_idx).unwrap_or(s.len())
    }
}

/// Similar to rev_char_to_byte_idx but if `char_idx` would be out-of-bounds the
/// number of codepoints in s is returned as an error.
pub fn rev_char_to_byte_idx_or_cp_count(s: &str, rev_char_idx: usize) -> Result<usize, usize> {
    let bytes = s.as_bytes();
    if rev_char_idx == 0 {
        return Ok(bytes.len());
    }

    let mut offset = s.len();
    let mut num_chars_seen = 0;

    // Auto-vectorized bulk processing, but skip if index is small.
    if rev_char_idx >= 16 {
        while offset >= 16 {
            let chunk = unsafe { bytes.get_unchecked(offset - 16..offset) };
            let chunk_seen: usize = chunk
                .iter()
                .map(|b| is_utf8_codepoint_start(*b) as usize)
                .sum();
            if num_chars_seen + chunk_seen >= rev_char_idx {
                break;
            }
            offset -= 16;
            num_chars_seen += chunk_seen;
        }
    }

    while offset > 0 {
        offset -= 1;
        let byte = unsafe { bytes.get_unchecked(offset) };
        num_chars_seen += is_utf8_codepoint_start(*byte) as usize;
        if num_chars_seen >= rev_char_idx {
            return Ok(offset);
        }
    }

    debug_assert!(offset == 0);
    Err(num_chars_seen)
}

/// Counts rev_char_idx code points from *the end* of the string, returning an
/// offset in bytes where this codepoint ends.
///
/// For example, rev_char_to_byte_idx(0, s) returns s.len(), and
/// rev_char_to_byte_idx(1, s) returns s.len() - width(last_codepoint_in_s).
///
/// If rev_char_idx is large enough that we would go out of bounds, 0 is returned.
pub fn rev_char_to_byte_idx(s: &str, rev_char_idx: usize) -> usize {
    if rev_char_idx >= s.len() {
        // No need to even count.
        0
    } else {
        rev_char_to_byte_idx_or_cp_count(s, rev_char_idx).unwrap_or(0)
    }
}

fn head_binary(opt_str_val: Option<&str>, opt_n: Option<i64>) -> Option<&str> {
    if let (Some(str_val), Some(n)) = (opt_str_val, opt_n) {
        let end_idx = head_binary_values(str_val, n);
        Some(unsafe { str_val.get_unchecked(..end_idx) })
    } else {
        None
    }
}

fn head_binary_values(str_val: &str, n: i64) -> usize {
    if n >= 0 {
        char_to_byte_idx(str_val, n as usize)
    } else {
        rev_char_to_byte_idx(str_val, (-n) as usize)
    }
}

fn tail_binary(opt_str_val: Option<&str>, opt_n: Option<i64>) -> Option<&str> {
    if let (Some(str_val), Some(n)) = (opt_str_val, opt_n) {
        let start_idx = tail_binary_values(str_val, n);
        Some(unsafe { str_val.get_unchecked(start_idx..) })
    } else {
        None
    }
}

fn tail_binary_values(str_val: &str, n: i64) -> usize {
    if n >= 0 {
        rev_char_to_byte_idx(str_val, n as usize)
    } else {
        char_to_byte_idx(str_val, (-n) as usize)
    }
}

fn substring_ternary_offsets(
    opt_str_val: Option<&str>,
    opt_offset: Option<i64>,
    opt_length: Option<u64>,
) -> Option<(usize, usize)> {
    let str_val = opt_str_val?;
    let offset = opt_offset?;
    Some(substring_ternary_offsets_value(
        str_val,
        offset,
        opt_length.unwrap_or(u64::MAX),
    ))
}

pub fn substring_ternary_offsets_value(
    str_val: &str,
    offset: i64,
    mut length: u64,
) -> (usize, usize) {
    // Fast-path: always empty string.
    if length == 0 || offset >= str_val.len() as i64 {
        return (0, 0);
    }

    let start_byte_offset = if offset >= 0 {
        char_to_byte_idx(str_val, offset as usize)
    } else {
        // Fast-path: always empty string.
        let end_offset_upper_bound = offset
            .saturating_add(str_val.len() as i64)
            .saturating_add(length.try_into().unwrap_or(i64::MAX));
        if end_offset_upper_bound < 0 {
            return (0, 0);
        }

        match rev_char_to_byte_idx_or_cp_count(str_val, (-offset) as usize) {
            Ok(so) => so,
            Err(n_cp) => {
                // Our offset was so negative it is before the start of our string.
                // This means our length must be reduced, assuming it is finite.
                length = length.saturating_sub((-offset) as u64 - n_cp as u64);
                0
            },
        }
    };

    let stop_byte_offset = char_to_byte_idx(&str_val[start_byte_offset..], length as usize);
    (start_byte_offset, start_byte_offset + stop_byte_offset)
}

fn substring_ternary(
    opt_str_val: Option<&str>,
    opt_offset: Option<i64>,
    opt_length: Option<u64>,
) -> Option<&str> {
    let (start, end) = substring_ternary_offsets(opt_str_val, opt_offset, opt_length)?;
    unsafe { opt_str_val.map(|str_val| str_val.get_unchecked(start..end)) }
}

pub fn update_view(mut view: View, start: usize, end: usize, val: &str) -> View {
    let length = (end - start) as u32;
    view.length = length;

    // SAFETY: we just compute the start /end.
    let subval = unsafe { val.get_unchecked(start..end).as_bytes() };

    if length <= 12 {
        View::new_inline(subval)
    } else {
        view.offset += start as u32;
        view.length = length;
        view.prefix = u32::from_le_bytes(subval[0..4].try_into().unwrap());
        view
    }
}

pub(super) fn substring(
    ca: &StringChunked,
    offset: &Int64Chunked,
    length: &UInt64Chunked,
) -> StringChunked {
    match (ca.len(), offset.len(), length.len()) {
        (1, 1, _) => {
            let str_val = ca.get(0);
            let offset = offset.get(0);
            unary_elementwise(length, |length| substring_ternary(str_val, offset, length))
                .with_name(ca.name().clone())
        },
        (_, 1, 1) => {
            let offset = offset.get(0);
            let length = length.get(0).unwrap_or(u64::MAX);

            let Some(offset) = offset else {
                return StringChunked::full_null(ca.name().clone(), ca.len());
            };

            unsafe {
                ca.apply_views(|view, val| {
                    let (start, end) = substring_ternary_offsets_value(val, offset, length);
                    update_view(view, start, end, val)
                })
            }
        },
        (1, _, 1) => {
            let str_val = ca.get(0);
            let length = length.get(0);
            unary_elementwise(offset, |offset| substring_ternary(str_val, offset, length))
                .with_name(ca.name().clone())
        },
        (1, len_b, len_c) if len_b == len_c => {
            let str_val = ca.get(0);
            binary_elementwise(offset, length, |offset, length| {
                substring_ternary(str_val, offset, length)
            })
        },
        (len_a, 1, len_c) if len_a == len_c => {
            fn infer<F: for<'a> FnMut(Option<&'a str>, Option<u64>) -> Option<&'a str>>(f: F) -> F where
            {
                f
            }
            let offset = offset.get(0);
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
            let length = length.get(0);
            binary_elementwise(
                ca,
                offset,
                infer(|str_val, offset| substring_ternary(str_val, offset, length)),
            )
        },
        _ => ternary_elementwise(ca, offset, length, substring_ternary),
    }
}

pub(super) fn head(ca: &StringChunked, n: &Int64Chunked) -> PolarsResult<StringChunked> {
    match (ca.len(), n.len()) {
        (len, 1) => {
            let n = n.get(0);
            let Some(n) = n else {
                return Ok(StringChunked::full_null(ca.name().clone(), len));
            };

            Ok(unsafe {
                ca.apply_views(|view, val| {
                    let end = head_binary_values(val, n);
                    update_view(view, 0, end, val)
                })
            })
        },
        // TODO! below should also work on only views
        (1, _) => {
            let str_val = ca.get(0);
            Ok(unary_elementwise(n, |n| head_binary(str_val, n)).with_name(ca.name().clone()))
        },
        (a, b) => {
            polars_ensure!(a == b, ShapeMismatch: "lengths of arguments do not align in 'str.head' got length: {} for column: {}, got length: {} for argument 'n'", a, ca.name(), b);
            Ok(binary_elementwise(ca, n, head_binary))
        },
    }
}

pub(super) fn tail(ca: &StringChunked, n: &Int64Chunked) -> PolarsResult<StringChunked> {
    Ok(match (ca.len(), n.len()) {
        (len, 1) => {
            let n = n.get(0);
            let Some(n) = n else {
                return Ok(StringChunked::full_null(ca.name().clone(), len));
            };
            unsafe {
                ca.apply_views(|view, val| {
                    let start = tail_binary_values(val, n);
                    update_view(view, start, val.len(), val)
                })
            }
        },
        // TODO! below should also work on only views
        (1, _) => {
            let str_val = ca.get(0);
            unary_elementwise(n, |n| tail_binary(str_val, n)).with_name(ca.name().clone())
        },
        (a, b) => {
            polars_ensure!(a == b, ShapeMismatch: "lengths of arguments do not align in 'str.tail' got length: {} for column: {}, got length: {} for argument 'n'", a, ca.name(), b);
            binary_elementwise(ca, n, tail_binary)
        },
    })
}
