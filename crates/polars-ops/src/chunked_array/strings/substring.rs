use arrow::array::View;
use polars_core::prelude::arity::{binary_elementwise, ternary_elementwise, unary_elementwise};
use polars_core::prelude::{ChunkFullNull, Int64Chunked, StringChunked, UInt64Chunked};
use polars_error::{polars_ensure, PolarsResult};

fn head_binary(opt_str_val: Option<&str>, opt_n: Option<i64>) -> Option<&str> {
    if let (Some(str_val), Some(n)) = (opt_str_val, opt_n) {
        let end_idx = head_binary_values(str_val, n);
        Some(unsafe { str_val.get_unchecked(..end_idx) })
    } else {
        None
    }
}

fn head_binary_values(str_val: &str, n: i64) -> usize {
    if n == 0 {
        0
    } else {
        let end_idx = if n > 0 {
            if n as usize >= str_val.len() {
                return str_val.len();
            }
            // End after the nth codepoint.
            str_val
                .char_indices()
                .nth(n as usize)
                .map(|(idx, _)| idx)
                .unwrap_or(str_val.len())
        } else {
            // End after the nth codepoint from the end.
            str_val
                .char_indices()
                .rev()
                .nth((-n - 1) as usize)
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        };
        end_idx
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
    // `max_len` is guaranteed to be at least the total number of characters.
    let max_len = str_val.len();
    if n == 0 {
        max_len
    } else {
        let start_idx = if n > 0 {
            if n as usize >= max_len {
                return 0;
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
        start_idx
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

fn substring_ternary_offsets_value(str_val: &str, offset: i64, length: u64) -> (usize, usize) {
    // Fast-path: always empty string.
    if length == 0 || offset >= str_val.len() as i64 {
        return (0, 0);
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
    let stop_byte_offset = indices
        .nth((length as usize).saturating_sub(length_reduction))
        .unwrap_or(str_val.len());
    (start_byte_offset, stop_byte_offset + start_byte_offset)
}

fn substring_ternary(
    opt_str_val: Option<&str>,
    opt_offset: Option<i64>,
    opt_length: Option<u64>,
) -> Option<&str> {
    let (start, end) = substring_ternary_offsets(opt_str_val, opt_offset, opt_length)?;
    unsafe { opt_str_val.map(|str_val| str_val.get_unchecked(start..end)) }
}

fn update_view(mut view: View, start: usize, end: usize, val: &str) -> View {
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
                .with_name(ca.name())
        },
        (_, 1, 1) => {
            let offset = offset.get(0);
            let length = length.get(0).unwrap_or(u64::MAX);

            let Some(offset) = offset else {
                return StringChunked::full_null(ca.name(), ca.len());
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
                .with_name(ca.name())
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
                return Ok(StringChunked::full_null(ca.name(), len));
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
            Ok(unary_elementwise(n, |n| head_binary(str_val, n)).with_name(ca.name()))
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
                return Ok(StringChunked::full_null(ca.name(), len));
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
            unary_elementwise(n, |n| tail_binary(str_val, n)).with_name(ca.name())
        },
        (a, b) => {
            polars_ensure!(a == b, ShapeMismatch: "lengths of arguments do not align in 'str.tail' got length: {} for column: {}, got length: {} for argument 'n'", a, ca.name(), b);
            binary_elementwise(ca, n, tail_binary)
        },
    })
}
