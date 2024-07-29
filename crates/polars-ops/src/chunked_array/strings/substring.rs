use std::iter::repeat;

use arrow::array::View;
use polars_core::prelude::{Int64Chunked, StringChunked, StringChunkedBuilder, UInt64Chunked};
use polars_error::{polars_ensure, PolarsResult};

use super::utils::{iter_with_view_and_buffers, subview};

fn head_binary(opt_str_val: Option<&str>, view: View, opt_n: Option<i64>) -> Option<View> {
    let str_val = opt_str_val?;
    let n = opt_n?;
    // `max_len` is guaranteed to be at least the total number of characters.
    let max_len = str_val.len();
    if n == 0 {
        Some(View::default())
    } else {
        let end_idx = if n > 0 {
            if n as usize >= max_len {
                return Some(view);
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
        Some(subview(str_val, view, 0, end_idx))
    }
}

fn tail_binary(opt_str_val: Option<&str>, view: View, opt_n: Option<i64>) -> Option<View> {
    let str_val = opt_str_val?;
    let n = opt_n?;
    // `max_len` is guaranteed to be at least the total number of characters.
    let max_len = str_val.len();
    if n == 0 {
        Some(View::default())
    } else {
        let start_idx = if n > 0 {
            if n as usize >= max_len {
                return Some(view);
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
        Some(subview(str_val, view, start_idx, str_val.len()))
    }
}

fn substring_ternary(
    opt_str_val: Option<&str>,
    view: View,
    opt_offset: Option<i64>,
    opt_length: Option<u64>,
) -> Option<View> {
    let str_val = opt_str_val?;
    let offset = opt_offset?;

    // Fast-path: always empty string.
    if opt_length == Some(0) || offset >= str_val.len() as i64 {
        return Some(View::default());
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

    let trailing_str_val = &str_val[start_byte_offset..];
    let mut indices = trailing_str_val.char_indices().map(|(o, _)| o);
    let stop_byte_offset = opt_length
        .and_then(|l| indices.nth((l as usize).saturating_sub(length_reduction)))
        .unwrap_or(trailing_str_val.len())
        + start_byte_offset;
    Some(subview(str_val, view, start_byte_offset, stop_byte_offset))
}

pub(super) fn substring(
    ca: &StringChunked,
    offset: &Int64Chunked,
    length: &UInt64Chunked,
) -> PolarsResult<StringChunked> {
    let result_len = match (ca.len(), offset.len(), length.len()) {
        (1, 1, len) => len,
        (len, 1, 1) => len,
        (1, len, 1) => len,
        (1, len_b, len_c) if len_b == len_c => len_b,
        (len_a, 1, len_c) if len_a == len_c => len_a,
        (len_a, len_b, 1) if len_a == len_b => len_a,
        (len_a, len_b, len_c) => {
            polars_ensure!(len_a == len_b, ComputeError: "offset's length: {} does not match that of the argument series: {}", len_b, len_a);
            polars_ensure!(len_a == len_c, ComputeError: "length's length: {} does not match that of the argument series: {}", len_c, len_a);
            len_a
        },
    };
    let mut builder = StringChunkedBuilder::new(ca.name(), result_len);
    for (((opt_s, (&view, buffers)), opt_offset), opt_length) in repeat(ca)
        .flat_map(iter_with_view_and_buffers)
        .zip(repeat(offset).flatten())
        .zip(repeat(length).flatten())
        .take(result_len)
    {
        if let Some(view) = substring_ternary(opt_s, view, opt_offset, opt_length) {
            builder.append_view(view, buffers);
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish())
}

pub(super) fn head(ca: &StringChunked, n: &Int64Chunked) -> PolarsResult<StringChunked> {
    let result_len = match (ca.len(), n.len()) {
        (1, len) => len,
        (len, 1) => len,
        (len_a, len_b) => {
            polars_ensure!(len_a == len_b, ComputeError: "n's length: {} does not match that of the argument series: {}", len_b, len_a);
            len_a
        },
    };
    let mut builder = StringChunkedBuilder::new(ca.name(), result_len);
    for ((opt_s, (&view, buffers)), opt_n) in repeat(ca)
        .flat_map(iter_with_view_and_buffers)
        .zip(repeat(n).flatten())
        .take(result_len)
    {
        if let Some(view) = head_binary(opt_s, view, opt_n) {
            builder.append_view(view, buffers);
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish())
}

pub(super) fn tail(ca: &StringChunked, n: &Int64Chunked) -> PolarsResult<StringChunked> {
    let result_len = match (ca.len(), n.len()) {
        (1, len) => len,
        (len, 1) => len,
        (len_a, len_b) => {
            polars_ensure!(len_a == len_b, ComputeError: "n's length: {} does not match that of the argument series: {}", len_b, len_a);
            len_a
        },
    };
    let mut builder = StringChunkedBuilder::new(ca.name(), result_len);
    for ((opt_s, (&view, buffers)), opt_n) in repeat(ca)
        .flat_map(iter_with_view_and_buffers)
        .zip(repeat(n).flatten())
        .take(result_len)
    {
        if let Some(view) = tail_binary(opt_s, view, opt_n) {
            builder.append_view(view, buffers);
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish())
}
