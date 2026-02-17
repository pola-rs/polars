use std::cmp::Ordering;

use polars_core::prelude::arity::{binary_elementwise, ternary_elementwise, unary_elementwise};
use polars_core::prelude::{BinaryChunked, ChunkFullNull, DataType, Int64Chunked, UInt64Chunked};
use polars_error::{PolarsResult, polars_ensure};

fn head_binary(opt_bytes: Option<&[u8]>, opt_n: Option<i64>) -> Option<&[u8]> {
    if let (Some(bytes), Some(n)) = (opt_bytes, opt_n) {
        let end_idx = head_binary_values(bytes, n);
        Some(&bytes[..end_idx])
    } else {
        None
    }
}

fn head_binary_values(bytes: &[u8], n: i64) -> usize {
    match n.cmp(&0) {
        Ordering::Equal => 0,
        Ordering::Greater => {
            // Take first n bytes
            std::cmp::min(n as usize, bytes.len())
        },
        Ordering::Less => {
            // End n bytes from the end
            bytes.len().saturating_sub((-n) as usize)
        },
    }
}

fn tail_binary(opt_bytes: Option<&[u8]>, opt_n: Option<i64>) -> Option<&[u8]> {
    if let (Some(bytes), Some(n)) = (opt_bytes, opt_n) {
        let start_idx = tail_binary_values(bytes, n);
        Some(&bytes[start_idx..])
    } else {
        None
    }
}

fn tail_binary_values(bytes: &[u8], n: i64) -> usize {
    let max_len = bytes.len();

    match n.cmp(&0) {
        Ordering::Equal => max_len,
        Ordering::Greater => {
            // Start from nth byte from the end
            max_len.saturating_sub(n as usize)
        },
        Ordering::Less => {
            // Start after the nth byte
            std::cmp::min((-n) as usize, max_len)
        },
    }
}

fn slice_ternary_offsets(
    opt_bytes: Option<&[u8]>,
    opt_offset: Option<i64>,
    opt_length: Option<u64>,
) -> Option<(usize, usize)> {
    let bytes = opt_bytes?;
    let offset = opt_offset?;
    Some(slice_ternary_offsets_value(
        bytes,
        offset,
        opt_length.unwrap_or(u64::MAX),
    ))
}

pub fn slice_ternary_offsets_value(bytes: &[u8], offset: i64, length: u64) -> (usize, usize) {
    // Fast-path: always empty slice
    if length == 0 || offset >= bytes.len() as i64 {
        return (0, 0);
    }

    let start_byte_offset = if offset >= 0 {
        std::cmp::min(offset as usize, bytes.len())
    } else {
        // If `offset` is negative, it counts from the end
        let abs_offset = (-offset) as usize;
        if abs_offset > bytes.len() {
            // Offset is before the start - handle length reduction
            let length_reduction = abs_offset - bytes.len();
            let adjusted_length = (length as usize).saturating_sub(length_reduction);
            return (0, std::cmp::min(adjusted_length, bytes.len()));
        }
        bytes.len() - abs_offset
    };

    let remaining = bytes.len() - start_byte_offset;
    let end_byte_offset = start_byte_offset + std::cmp::min(length as usize, remaining);

    (start_byte_offset, end_byte_offset)
}

fn slice_ternary(
    opt_bytes: Option<&[u8]>,
    opt_offset: Option<i64>,
    opt_length: Option<u64>,
) -> Option<&[u8]> {
    let (start, end) = slice_ternary_offsets(opt_bytes, opt_offset, opt_length)?;
    opt_bytes.map(|bytes| &bytes[start..end])
}

pub(super) fn slice(
    ca: &BinaryChunked,
    offset: &Int64Chunked,
    length: &UInt64Chunked,
) -> BinaryChunked {
    match (ca.len(), offset.len(), length.len()) {
        (1, 1, _) => {
            let bytes = ca.get(0);
            let offset = offset.get(0);
            unary_elementwise(length, |length| slice_ternary(bytes, offset, length))
                .with_name(ca.name().clone())
        },
        (_, 1, 1) => {
            let offset = offset.get(0);
            let length = length.get(0).unwrap_or(u64::MAX);

            let Some(offset) = offset else {
                return BinaryChunked::full_null(ca.name().clone(), ca.len());
            };

            ca.apply_nonnull_values_generic(DataType::Binary, |val| {
                let (start, end) = slice_ternary_offsets_value(val, offset, length);
                &val[start..end]
            })
        },
        (1, _, 1) => {
            let bytes = ca.get(0);
            let length = length.get(0);
            unary_elementwise(offset, |offset| slice_ternary(bytes, offset, length))
                .with_name(ca.name().clone())
        },
        (1, len_b, len_c) if len_b == len_c => {
            let bytes = ca.get(0);
            binary_elementwise(offset, length, |offset, length| {
                slice_ternary(bytes, offset, length)
            })
        },
        (len_a, 1, len_c) if len_a == len_c => {
            fn infer<F: for<'a> FnMut(Option<&'a [u8]>, Option<u64>) -> Option<&'a [u8]>>(
                f: F,
            ) -> F {
                f
            }
            let offset = offset.get(0);
            binary_elementwise(
                ca,
                length,
                infer(|bytes, length| slice_ternary(bytes, offset, length)),
            )
        },
        (len_a, len_b, 1) if len_a == len_b => {
            fn infer<F: for<'a> FnMut(Option<&'a [u8]>, Option<i64>) -> Option<&'a [u8]>>(
                f: F,
            ) -> F {
                f
            }
            let length = length.get(0);
            binary_elementwise(
                ca,
                offset,
                infer(|bytes, offset| slice_ternary(bytes, offset, length)),
            )
        },
        _ => ternary_elementwise(ca, offset, length, slice_ternary),
    }
}

pub(super) fn head(ca: &BinaryChunked, n: &Int64Chunked) -> PolarsResult<BinaryChunked> {
    match (ca.len(), n.len()) {
        (len, 1) => {
            let n = n.get(0);
            let Some(n) = n else {
                return Ok(BinaryChunked::full_null(ca.name().clone(), len));
            };

            Ok(ca.apply_nonnull_values_generic(DataType::Binary, |val| {
                let end = head_binary_values(val, n);
                &val[..end]
            }))
        },
        (1, _) => {
            let bytes = ca.get(0);
            Ok(unary_elementwise(n, |n| head_binary(bytes, n)).with_name(ca.name().clone()))
        },
        (a, b) => {
            polars_ensure!(a == b, ShapeMismatch: "lengths of arguments do not align in 'bin.head' got length: {} for column: {}, got length: {} for argument 'n'", a, ca.name(), b);
            Ok(binary_elementwise(ca, n, head_binary))
        },
    }
}

pub(super) fn tail(ca: &BinaryChunked, n: &Int64Chunked) -> PolarsResult<BinaryChunked> {
    Ok(match (ca.len(), n.len()) {
        (len, 1) => {
            let n = n.get(0);
            let Some(n) = n else {
                return Ok(BinaryChunked::full_null(ca.name().clone(), len));
            };

            ca.apply_nonnull_values_generic(DataType::Binary, |val| {
                let start = tail_binary_values(val, n);
                &val[start..]
            })
        },
        (1, _) => {
            let bytes = ca.get(0);
            unary_elementwise(n, |n| tail_binary(bytes, n)).with_name(ca.name().clone())
        },
        (a, b) => {
            polars_ensure!(a == b, ShapeMismatch: "lengths of arguments do not align in 'bin.tail' got length: {} for column: {}, got length: {} for argument 'n'", a, ca.name(), b);
            binary_elementwise(ca, n, tail_binary)
        },
    })
}
