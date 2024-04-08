use polars_core::prelude::arity::{binary_elementwise, ternary_elementwise, unary_elementwise};
use polars_core::prelude::{Int64Chunked, StringChunked, UInt64Chunked};

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
        // TODO: Clippy lint is broken, remove attr once fixed.
        // https://github.com/rust-lang/rust-clippy/issues/12580
        #[cfg_attr(feature = "nightly", allow(clippy::manual_unwrap_or_default))]
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
            // SAFETY: index `0` is in bound.
            let str_val = unsafe { ca.get_unchecked(0) };
            // SAFETY: index `0` is in bound.
            let offset = unsafe { offset.get_unchecked(0) };
            unary_elementwise(length, |length| substring_ternary(str_val, offset, length))
                .with_name(ca.name())
        },
        (_, 1, 1) => {
            // SAFETY: index `0` is in bound.
            let offset = unsafe { offset.get_unchecked(0) };
            // SAFETY: index `0` is in bound.
            let length = unsafe { length.get_unchecked(0) };
            unary_elementwise(ca, |str_val| substring_ternary(str_val, offset, length))
        },
        (1, _, 1) => {
            // SAFETY: index `0` is in bound.
            let str_val = unsafe { ca.get_unchecked(0) };
            // SAFETY: index `0` is in bound.
            let length = unsafe { length.get_unchecked(0) };
            unary_elementwise(offset, |offset| substring_ternary(str_val, offset, length))
                .with_name(ca.name())
        },
        (1, len_b, len_c) if len_b == len_c => {
            // SAFETY: index `0` is in bound.
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
