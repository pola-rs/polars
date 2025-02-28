/// Row encoding for UTF-8 strings
///
/// This encoding is based on the fact that in UTF-8 the bytes 0xFC - 0xFF are never valid bytes.
/// To make this work with the row encoding, we add 2 to each byte which gives us two bytes which
/// never occur in UTF-8 before and after the possible byte range. The values 0x00 and 0xFF are
/// reserved for the null sentinel. The values 0x01 and 0xFE are reserved as a sequence terminator
/// byte.
///
/// This allows the string row encoding to have a constant 1 byte overhead.
use std::mem::MaybeUninit;

use arrow::array::{MutableBinaryViewArray, Utf8ViewArray};
use arrow::bitmap::BitmapBuilder;

use crate::row::RowEncodingOptions;

#[inline]
pub fn len_from_item(a: Option<usize>, _opt: RowEncodingOptions) -> usize {
    // Length = 1                i.f.f. str is null
    // Length = len(str) + 1     i.f.f. str is non-null
    1 + a.unwrap_or_default()
}

pub unsafe fn len_from_buffer(row: &[u8], opt: RowEncodingOptions) -> usize {
    // null
    if *row.get_unchecked(0) == opt.null_sentinel() {
        return 1;
    }

    let end = if opt.contains(RowEncodingOptions::DESCENDING) {
        unsafe { row.iter().position(|&b| b == 0xFE).unwrap_unchecked() }
    } else {
        unsafe { row.iter().position(|&b| b == 0x01).unwrap_unchecked() }
    };

    end + 1
}

pub unsafe fn encode_str<'a, I: Iterator<Item = Option<&'a str>>>(
    buffer: &mut [MaybeUninit<u8>],
    input: I,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    let null_sentinel = opt.null_sentinel();
    let t = if opt.contains(RowEncodingOptions::DESCENDING) {
        0xFF
    } else {
        0x00
    };

    for (offset, opt_value) in offsets.iter_mut().zip(input) {
        let dst = buffer.get_unchecked_mut(*offset..);

        match opt_value {
            None => {
                *unsafe { dst.get_unchecked_mut(0) } = MaybeUninit::new(null_sentinel);
                *offset += 1;
            },
            Some(s) => {
                for (i, &b) in s.as_bytes().iter().enumerate() {
                    *unsafe { dst.get_unchecked_mut(i) } = MaybeUninit::new(t ^ (b + 2));
                }
                *unsafe { dst.get_unchecked_mut(s.len()) } = MaybeUninit::new(t ^ 0x01);
                *offset += 1 + s.len();
            },
        }
    }
}

pub unsafe fn decode_str(rows: &mut [&[u8]], opt: RowEncodingOptions) -> Utf8ViewArray {
    let null_sentinel = opt.null_sentinel();
    let descending = opt.contains(RowEncodingOptions::DESCENDING);

    let num_rows = rows.len();
    let mut array = MutableBinaryViewArray::<str>::with_capacity(rows.len());

    let mut scratch = Vec::new();
    for row in rows.iter_mut() {
        let sentinel = *unsafe { row.get_unchecked(0) };
        if sentinel == null_sentinel {
            *row = unsafe { row.get_unchecked(1..) };
            break;
        }

        scratch.clear();
        if descending {
            scratch.extend(row.iter().take_while(|&b| *b != 0xFE).map(|&v| !v - 2));
        } else {
            scratch.extend(row.iter().take_while(|&b| *b != 0x01).map(|&v| v - 2));
        }

        *row = row.get_unchecked(1 + scratch.len()..);
        array.push_value_ignore_validity(unsafe { std::str::from_utf8_unchecked(&scratch) });
    }

    if array.len() == num_rows {
        return array.into();
    }

    let mut validity = BitmapBuilder::with_capacity(num_rows);
    validity.extend_constant(array.len(), true);
    validity.push(false);
    array.push_value_ignore_validity("");

    for row in rows[array.len()..].iter_mut() {
        let sentinel = *unsafe { row.get_unchecked(0) };
        validity.push(sentinel != null_sentinel);
        if sentinel == null_sentinel {
            *row = unsafe { row.get_unchecked(1..) };
            array.push_value_ignore_validity("");
            continue;
        }

        scratch.clear();
        if descending {
            scratch.extend(row.iter().take_while(|&b| *b != 0xFE).map(|&v| !v - 2));
        } else {
            scratch.extend(row.iter().take_while(|&b| *b != 0x01).map(|&v| v - 2));
        }

        *row = row.get_unchecked(1 + scratch.len()..);
        array.push_value_ignore_validity(unsafe { std::str::from_utf8_unchecked(&scratch) });
    }

    let out: Utf8ViewArray = array.into();
    out.with_validity(validity.into_opt_validity())
}
