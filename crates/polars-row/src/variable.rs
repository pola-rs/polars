//! Variable length values are encoded as
//!
//! - single `0_u8` if null
//! - single `1_u8` if empty array
//! - `2_u8` if not empty, followed by one or more blocks
//!
//! where a block is encoded as
//!
//! - [`BLOCK_SIZE`] bytes of string data, padded with 0s
//! - `0xFF_u8` if this is not the last block for this string
//! - otherwise the length of the block as a `u8`

use std::mem::MaybeUninit;

use arrow::array::{BinaryViewArray, MutableBinaryViewArray, Utf8ViewArray};
use arrow::bitmap::MutableBitmap;
use polars_utils::slice::Slice2Uninit;

use crate::fixed::{decode_opt_nulls, get_null_sentinel};
use crate::EncodingField;

/// The block size of the variable length encoding
pub(crate) const BLOCK_SIZE: usize = 32;

/// The continuation token.
pub(crate) const BLOCK_CONTINUATION_TOKEN: u8 = 0xFF;

/// Indicates an empty string
pub(crate) const EMPTY_SENTINEL: u8 = 1;

/// Indicates a non-empty string
pub(crate) const NON_EMPTY_SENTINEL: u8 = 2;

/// Returns the ceil of `value`/`divisor`
#[inline]
pub fn ceil(value: usize, divisor: usize) -> usize {
    // Rewrite as `value.div_ceil(&divisor)` after
    // https://github.com/rust-lang/rust/issues/88581 is merged.
    value / divisor + (0 != value % divisor) as usize
}

#[inline]
fn padded_length(a: usize) -> usize {
    1 + ceil(a, BLOCK_SIZE) * (BLOCK_SIZE + 1)
}

#[inline]
pub fn encoded_len_from_len(a: Option<usize>, field: &EncodingField) -> usize {
    if field.no_order {
        4 + a.unwrap_or(0)
    } else {
        a.map_or(1, padded_length)
    }
}

#[inline]
pub fn encoded_str_len_from_len(a: Option<usize>, _field: &EncodingField) -> usize {
    // Length = 1                i.f.f. str is null or empty
    // Length = 1 + len(str) + 1 i.f.f. str is non-null and non-empty
    a.map_or(1, |v| 1 + v + usize::from(v > 0))
}

unsafe fn encode_one_str(
    out: &mut [MaybeUninit<u8>],
    val: Option<&str>,
    field: &EncodingField,
) -> usize {
    match val {
        None => {
            out[0] = MaybeUninit::new(get_null_sentinel(field));
            1
        },
        Some("") => {
            out[0] = MaybeUninit::new(field.empty_str_token());
            1
        },
        Some(s) => {
            out[0] = MaybeUninit::new(field.non_empty_str_token());

            // UTF-8 has 0xFF as an always invalid byte, so we could use this as a delimiter.
            // However, we want our terminating byte to always be smaller than any other character
            // so we add 1 to all bytes and make 0x00 our terminating byte.
            let t = if field.descending { 0xFF } else { 0x00 };
            for (i, &b) in s.as_bytes().iter().enumerate() {
                out[1 + i] = MaybeUninit::new(t ^ (b + 1));
            }
            out[1 + s.len()] = MaybeUninit::new(t);

            2 + s.len()
        },
    }
}

pub(crate) unsafe fn encode_str_iter<'a, I: Iterator<Item = Option<&'a str>>>(
    buffer: &mut [MaybeUninit<u8>],
    input: I,
    field: &EncodingField,
    offsets: &mut [usize],
) {
    for (offset, opt_value) in offsets.iter_mut().zip(input) {
        let dst = buffer.get_unchecked_mut(*offset..);
        let written_len = encode_one_str(dst, opt_value, field);
        *offset += written_len;
    }
}

unsafe fn encode_one_no_order(
    out: &mut [MaybeUninit<u8>],
    val: Option<&[MaybeUninit<u8>]>,
    field: &EncodingField,
) -> usize {
    debug_assert!(field.no_order);
    match val {
        Some(val) => {
            assert!(val.len() < u32::MAX as usize);
            let encoded_len = (val.len() as u32).to_le_bytes().map(MaybeUninit::new);
            std::ptr::copy_nonoverlapping(encoded_len.as_ptr(), out.as_mut_ptr(), 4);
            std::ptr::copy_nonoverlapping(val.as_ptr(), out.as_mut_ptr().add(4), val.len());
            4 + val.len()
        },
        None => {
            let sentinel = u32::MAX.to_le_bytes().map(MaybeUninit::new);
            std::ptr::copy_nonoverlapping(sentinel.as_ptr(), out.as_mut_ptr(), 4);
            4
        },
    }
}

/// Encode one strings/bytes object and return the written length.
///
/// # Safety
/// `out` must have allocated enough room
unsafe fn encode_one(
    out: &mut [MaybeUninit<u8>],
    val: Option<&[MaybeUninit<u8>]>,
    field: &EncodingField,
) -> usize {
    match val {
        Some([]) => {
            let byte = if field.descending {
                !EMPTY_SENTINEL
            } else {
                EMPTY_SENTINEL
            };
            *out.get_unchecked_mut(0) = MaybeUninit::new(byte);
            1
        },
        Some(val) => {
            let block_count = ceil(val.len(), BLOCK_SIZE);
            let end_offset = 1 + block_count * (BLOCK_SIZE + 1);

            let dst = out.get_unchecked_mut(..end_offset);

            // Write `2_u8` to demarcate as non-empty, non-null string
            *dst.get_unchecked_mut(0) = MaybeUninit::new(NON_EMPTY_SENTINEL);

            let src_chunks = val.chunks_exact(BLOCK_SIZE);
            let src_remainder = src_chunks.remainder();

            // + 1 is for the BLOCK CONTINUATION TOKEN
            let dst_chunks = dst.get_unchecked_mut(1..).chunks_exact_mut(BLOCK_SIZE + 1);

            for (src, dst) in src_chunks.zip(dst_chunks) {
                // we copy src.len() that leaves 1 bytes for the continuation tkn.
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), src.len());
                // Indicate that there are further blocks to follow
                *dst.get_unchecked_mut(BLOCK_SIZE) = MaybeUninit::new(BLOCK_CONTINUATION_TOKEN);
            }

            // exactly BLOCK_SIZE bytes
            // this means we only need to set the length
            // all other bytes are already initialized
            if src_remainder.is_empty() {
                // overwrite the latest continuation marker.
                // replace the "there is another block" with
                // "we are finished this, this is the length of this block"
                *dst.last_mut().unwrap_unchecked() = MaybeUninit::new(BLOCK_SIZE as u8);
            }
            // there are remainder bytes
            else {
                // get the last block
                let start_offset = 1 + (block_count - 1) * (BLOCK_SIZE + 1);
                let last_dst = dst.get_unchecked_mut(start_offset..);
                let n_bytes_to_write = src_remainder.len();

                std::ptr::copy_nonoverlapping(
                    src_remainder.as_ptr(),
                    last_dst.as_mut_ptr(),
                    n_bytes_to_write,
                );
                // write remainder as zeros
                last_dst
                    .get_unchecked_mut(n_bytes_to_write..last_dst.len() - 1)
                    .fill(MaybeUninit::new(0));
                *dst.last_mut().unwrap_unchecked() = MaybeUninit::new(src_remainder.len() as u8);
            }

            if field.descending {
                for byte in dst {
                    *byte = MaybeUninit::new(!byte.assume_init());
                }
            }
            end_offset
        },
        None => {
            *out.get_unchecked_mut(0) = MaybeUninit::new(get_null_sentinel(field));
            // // write remainder as zeros
            // out.get_unchecked_mut(1..).fill(MaybeUninit::new(0));
            1
        },
    }
}

pub(crate) unsafe fn encode_iter<'a, I: Iterator<Item = Option<&'a [u8]>>>(
    buffer: &mut [MaybeUninit<u8>],
    input: I,
    field: &EncodingField,
    row_starts: &mut [usize],
) {
    if field.no_order {
        for (offset, opt_value) in row_starts.iter_mut().zip(input) {
            let dst = buffer.get_unchecked_mut(*offset..);
            let written_len = encode_one_no_order(dst, opt_value.map(|v| v.as_uninit()), field);
            *offset += written_len;
        }
    } else {
        for (offset, opt_value) in row_starts.iter_mut().zip(input) {
            let dst = buffer.get_unchecked_mut(*offset..);
            let written_len = encode_one(dst, opt_value.map(|v| v.as_uninit()), field);
            *offset += written_len;
        }
    }
}

pub(crate) unsafe fn encoded_str_len(row: &[u8], null_sentinel: u8, descending: bool) -> usize {
    // null
    if *row.get_unchecked(0) == null_sentinel {
        return 1;
    }

    let end = if descending {
        row[1..].iter().position(|&b| b == 0xFF)
    } else {
        row[1..].iter().position(|&b| b == 0x00)
    }
    .unwrap();

    end + 2
}

pub(crate) unsafe fn encoded_item_len(
    row: &[u8],
    non_empty_sentinel: u8,
    continuation_token: u8,
) -> usize {
    // empty or null
    if *row.get_unchecked(0) != non_empty_sentinel {
        return 1;
    }

    let mut idx = 1;
    loop {
        let sentinel = *row.get_unchecked(idx + BLOCK_SIZE);
        if sentinel == continuation_token {
            idx += BLOCK_SIZE + 1;
            continue;
        }
        return idx + BLOCK_SIZE + 1;
    }
}

unsafe fn decoded_len(
    row: &[u8],
    non_empty_sentinel: u8,
    continuation_token: u8,
    descending: bool,
) -> usize {
    // empty or null
    if *row.get_unchecked(0) != non_empty_sentinel {
        return 0;
    }

    let mut str_len = 0;
    let mut idx = 1;
    loop {
        let sentinel = *row.get_unchecked(idx + BLOCK_SIZE);
        if sentinel == continuation_token {
            idx += BLOCK_SIZE + 1;
            str_len += BLOCK_SIZE;
            continue;
        }
        // the sentinel of the last block has the length
        // of that block. The rest is padding.
        let block_length = if descending {
            // all bits were inverted on encoding
            !sentinel
        } else {
            sentinel
        };
        return str_len + block_length as usize;
    }
}

unsafe fn decoded_len_unordered(row: &[u8]) -> Option<u32> {
    let len = u32::from_le_bytes(row.get_unchecked(0..4).try_into().unwrap());
    Some(len).filter(|l| *l < u32::MAX)
}

unsafe fn decode_binview_unordered(rows: &mut [&[u8]]) -> BinaryViewArray {
    let mut mutable = MutableBinaryViewArray::with_capacity(rows.len());
    for row in rows.iter_mut() {
        if let Some(len) = decoded_len_unordered(row) {
            mutable.push_value(row.get_unchecked(4..4 + len as usize));
            *row = row.get_unchecked(4 + len as usize..);
        } else {
            mutable.push_null();
        }
    }
    mutable.freeze()
}

pub(super) unsafe fn decode_binview(rows: &mut [&[u8]], field: &EncodingField) -> BinaryViewArray {
    if field.no_order {
        return decode_binview_unordered(rows);
    }

    let (non_empty_sentinel, continuation_token) = if field.descending {
        (!NON_EMPTY_SENTINEL, !BLOCK_CONTINUATION_TOKEN)
    } else {
        (NON_EMPTY_SENTINEL, BLOCK_CONTINUATION_TOKEN)
    };

    let null_sentinel = get_null_sentinel(field);
    let validity = decode_opt_nulls(rows, null_sentinel);

    let mut mutable = MutableBinaryViewArray::with_capacity(rows.len());

    let mut scratch = vec![];
    for row in rows {
        scratch.set_len(0);
        let str_len = decoded_len(
            row,
            non_empty_sentinel,
            continuation_token,
            field.descending,
        );
        let mut to_read = str_len;
        // we start at one, as we skip the validity byte
        let mut offset = 1;

        while to_read >= BLOCK_SIZE {
            to_read -= BLOCK_SIZE;
            scratch.extend_from_slice(row.get_unchecked(offset..offset + BLOCK_SIZE));
            offset += BLOCK_SIZE + 1;
        }

        if to_read != 0 {
            scratch.extend_from_slice(row.get_unchecked(offset..offset + to_read));
            offset += BLOCK_SIZE + 1;
        }
        *row = row.get_unchecked(offset..);

        if field.descending {
            scratch.iter_mut().for_each(|o| *o = !*o)
        }
        mutable.push_value_ignore_validity(&scratch);
    }

    let out: BinaryViewArray = mutable.into();
    out.with_validity(validity)
}

pub(super) unsafe fn decode_strview(rows: &mut [&[u8]], field: &EncodingField) -> Utf8ViewArray {
    let null_sentinel = get_null_sentinel(field);
    let empty_str_token = field.empty_str_token();
    let non_empty_str_token = field.non_empty_str_token();

    let num_rows = rows.len();
    let mut mutable = MutableBinaryViewArray::<str>::with_capacity(rows.len());

    // @TODO: 2 loop system
    let mut validity = MutableBitmap::new();

    let mut scratch = Vec::new();
    for row in rows.iter_mut() {
        let sentinel = row[0];
        *row = &row[1..];

        if sentinel == null_sentinel {
            validity.reserve(num_rows);
            validity.extend_constant(mutable.len(), true);
            validity.push(false);
            mutable.push_value_ignore_validity("");
            break;
        }

        if sentinel == empty_str_token {
            mutable.push_value_ignore_validity("");
            continue;
        }

        scratch.clear();
        debug_assert_eq!(sentinel, non_empty_str_token);
        if field.descending {
            scratch.extend(row.iter().take_while(|&b| *b != 0xFF).map(|&v| !v - 1));
        } else {
            scratch.extend(row.iter().take_while(|&b| *b != 0x00).map(|&v| v - 1));
        }

        *row = &row[1 + scratch.len()..];
        mutable.push_value_ignore_validity(unsafe { std::str::from_utf8_unchecked(&scratch) });
    }

    if mutable.len() == num_rows {
        return mutable.into();
    }

    for row in rows[mutable.len()..].iter_mut() {
        let sentinel = row[0];
        *row = &row[1..];

        validity.push(sentinel != null_sentinel);

        if sentinel == null_sentinel || sentinel == empty_str_token {
            mutable.push_value_ignore_validity("");
            continue;
        }

        scratch.clear();
        debug_assert_eq!(sentinel, non_empty_str_token);
        if field.descending {
            scratch.extend(row.iter().take_while(|&b| *b != 0xFF).map(|&v| !v - 1));
        } else {
            scratch.extend(row.iter().take_while(|&b| *b != 0x00).map(|&v| v - 1));
        }

        *row = &row[1 + scratch.len()..];
        mutable.push_value_ignore_validity(unsafe { std::str::from_utf8_unchecked(&scratch) });
    }

    let out: Utf8ViewArray = mutable.into();
    out.with_validity(Some(validity.freeze()))
}
