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

use arrow::array::{BinaryArray, BinaryViewArray, MutableBinaryViewArray};
use arrow::datatypes::ArrowDataType;
use arrow::offset::Offsets;
use polars_utils::slice::{GetSaferUnchecked, Slice2Uninit};

use crate::fixed::{decode_nulls, get_null_sentinel};
use crate::row::RowsEncoded;
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
fn padded_length_opt(a: Option<usize>) -> usize {
    if let Some(a) = a {
        padded_length(a)
    } else {
        1
    }
}

#[inline]
fn length_opt(a: Option<usize>) -> usize {
    if let Some(a) = a {
        1 + a
    } else {
        1
    }
}

#[inline]
pub fn encoded_len(a: Option<&[u8]>, field: &EncodingField) -> usize {
    if field.no_order {
        length_opt(a.map(|v| v.len()))
    } else {
        padded_length_opt(a.map(|v| v.len()))
    }
}

unsafe fn encode_one_no_order(
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
            *out.get_unchecked_release_mut(0) = MaybeUninit::new(byte);
            1
        },
        Some(val) => {
            let end_offset = 1 + val.len();

            // Write `2_u8` to demarcate as non-empty, non-null string
            *out.get_unchecked_release_mut(0) = MaybeUninit::new(NON_EMPTY_SENTINEL);
            std::ptr::copy_nonoverlapping(val.as_ptr(), out.as_mut_ptr().add(1), val.len());

            end_offset
        },
        None => {
            *out.get_unchecked_release_mut(0) = MaybeUninit::new(get_null_sentinel(field));
            // // write remainder as zeros
            // out.get_unchecked_release_mut(1..).fill(MaybeUninit::new(0));
            1
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
            *out.get_unchecked_release_mut(0) = MaybeUninit::new(byte);
            1
        },
        Some(val) => {
            let block_count = ceil(val.len(), BLOCK_SIZE);
            let end_offset = 1 + block_count * (BLOCK_SIZE + 1);

            let dst = out.get_unchecked_release_mut(..end_offset);

            // Write `2_u8` to demarcate as non-empty, non-null string
            *dst.get_unchecked_release_mut(0) = MaybeUninit::new(NON_EMPTY_SENTINEL);

            let src_chunks = val.chunks_exact(BLOCK_SIZE);
            let src_remainder = src_chunks.remainder();

            // + 1 is for the BLOCK CONTINUATION TOKEN
            let dst_chunks = dst
                .get_unchecked_release_mut(1..)
                .chunks_exact_mut(BLOCK_SIZE + 1);

            for (src, dst) in src_chunks.zip(dst_chunks) {
                // we copy src.len() that leaves 1 bytes for the continuation tkn.
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), src.len());
                // Indicate that there are further blocks to follow
                *dst.get_unchecked_release_mut(BLOCK_SIZE) =
                    MaybeUninit::new(BLOCK_CONTINUATION_TOKEN);
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
                let last_dst = dst.get_unchecked_release_mut(start_offset..);
                let n_bytes_to_write = src_remainder.len();

                std::ptr::copy_nonoverlapping(
                    src_remainder.as_ptr(),
                    last_dst.as_mut_ptr(),
                    n_bytes_to_write,
                );
                // write remainder as zeros
                last_dst
                    .get_unchecked_release_mut(n_bytes_to_write..last_dst.len() - 1)
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
            *out.get_unchecked_release_mut(0) = MaybeUninit::new(get_null_sentinel(field));
            // // write remainder as zeros
            // out.get_unchecked_release_mut(1..).fill(MaybeUninit::new(0));
            1
        },
    }
}
pub(crate) unsafe fn encode_iter<'a, I: Iterator<Item = Option<&'a [u8]>>>(
    input: I,
    out: &mut RowsEncoded,
    field: &EncodingField,
) {
    out.values.set_len(0);
    let values = out.values.spare_capacity_mut();

    if field.no_order {
        for (offset, opt_value) in out.offsets.iter_mut().skip(1).zip(input) {
            let dst = values.get_unchecked_release_mut(*offset..);
            let written_len = encode_one_no_order(dst, opt_value.map(|v| v.as_uninit()), field);
            *offset += written_len;
        }
    } else {
        for (offset, opt_value) in out.offsets.iter_mut().skip(1).zip(input) {
            let dst = values.get_unchecked_release_mut(*offset..);
            let written_len = encode_one(dst, opt_value.map(|v| v.as_uninit()), field);
            *offset += written_len;
        }
    }
    let offset = out.offsets.last().unwrap();
    let dst = values.get_unchecked_release_mut(*offset..);
    // write remainder as zeros
    dst.fill(MaybeUninit::new(0));
    out.values.set_len(out.values.capacity())
}

unsafe fn has_nulls(rows: &[&[u8]], null_sentinel: u8) -> bool {
    rows.iter()
        .any(|row| *row.get_unchecked(0) == null_sentinel)
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

pub(super) unsafe fn decode_binary(rows: &mut [&[u8]], field: &EncodingField) -> BinaryArray<i64> {
    let (non_empty_sentinel, continuation_token) = if field.descending {
        (!NON_EMPTY_SENTINEL, !BLOCK_CONTINUATION_TOKEN)
    } else {
        (NON_EMPTY_SENTINEL, BLOCK_CONTINUATION_TOKEN)
    };

    let null_sentinel = get_null_sentinel(field);
    let validity = if has_nulls(rows, null_sentinel) {
        Some(decode_nulls(rows, null_sentinel))
    } else {
        None
    };
    let values_cap = rows
        .iter()
        .map(|row| {
            decoded_len(
                row,
                non_empty_sentinel,
                continuation_token,
                field.descending,
            )
        })
        .sum();
    let mut values = Vec::with_capacity(values_cap);
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    offsets.push(0);

    for row in rows {
        // TODO: cache the string lengths in a scratch? We just computed them above.
        let str_len = decoded_len(
            row,
            non_empty_sentinel,
            continuation_token,
            field.descending,
        );
        let values_offset = values.len();

        let mut to_read = str_len;
        // we start at one, as we skip the validity byte
        let mut offset = 1;

        while to_read >= BLOCK_SIZE {
            to_read -= BLOCK_SIZE;
            values.extend_from_slice(row.get_unchecked_release(offset..offset + BLOCK_SIZE));
            offset += BLOCK_SIZE + 1;
        }

        if to_read != 0 {
            values.extend_from_slice(row.get_unchecked_release(offset..offset + to_read));
            offset += BLOCK_SIZE + 1;
        }
        *row = row.get_unchecked(offset..);
        offsets.push(values.len() as i64);

        if field.descending {
            values
                .get_unchecked_release_mut(values_offset..)
                .iter_mut()
                .for_each(|o| *o = !*o)
        }
    }

    BinaryArray::new(
        ArrowDataType::LargeBinary,
        Offsets::new_unchecked(offsets).into(),
        values.into(),
        validity,
    )
}

pub(super) unsafe fn decode_binview(rows: &mut [&[u8]], field: &EncodingField) -> BinaryViewArray {
    let (non_empty_sentinel, continuation_token) = if field.descending {
        (!NON_EMPTY_SENTINEL, !BLOCK_CONTINUATION_TOKEN)
    } else {
        (NON_EMPTY_SENTINEL, BLOCK_CONTINUATION_TOKEN)
    };

    let null_sentinel = get_null_sentinel(field);
    let validity = if has_nulls(rows, null_sentinel) {
        Some(decode_nulls(rows, null_sentinel))
    } else {
        None
    };

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
            scratch.extend_from_slice(row.get_unchecked_release(offset..offset + BLOCK_SIZE));
            offset += BLOCK_SIZE + 1;
        }

        if to_read != 0 {
            scratch.extend_from_slice(row.get_unchecked_release(offset..offset + to_read));
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
