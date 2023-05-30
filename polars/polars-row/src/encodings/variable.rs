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

use polars_utils::slice::GetSaferUnchecked;

use crate::encodings::fixed::null_sentinel;
use crate::row::RowsEncoded;
use crate::SortField;

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
pub fn encoded_len(a: Option<&[u8]>) -> usize {
    padded_length_opt(a.map(|v| v.len()))
}

/// Encode one strings/bytes object and return the written length.
///
/// # Safety
/// `out` must have allocated enough room
unsafe fn encode_one(out: &mut [u8], val: Option<&[u8]>, field: &SortField) -> usize {
    match val {
        Some(val) if val.is_empty() => {
            let byte = if field.descending {
                !EMPTY_SENTINEL
            } else {
                EMPTY_SENTINEL
            };
            *out.get_unchecked_release_mut(0) = byte;
            1
        }
        Some(val) => {
            let block_count = ceil(val.len(), BLOCK_SIZE);
            let end_offset = 1 + block_count * (BLOCK_SIZE + 1);

            let dst = out.get_unchecked_release_mut(..end_offset);

            // Write `2_u8` to demarcate as non-empty, non-null string
            *dst.get_unchecked_release_mut(0) = NON_EMPTY_SENTINEL;

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
                *dst.get_unchecked_release_mut(BLOCK_SIZE) = BLOCK_CONTINUATION_TOKEN;
            }

            if src_remainder.is_empty() {
                // overwrite the latest continuation marker.
                // replace the "there is another block" with
                // "we are finished this, this is the length of this block"
                *dst.last_mut().unwrap_unchecked() = BLOCK_SIZE as u8;
            } else {
                // get the last block
                let start_offset = 1 + (block_count - 1) * (BLOCK_SIZE + 1);
                let last_dst = dst.get_unchecked_release_mut(start_offset..);
                std::ptr::copy_nonoverlapping(
                    src_remainder.as_ptr(),
                    last_dst.as_mut_ptr(),
                    src_remainder.len(),
                );
                *dst.last_mut().unwrap_unchecked() = src_remainder.len() as u8;
            }

            if field.descending {
                for byte in dst {
                    *byte = !*byte;
                }
            }
            end_offset
        }
        None => {
            *out.get_unchecked_release_mut(0) = null_sentinel(field);
            1
        }
    }
}
pub(crate) unsafe fn encode_iter<'a, I: Iterator<Item = Option<&'a [u8]>>>(
    input: I,
    out: &mut RowsEncoded,
    field: &SortField,
) {
    for (offset, opt_value) in out.offsets.iter_mut().skip(1).zip(input) {
        let dst = out.buf.get_unchecked_release_mut(*offset..);
        let written_len = encode_one(dst, opt_value, field);
        *offset += written_len;
    }
}
