//! General utilities for bitmaps representing items where LSB is the first item.
mod chunk_iterator;
mod chunks_exact_mut;
mod fmt;
mod iterator;
mod slice_iterator;
mod zip_validity;

pub(crate) use chunk_iterator::merge_reversed;
pub use chunk_iterator::{BitChunk, BitChunkIterExact, BitChunks, BitChunksExact};
pub use chunks_exact_mut::BitChunksExactMut;
pub use fmt::fmt;
pub use iterator::BitmapIter;
use polars_utils::slice::GetSaferUnchecked;
pub use slice_iterator::SlicesIterator;
pub use zip_validity::{ZipValidity, ZipValidityIter};

/// Returns whether bit at position `i` in `byte` is set or not
#[inline]
pub fn is_set(byte: u8, i: usize) -> bool {
    debug_assert!(i < 8);
    byte & (1 << i) != 0
}

/// Sets bit at position `i` in `byte`.
#[inline]
pub fn set(byte: u8, i: usize, value: bool) -> u8 {
    debug_assert!(i < 8);

    let mask = !(1 << i);
    let insert = (value as u8) << i;
    (byte & mask) | insert
}

/// Sets bit at position `i` in `bytes`.
/// # Panics
/// This function panics iff `i >= bytes.len() * 8`.
#[inline]
pub fn set_bit(bytes: &mut [u8], i: usize, value: bool) {
    bytes[i / 8] = set(bytes[i / 8], i % 8, value);
}

/// Sets bit at position `i` in `bytes` without doing bound checks
/// # Safety
/// `i >= bytes.len() * 8` results in undefined behavior.
#[inline]
pub unsafe fn set_bit_unchecked(bytes: &mut [u8], i: usize, value: bool) {
    let byte = bytes.get_unchecked_mut(i / 8);
    *byte = set(*byte, i % 8, value);
}

/// Returns whether bit at position `i` in `bytes` is set.
/// # Panic
/// This function panics iff `i >= bytes.len() * 8`.
#[inline]
pub fn get_bit(bytes: &[u8], i: usize) -> bool {
    let byte = bytes[i / 8];
    let bit = (byte >> (i % 8)) & 1;
    bit != 0
}

/// Returns whether bit at position `i` in `bytes` is set or not.
///
/// # Safety
/// `i >= bytes.len() * 8` results in undefined behavior.
#[inline]
pub unsafe fn get_bit_unchecked(bytes: &[u8], i: usize) -> bool {
    let byte = *bytes.get_unchecked_release(i / 8);
    let bit = (byte >> (i % 8)) & 1;
    bit != 0
}

/// Returns the number of bytes required to hold `bits` bits.
#[inline]
pub fn bytes_for(bits: usize) -> usize {
    bits.saturating_add(7) / 8
}

/// Returns the number of zero bits in the slice offsetted by `offset` and a length of `length`.
/// # Panics
/// This function panics iff `(offset + len).saturating_add(7) / 8 >= slice.len()`
/// because it corresponds to the situation where `len` is beyond bounds.
pub fn count_zeros(mut slice: &[u8], mut offset: usize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }

    // Reduce the slice only to relevant bytes.
    let first_byte_idx = offset / 8;
    let last_byte_idx = (offset + len - 1) / 8;
    slice = &slice[first_byte_idx..=last_byte_idx];
    offset %= 8;

    // Fast path for single u64.
    if slice.len() <= 8 {
        let mut tmp = [0u8; 8];
        tmp[..slice.len()].copy_from_slice(slice);
        let word = u64::from_ne_bytes(tmp) >> offset;
        let masked = word << (64 - len);
        return len - masked.count_ones() as usize;
    }

    let mut len_uncounted = len;
    let mut num_ones = 0;

    // Handle first partial byte.
    if offset != 0 {
        let partial_byte;
        (partial_byte, slice) = slice.split_first().unwrap();
        num_ones += (partial_byte >> offset).count_ones() as usize;
        len_uncounted -= 8 - offset;
    }

    // Handle last partial byte.
    let final_partial_len = len_uncounted % 8;
    if final_partial_len != 0 {
        let partial_byte;
        (partial_byte, slice) = slice.split_last().unwrap();
        let masked = partial_byte << (8 - final_partial_len);
        num_ones += masked.count_ones() as usize;
    }

    // SAFETY: transmuting u8 to u64 is fine.
    let (start, mid, end) = unsafe { slice.align_to::<u64>() };

    // Handle unaligned ends.
    let mut tmp = [0u8; 8];
    tmp[..start.len()].copy_from_slice(start);
    num_ones += u64::from_ne_bytes(tmp).count_ones() as usize;
    tmp = [0u8; 8];
    tmp[..end.len()].copy_from_slice(end);
    num_ones += u64::from_ne_bytes(tmp).count_ones() as usize;

    // Handle the bulk.
    num_ones += mid
        .iter()
        .copied()
        .map(|w| w.count_ones() as usize)
        .sum::<usize>();

    len - num_ones
}

/// Takes the given slice of bytes plus a bit offset and bit length and returns
/// the slice so that it starts at a byte-aligned boundary.
///
/// Returns (in order):
///  - the bits of the first byte if it isn't a full byte
///  - the number of bits in the first partial byte
///  - the rest of the bits as a byteslice
///  - the number of bits in the byteslice
#[inline]
pub fn align_bitslice_start_u8(
    slice: &[u8],
    offset: usize,
    len: usize,
) -> (u8, usize, &[u8], usize) {
    if len == 0 {
        return (0, 0, &[], 0);
    }

    // Protects the below get_uncheckeds.
    assert!(slice.len() * 8 >= offset + len);

    let mut first_byte_idx = offset / 8;
    let partial_offset = offset % 8;
    let bits_in_partial_byte = (8 - partial_offset).min(len) % 8;
    let mut partial_byte = unsafe { *slice.get_unchecked(first_byte_idx) };
    partial_byte >>= partial_offset;
    partial_byte &= (1 << bits_in_partial_byte) - 1;
    first_byte_idx += (partial_offset > 0) as usize;

    let rest_slice = unsafe { slice.get_unchecked(first_byte_idx..) };
    (
        partial_byte,
        bits_in_partial_byte,
        rest_slice,
        len - bits_in_partial_byte,
    )
}
