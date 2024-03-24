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
use polars_utils::slice::{load_padded_le_u64, GetSaferUnchecked};
pub use slice_iterator::SlicesIterator;
pub use zip_validity::{ZipValidity, ZipValidityIter};

use crate::bitmap::aligned::AlignedBitmapSlice;

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
/// This function panics iff `offset + len > 8 * slice.len()``.
pub fn count_zeros(slice: &[u8], offset: usize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }

    assert!(8 * slice.len() >= offset + len);

    // Fast-path: fits in a single u64 load.
    let first_byte_idx = offset / 8;
    let offset_in_byte = offset % 8;
    if offset_in_byte + len <= 64 {
        let mut word = load_padded_le_u64(&slice[first_byte_idx..]);
        word >>= offset_in_byte;
        word <<= 64 - len;
        return len - word.count_ones() as usize;
    }

    let aligned = AlignedBitmapSlice::<u64>::new(slice, offset, len);
    let ones_in_prefix = aligned.prefix().count_ones() as usize;
    let ones_in_bulk: usize = aligned.bulk_iter().map(|w| w.count_ones() as usize).sum();
    let ones_in_suffix = aligned.suffix().count_ones() as usize;
    len - ones_in_prefix - ones_in_bulk - ones_in_suffix
}
