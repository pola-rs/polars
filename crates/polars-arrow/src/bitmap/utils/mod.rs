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
pub use slice_iterator::SlicesIterator;
pub use zip_validity::{ZipValidity, ZipValidityIter};

const BIT_MASK: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
const UNSET_BIT_MASK: [u8; 8] = [
    255 - 1,
    255 - 2,
    255 - 4,
    255 - 8,
    255 - 16,
    255 - 32,
    255 - 64,
    255 - 128,
];

/// Returns whether bit at position `i` in `byte` is set or not
#[inline]
pub fn is_set(byte: u8, i: usize) -> bool {
    (byte & BIT_MASK[i]) != 0
}

/// Sets bit at position `i` in `byte`
#[inline]
pub fn set(byte: u8, i: usize, value: bool) -> u8 {
    if value {
        byte | BIT_MASK[i]
    } else {
        byte & UNSET_BIT_MASK[i]
    }
}

/// Sets bit at position `i` in `data`
/// # Panics
/// panics if `i >= data.len() / 8`
#[inline]
pub fn set_bit(data: &mut [u8], i: usize, value: bool) {
    data[i / 8] = set(data[i / 8], i % 8, value);
}

/// Sets bit at position `i` in `data` without doing bound checks
/// # Safety
/// caller must ensure that `i < data.len() / 8`
#[inline]
pub unsafe fn set_bit_unchecked(data: &mut [u8], i: usize, value: bool) {
    let byte = data.get_unchecked_mut(i / 8);
    *byte = set(*byte, i % 8, value);
}

/// Returns whether bit at position `i` in `data` is set
/// # Panic
/// This function panics iff `i / 8 >= bytes.len()`
#[inline]
pub fn get_bit(bytes: &[u8], i: usize) -> bool {
    is_set(bytes[i / 8], i % 8)
}

/// Returns whether bit at position `i` in `data` is set or not.
///
/// # Safety
/// `i >= data.len() * 8` results in undefined behavior
#[inline]
pub unsafe fn get_bit_unchecked(data: &[u8], i: usize) -> bool {
    (*data.as_ptr().add(i >> 3) & BIT_MASK[i & 7]) != 0
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
