/// Forked from Arrow until their API stabilizes.
///
/// Note that the bound checks are optimized away.
///
use crate::bitmap::utils::{BitChunkIterExact, BitChunks, BitChunksExact};
use crate::bitmap::Bitmap;
const BIT_MASK: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Returns the nearest number that is `>=` than `num` and is a multiple of 64
#[inline]
pub fn round_upto_multiple_of_64(num: usize) -> usize {
    round_upto_power_of_2(num, 64)
}

/// Returns the nearest multiple of `factor` that is `>=` than `num`. Here `factor` must
/// be a power of 2.
pub fn round_upto_power_of_2(num: usize, factor: usize) -> usize {
    debug_assert!(factor > 0 && (factor & (factor - 1)) == 0);
    (num + (factor - 1)) & !(factor - 1)
}

/// Returns whether bit at position `i` in `data` is set or not
#[inline]
pub fn get_bit(data: &[u8], i: usize) -> bool {
    (data[i >> 3] & BIT_MASK[i & 7]) != 0
}

/// Returns whether bit at position `i` in `data` is set or not.
///
/// # Safety
///
/// Note this doesn't do any bound checking, for performance reason. The caller is
/// responsible to guarantee that `i` is within bounds.
#[inline]
pub unsafe fn get_bit_raw(data: *const u8, i: usize) -> bool {
    (*data.add(i >> 3) & BIT_MASK[i & 7]) != 0
}

/// Sets bit at position `i` for `data`
#[inline]
pub fn set_bit(data: &mut [u8], i: usize) {
    data[i >> 3] |= BIT_MASK[i & 7];
}

/// Sets bit at position `i` for `data`
///
/// # Safety
///
/// Note this doesn't do any bound checking, for performance reason. The caller is
/// responsible to guarantee that `i` is within bounds.
#[inline]
pub unsafe fn set_bit_raw(data: *mut u8, i: usize) {
    *data.add(i >> 3) |= BIT_MASK[i & 7];
}

/// Sets bit at position `i` for `data` to 0
#[inline]
pub fn unset_bit(data: &mut [u8], i: usize) {
    data[i >> 3] ^= BIT_MASK[i & 7];
}

/// Sets bit at position `i` for `data` to 0
///
/// # Safety
///
/// Note this doesn't do any bound checking, for performance reason. The caller is
/// responsible to guarantee that `i` is within bounds.
#[inline]
pub unsafe fn unset_bit_raw(data: *mut u8, i: usize) {
    *data.add(i >> 3) ^= BIT_MASK[i & 7];
}

/// Returns the ceil of `value`/`divisor`
#[inline]
pub fn ceil(value: usize, divisor: usize) -> usize {
    let (quot, rem) = (value / divisor, value % divisor);
    if rem > 0 && divisor > 0 {
        quot + 1
    } else {
        quot
    }
}

fn first_set_bit_impl<I>(mut mask_chunks: I) -> usize
where
    I: BitChunkIterExact<u64>,
{
    let mut total = 0usize;
    const SIZE: u32 = 64;
    for chunk in &mut mask_chunks {
        let pos = chunk.trailing_zeros();
        if pos != SIZE {
            return total + pos as usize;
        } else {
            total += SIZE as usize
        }
    }
    if let Some(pos) = mask_chunks.remainder_iter().position(|v| v) {
        total += pos;
        return total;
    }
    // all null, return the first
    0
}

pub fn first_set_bit(mask: &Bitmap) -> usize {
    if mask.unset_bits() == 0 || mask.unset_bits() == mask.len() {
        return 0;
    }
    let (slice, offset, length) = mask.as_slice();
    if offset == 0 {
        let mask_chunks = BitChunksExact::<u64>::new(slice, length);
        first_set_bit_impl(mask_chunks)
    } else {
        let mask_chunks = mask.chunks::<u64>();
        first_set_bit_impl(mask_chunks)
    }
}

fn first_unset_bit_impl<I>(mut mask_chunks: I) -> usize
where
    I: BitChunkIterExact<u64>,
{
    let mut total = 0usize;
    const SIZE: u32 = 64;
    for chunk in &mut mask_chunks {
        let pos = chunk.trailing_ones();
        if pos != SIZE {
            return total + pos as usize;
        } else {
            total += SIZE as usize
        }
    }
    if let Some(pos) = mask_chunks.remainder_iter().position(|v| !v) {
        total += pos;
        return total;
    }
    // all null, return the first
    0
}

pub fn first_unset_bit(mask: &Bitmap) -> usize {
    if mask.unset_bits() == 0 || mask.unset_bits() == mask.len() {
        return 0;
    }
    let (slice, offset, length) = mask.as_slice();
    if offset == 0 {
        let mask_chunks = BitChunksExact::<u64>::new(slice, length);
        first_unset_bit_impl(mask_chunks)
    } else {
        let mask_chunks = mask.chunks::<u64>();
        first_unset_bit_impl(mask_chunks)
    }
}

pub fn find_first_true_false_null(
    mut bit_chunks: BitChunks<u64>,
    mut validity_chunks: BitChunks<u64>,
) -> (Option<usize>, Option<usize>, Option<usize>) {
    let (mut true_index, mut false_index, mut null_index) = (None, None, None);
    let (mut true_not_found_mask, mut false_not_found_mask, mut null_not_found_mask) =
        (!0u64, !0u64, !0u64); // All ones while not found.
    let mut offset: usize = 0;
    let mut all_found = false;
    for (truth_mask, null_mask) in (&mut bit_chunks).zip(&mut validity_chunks) {
        let mask = null_mask & truth_mask & true_not_found_mask;
        if mask > 0 {
            true_index = Some(offset + mask.trailing_zeros() as usize);
            true_not_found_mask = 0;
        }
        let mask = null_mask & !truth_mask & false_not_found_mask;
        if mask > 0 {
            false_index = Some(offset + mask.trailing_zeros() as usize);
            false_not_found_mask = 0;
        }
        if !null_mask & null_not_found_mask > 0 {
            null_index = Some(offset + null_mask.trailing_ones() as usize);
            null_not_found_mask = 0;
        }
        if null_not_found_mask | true_not_found_mask | false_not_found_mask == 0 {
            all_found = true;
            break;
        }
        offset += 64;
    }
    if !all_found {
        for (val, not_null) in bit_chunks
            .remainder_iter()
            .zip(validity_chunks.remainder_iter())
        {
            if true_index.is_none() && not_null && val {
                true_index = Some(offset);
            } else if false_index.is_none() && not_null && !val {
                false_index = Some(offset);
            } else if null_index.is_none() && !not_null {
                null_index = Some(offset);
            }
            offset += 1;
        }
    }
    (true_index, false_index, null_index)
}

pub fn find_first_true_false_no_null(
    mut bit_chunks: BitChunks<u64>,
) -> (Option<usize>, Option<usize>) {
    let (mut true_index, mut false_index) = (None, None);
    let (mut true_not_found_mask, mut false_not_found_mask) = (!0u64, !0u64); // All ones while not found.
    let mut offset: usize = 0;
    let mut all_found = false;
    for truth_mask in &mut bit_chunks {
        let mask = truth_mask & true_not_found_mask;
        if mask > 0 {
            true_index = Some(offset + mask.trailing_zeros() as usize);
            true_not_found_mask = 0;
        }
        let mask = !truth_mask & false_not_found_mask;
        if mask > 0 {
            false_index = Some(offset + mask.trailing_zeros() as usize);
            false_not_found_mask = 0;
        }
        if true_not_found_mask | false_not_found_mask == 0 {
            all_found = true;
            break;
        }
        offset += 64;
    }
    if !all_found {
        for val in bit_chunks.remainder_iter() {
            if true_index.is_none() && val {
                true_index = Some(offset);
            } else if false_index.is_none() && !val {
                false_index = Some(offset);
            }
            offset += 1;
        }
    }
    (true_index, false_index)
}
