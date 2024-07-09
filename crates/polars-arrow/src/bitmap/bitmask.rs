#[cfg(feature = "simd")]
use std::simd::{LaneCount, Mask, MaskElement, SupportedLaneCount};

use polars_utils::slice::load_padded_le_u64;

use crate::bitmap::Bitmap;

/// Returns the nth set bit in w, if n+1 bits are set. The indexing is
/// zero-based, nth_set_bit_u32(w, 0) returns the least significant set bit in w.
fn nth_set_bit_u32(w: u32, n: u32) -> Option<u32> {
    // If we have BMI2's PDEP available, we use it. It takes the lower order
    // bits of the first argument and spreads it along its second argument
    // where those bits are 1. So PDEP(abcdefgh, 11001001) becomes ef00g00h.
    // We use this by setting the first argument to 1 << n, which means the
    // first n-1 zero bits of it will spread to the first n-1 one bits of w,
    // after which the one bit will exactly get copied to the nth one bit of w.
    #[cfg(target_feature = "bmi2")]
    {
        if n >= 32 {
            return None;
        }

        let nth_set_bit = unsafe { core::arch::x86_64::_pdep_u32(1 << n, w) };
        if nth_set_bit == 0 {
            return None;
        }

        Some(nth_set_bit.trailing_zeros())
    }

    #[cfg(not(target_feature = "bmi2"))]
    {
        // Each block of 2/4/8/16 bits contains how many set bits there are in that block.
        let set_per_2 = w - ((w >> 1) & 0x55555555);
        let set_per_4 = (set_per_2 & 0x33333333) + ((set_per_2 >> 2) & 0x33333333);
        let set_per_8 = (set_per_4 + (set_per_4 >> 4)) & 0x0f0f0f0f;
        let set_per_16 = (set_per_8 + (set_per_8 >> 8)) & 0x00ff00ff;
        let set_per_32 = (set_per_16 + (set_per_16 >> 16)) & 0xff;
        if n >= set_per_32 {
            return None;
        }

        let mut idx = 0;
        let mut n = n;
        let next16 = set_per_16 & 0xff;
        if n >= next16 {
            n -= next16;
            idx += 16;
        }
        let next8 = (set_per_8 >> idx) & 0xff;
        if n >= next8 {
            n -= next8;
            idx += 8;
        }
        let next4 = (set_per_4 >> idx) & 0b1111;
        if n >= next4 {
            n -= next4;
            idx += 4;
        }
        let next2 = (set_per_2 >> idx) & 0b11;
        if n >= next2 {
            n -= next2;
            idx += 2;
        }
        let next1 = (w >> idx) & 0b1;
        if n >= next1 {
            idx += 1;
        }
        Some(idx)
    }
}

#[derive(Default, Clone)]
pub struct BitMask<'a> {
    bytes: &'a [u8],
    offset: usize,
    len: usize,
}

impl<'a> BitMask<'a> {
    pub fn from_bitmap(bitmap: &'a Bitmap) -> Self {
        let (bytes, offset, len) = bitmap.as_slice();
        // Check length so we can use unsafe access in our get.
        assert!(bytes.len() * 8 >= len + offset);
        Self { bytes, offset, len }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn split_at(&self, idx: usize) -> (Self, Self) {
        assert!(idx <= self.len);
        unsafe { self.split_at_unchecked(idx) }
    }

    /// # Safety
    /// The index must be in-bounds.
    #[inline]
    pub unsafe fn split_at_unchecked(&self, idx: usize) -> (Self, Self) {
        debug_assert!(idx <= self.len);
        let left = Self { len: idx, ..*self };
        let right = Self {
            len: self.len - idx,
            offset: self.offset + idx,
            ..*self
        };
        (left, right)
    }

    #[cfg(feature = "simd")]
    #[inline]
    pub fn get_simd<T, const N: usize>(&self, idx: usize) -> Mask<T, N>
    where
        T: MaskElement,
        LaneCount<N>: SupportedLaneCount,
    {
        // We don't support 64-lane masks because then we couldn't load our
        // bitwise mask as a u64 and then do the byteshift on it.

        let lanes = LaneCount::<N>::BITMASK_LEN;
        assert!(lanes < 64);

        let start_byte_idx = (self.offset + idx) / 8;
        let byte_shift = (self.offset + idx) % 8;
        if idx + lanes <= self.len {
            // SAFETY: fast path, we know this is completely in-bounds.
            let mask = load_padded_le_u64(unsafe { self.bytes.get_unchecked(start_byte_idx..) });
            Mask::from_bitmask(mask >> byte_shift)
        } else if idx < self.len {
            // SAFETY: we know that at least the first byte is in-bounds.
            // This is partially out of bounds, we have to do extra masking.
            let mask = load_padded_le_u64(unsafe { self.bytes.get_unchecked(start_byte_idx..) });
            let num_out_of_bounds = idx + lanes - self.len;
            let shifted = (mask << num_out_of_bounds) >> (num_out_of_bounds + byte_shift);
            Mask::from_bitmask(shifted)
        } else {
            Mask::from_bitmask(0u64)
        }
    }

    #[inline]
    pub fn get_u32(&self, idx: usize) -> u32 {
        let start_byte_idx = (self.offset + idx) / 8;
        let byte_shift = (self.offset + idx) % 8;
        if idx + 32 <= self.len {
            // SAFETY: fast path, we know this is completely in-bounds.
            let mask = load_padded_le_u64(unsafe { self.bytes.get_unchecked(start_byte_idx..) });
            (mask >> byte_shift) as u32
        } else if idx < self.len {
            // SAFETY: we know that at least the first byte is in-bounds.
            // This is partially out of bounds, we have to do extra masking.
            let mask = load_padded_le_u64(unsafe { self.bytes.get_unchecked(start_byte_idx..) });
            let out_of_bounds_mask = (1u32 << (self.len - idx)) - 1;
            ((mask >> byte_shift) as u32) & out_of_bounds_mask
        } else {
            0
        }
    }

    /// Computes the index of the nth set bit after start.
    ///
    /// Both are zero-indexed, so nth_set_bit_idx(0, 0) finds the index of the
    /// first bit set (which can be 0 as well). The returned index is absolute,
    /// not relative to start.
    pub fn nth_set_bit_idx(&self, mut n: usize, mut start: usize) -> Option<usize> {
        while start < self.len {
            let next_u32_mask = self.get_u32(start);
            if next_u32_mask == u32::MAX {
                // Happy fast path for dense non-null section.
                if n < 32 {
                    return Some(start + n);
                }
                n -= 32;
            } else {
                let ones = next_u32_mask.count_ones() as usize;
                if n < ones {
                    let idx = unsafe {
                        // SAFETY: we know the nth bit is in the mask.
                        nth_set_bit_u32(next_u32_mask, n as u32).unwrap_unchecked() as usize
                    };
                    return Some(start + idx);
                }
                n -= ones;
            }

            start += 32;
        }

        None
    }

    /// Computes the index of the nth set bit before end, counting backwards.
    ///
    /// Both are zero-indexed, so nth_set_bit_idx_rev(0, len) finds the index of
    /// the last bit set (which can be 0 as well). The returned index is
    /// absolute (and starts at the beginning), not relative to end.
    pub fn nth_set_bit_idx_rev(&self, mut n: usize, mut end: usize) -> Option<usize> {
        while end > 0 {
            // We want to find bits *before* end, so if end < 32 we must mask
            // out the bits after the endth.
            let (u32_mask_start, u32_mask_mask) = if end >= 32 {
                (end - 32, u32::MAX)
            } else {
                (0, (1 << end) - 1)
            };
            let next_u32_mask = self.get_u32(u32_mask_start) & u32_mask_mask;
            if next_u32_mask == u32::MAX {
                // Happy fast path for dense non-null section.
                if n < 32 {
                    return Some(end - 1 - n);
                }
                n -= 32;
            } else {
                let ones = next_u32_mask.count_ones() as usize;
                if n < ones {
                    let rev_n = ones - 1 - n;
                    let idx = unsafe {
                        // SAFETY: we know the rev_nth bit is in the mask.
                        nth_set_bit_u32(next_u32_mask, rev_n as u32).unwrap_unchecked() as usize
                    };
                    return Some(u32_mask_start + idx);
                }
                n -= ones;
            }

            end = u32_mask_start;
        }

        None
    }

    #[inline]
    pub fn get(&self, idx: usize) -> bool {
        let byte_idx = (self.offset + idx) / 8;
        let byte_shift = (self.offset + idx) % 8;

        if idx < self.len {
            // SAFETY: we know this is in-bounds.
            let byte = unsafe { *self.bytes.get_unchecked(byte_idx) };
            (byte >> byte_shift) & 1 == 1
        } else {
            false
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn naive_nth_bit_set(mut w: u32, mut n: u32) -> Option<u32> {
        for i in 0..32 {
            if w & (1 << i) != 0 {
                if n == 0 {
                    return Some(i);
                }
                n -= 1;
                w ^= 1 << i;
            }
        }
        None
    }

    #[test]
    fn test_nth_set_bit_u32() {
        for n in 0..256 {
            assert_eq!(nth_set_bit_u32(0, n), None);
        }

        for i in 0..32 {
            assert_eq!(nth_set_bit_u32(1 << i, 0), Some(i));
            assert_eq!(nth_set_bit_u32(1 << i, 1), None);
        }

        for i in 0..10000 {
            let rnd = (0xbdbc9d8ec9d5c461u64.wrapping_mul(i as u64) >> 32) as u32;
            for i in 0..=32 {
                assert_eq!(nth_set_bit_u32(rnd, i), naive_nth_bit_set(rnd, i));
            }
        }
    }
}
