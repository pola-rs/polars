#[cfg(feature = "simd")]
use std::simd::ToBitMask;

#[cfg(feature = "simd")]
use num_traits::AsPrimitive;

use crate::bitmap::Bitmap;

// Loads a u64 from the given byteslice, as if it were padded with zeros.
fn load_padded_le_u64(bytes: &[u8]) -> u64 {
    let len = bytes.len();
    if len >= 8 {
        return u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    }

    if len >= 4 {
        let lo = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let hi = u32::from_le_bytes(bytes[len - 4..len].try_into().unwrap());
        return (lo as u64) | ((hi as u64) << (8 * (len - 4)));
    }

    if len == 0 {
        return 0;
    }

    let lo = bytes[0] as u64;
    let mid = (bytes[len / 2] as u64) << (8 * (len / 2));
    let hi = (bytes[len - 1] as u64) << (8 * (len - 1));
    lo | mid | hi
}

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
    pub fn get_simd<T>(&self, idx: usize) -> T
    where
        T: ToBitMask,
        <T as ToBitMask>::BitMask: Copy + 'static,
        u64: AsPrimitive<<T as ToBitMask>::BitMask>,
    {
        // We don't support 64-lane masks because then we couldn't load our
        // bitwise mask as a u64 and then do the byteshift on it.

        let lanes = std::mem::size_of::<T::BitMask>() * 8;
        assert!(lanes < 64);

        let start_byte_idx = (self.offset + idx) / 8;
        let byte_shift = (self.offset + idx) % 8;
        if idx + lanes <= self.len {
            // SAFETY: fast path, we know this is completely in-bounds.
            let mask = load_padded_le_u64(unsafe { self.bytes.get_unchecked(start_byte_idx..) });
            T::from_bitmask((mask >> byte_shift).as_())
        } else if idx < self.len {
            // SAFETY: we know that at least the first byte is in-bounds.
            // This is partially out of bounds, we have to do extra masking.
            let mask = load_padded_le_u64(unsafe { self.bytes.get_unchecked(start_byte_idx..) });
            let num_out_of_bounds = idx + lanes - self.len;
            let shifted = (mask << num_out_of_bounds) >> (num_out_of_bounds + byte_shift);
            T::from_bitmask(shifted.as_())
        } else {
            T::from_bitmask((0u64).as_())
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
            let num_out_of_bounds = idx + 32 - self.len;
            let shifted = (mask << num_out_of_bounds) >> (num_out_of_bounds + byte_shift);
            shifted as u32
        } else {
            0
        }
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
