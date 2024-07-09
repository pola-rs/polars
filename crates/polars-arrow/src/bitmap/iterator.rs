use polars_utils::slice::load_padded_le_u64;

use super::bitmask::BitMask;
use super::Bitmap;
use crate::trusted_len::TrustedLen;

/// Calculates how many iterations are remaining, assuming:
///  - We have length elements left.
///  - We need max(consume, min_length_for_iter) elements to start a new iteration.
///  - On each iteration we consume the given amount of elements.
fn calc_iters_remaining(length: usize, min_length_for_iter: usize, consume: usize) -> usize {
    let min_length_for_iter = min_length_for_iter.max(consume);
    if length < min_length_for_iter {
        return 0;
    }

    let obvious_part = length - min_length_for_iter;
    let obvious_iters = obvious_part / consume;
    // let obvious_part_remaining = obvious_part % consume;
    // let total_remaining = min_length_for_iter + obvious_part_remaining;
    // assert!(total_remaining >= min_length_for_iter);          // We have at least 1 more iter.
    // assert!(obvious_part_remaining < consume);                // Basic modulo property.
    // assert!(total_remaining < min_length_for_iter + consume); // Add min_length_for_iter to both sides.
    // assert!(total_remaining - consume < min_length_for_iter); // Not enough remaining after 1 iter.
    1 + obvious_iters // Thus always exactly 1 more iter.
}

pub struct TrueIdxIter<'a> {
    mask: BitMask<'a>,
    first_unknown: usize,
    i: usize,
    len: usize,
    remaining: usize,
}

impl<'a> TrueIdxIter<'a> {
    #[inline]
    pub fn new(len: usize, validity: Option<&'a Bitmap>) -> Self {
        if let Some(bitmap) = validity {
            assert!(len == bitmap.len());
            Self {
                mask: BitMask::from_bitmap(bitmap),
                first_unknown: 0,
                i: 0,
                remaining: bitmap.len() - bitmap.unset_bits(),
                len,
            }
        } else {
            Self {
                mask: BitMask::default(),
                first_unknown: len,
                i: 0,
                remaining: len,
                len,
            }
        }
    }
}

impl<'a> Iterator for TrueIdxIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Fast path for many non-nulls in a row.
        if self.i < self.first_unknown {
            let ret = self.i;
            self.i += 1;
            self.remaining -= 1;
            return Some(ret);
        }

        while self.i < self.len {
            let mask = self.mask.get_u32(self.i);
            let num_null = mask.trailing_zeros();
            self.i += num_null as usize;
            if num_null < 32 {
                self.first_unknown = self.i + (mask >> num_null).trailing_ones() as usize;
                let ret = self.i;
                self.i += 1;
                self.remaining -= 1;
                return Some(ret);
            }
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

unsafe impl<'a> TrustedLen for TrueIdxIter<'a> {}

pub struct FastU32BitmapIter<'a> {
    bytes: &'a [u8],
    shift: u32,
    bits_left: usize,
}

impl<'a> FastU32BitmapIter<'a> {
    pub fn new(bytes: &'a [u8], offset: usize, len: usize) -> Self {
        assert!(bytes.len() * 8 >= offset + len);
        let shift = (offset % 8) as u32;
        let bytes = &bytes[offset / 8..];
        Self {
            bytes,
            shift,
            bits_left: len,
        }
    }

    // The iteration logic that would normally follow the fast-path.
    fn next_remainder(&mut self) -> Option<u32> {
        if self.bits_left > 0 {
            let word = load_padded_le_u64(self.bytes);
            let mask;
            if self.bits_left >= 32 {
                mask = u32::MAX;
                self.bits_left -= 32;
                self.bytes = unsafe { self.bytes.get_unchecked(4..) };
            } else {
                mask = (1 << self.bits_left) - 1;
                self.bits_left = 0;
            }

            return Some((word >> self.shift) as u32 & mask);
        }

        None
    }

    /// Returns the remainder bits and how many there are,
    /// assuming the iterator was fully consumed.
    pub fn remainder(mut self) -> (u64, usize) {
        let bits_left = self.bits_left;
        let lo = self.next_remainder().unwrap_or(0);
        let hi = self.next_remainder().unwrap_or(0);
        (((hi as u64) << 32) | (lo as u64), bits_left)
    }
}

impl<'a> Iterator for FastU32BitmapIter<'a> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Fast path, can load a whole u64.
        if self.bits_left >= 64 {
            let chunk;
            unsafe {
                // SAFETY: bits_left ensures this is in-bounds.
                chunk = self.bytes.get_unchecked(0..8);
                self.bytes = self.bytes.get_unchecked(4..);
            }
            self.bits_left -= 32;
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            return Some((word >> self.shift) as u32);
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = calc_iters_remaining(self.bits_left, 64, 32);
        (hint, Some(hint))
    }
}

unsafe impl<'a> TrustedLen for FastU32BitmapIter<'a> {}

pub struct FastU56BitmapIter<'a> {
    bytes: &'a [u8],
    shift: u32,
    bits_left: usize,
}

impl<'a> FastU56BitmapIter<'a> {
    pub fn new(bytes: &'a [u8], offset: usize, len: usize) -> Self {
        assert!(bytes.len() * 8 >= offset + len);
        let shift = (offset % 8) as u32;
        let bytes = &bytes[offset / 8..];
        Self {
            bytes,
            shift,
            bits_left: len,
        }
    }

    // The iteration logic that would normally follow the fast-path.
    fn next_remainder(&mut self) -> Option<u64> {
        if self.bits_left > 0 {
            let word = load_padded_le_u64(self.bytes);
            let mask;
            if self.bits_left >= 56 {
                mask = (1 << 56) - 1;
                self.bits_left -= 56;
                self.bytes = unsafe { self.bytes.get_unchecked(7..) };
            } else {
                mask = (1 << self.bits_left) - 1;
                self.bits_left = 0;
            };

            return Some((word >> self.shift) & mask);
        }

        None
    }

    /// Returns the remainder bits and how many there are,
    /// assuming the iterator was fully consumed. Output is safe but
    /// not specified if the iterator wasn't fully consumed.
    pub fn remainder(mut self) -> (u64, usize) {
        let bits_left = self.bits_left;
        let lo = self.next_remainder().unwrap_or(0);
        let hi = self.next_remainder().unwrap_or(0);
        ((hi << 56) | lo, bits_left)
    }
}

impl<'a> Iterator for FastU56BitmapIter<'a> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Fast path, can load a whole u64.
        if self.bits_left >= 64 {
            let chunk;
            unsafe {
                // SAFETY: bits_left ensures this is in-bounds.
                chunk = self.bytes.get_unchecked(0..8);
                self.bytes = self.bytes.get_unchecked(7..);
                self.bits_left -= 56;
            }

            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            let mask = (1 << 56) - 1;
            return Some((word >> self.shift) & mask);
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = calc_iters_remaining(self.bits_left, 64, 56);
        (hint, Some(hint))
    }
}

unsafe impl<'a> TrustedLen for FastU56BitmapIter<'a> {}

pub struct FastU64BitmapIter<'a> {
    bytes: &'a [u8],
    shift: u32,
    bits_left: usize,
    next_word: u64,
}

impl<'a> FastU64BitmapIter<'a> {
    pub fn new(bytes: &'a [u8], offset: usize, len: usize) -> Self {
        assert!(bytes.len() * 8 >= offset + len);
        let shift = (offset % 8) as u32;
        let bytes = &bytes[offset / 8..];
        let next_word = load_padded_le_u64(bytes);
        let bytes = bytes.get(8..).unwrap_or(&[]);
        Self {
            bytes,
            shift,
            bits_left: len,
            next_word,
        }
    }

    #[inline]
    fn combine(&self, lo: u64, hi: u64) -> u64 {
        // Compiles to 128-bit SHRD instruction on x86-64.
        // Yes, the % 64 is important for the compiler to generate optimal code.
        let wide = ((hi as u128) << 64) | lo as u128;
        (wide >> (self.shift % 64)) as u64
    }

    // The iteration logic that would normally follow the fast-path.
    fn next_remainder(&mut self) -> Option<u64> {
        if self.bits_left > 0 {
            let lo = self.next_word;
            let hi = load_padded_le_u64(self.bytes);
            let mask;
            if self.bits_left >= 64 {
                mask = u64::MAX;
                self.bits_left -= 64;
                self.bytes = self.bytes.get(8..).unwrap_or(&[]);
            } else {
                mask = (1 << self.bits_left) - 1;
                self.bits_left = 0;
            };
            self.next_word = hi;

            return Some(self.combine(lo, hi) & mask);
        }

        None
    }

    /// Returns the remainder bits and how many there are,
    /// assuming the iterator was fully consumed. Output is safe but
    /// not specified if the iterator wasn't fully consumed.
    pub fn remainder(mut self) -> ([u64; 2], usize) {
        let bits_left = self.bits_left;
        let lo = self.next_remainder().unwrap_or(0);
        let hi = self.next_remainder().unwrap_or(0);
        ([lo, hi], bits_left)
    }
}

impl<'a> Iterator for FastU64BitmapIter<'a> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Fast path: can load two u64s in a row.
        // (Note that we already loaded one in the form of self.next_word).
        if self.bits_left >= 128 {
            let chunk;
            unsafe {
                // SAFETY: bits_left ensures this is in-bounds.
                chunk = self.bytes.get_unchecked(0..8);
                self.bytes = self.bytes.get_unchecked(8..);
            }
            let lo = self.next_word;
            let hi = u64::from_le_bytes(chunk.try_into().unwrap());
            self.next_word = hi;
            self.bits_left -= 64;

            return Some(self.combine(lo, hi));
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = calc_iters_remaining(self.bits_left, 128, 64);
        (hint, Some(hint))
    }
}

unsafe impl<'a> TrustedLen for FastU64BitmapIter<'a> {}

/// This crates' equivalent of [`std::vec::IntoIter`] for [`Bitmap`].
#[derive(Debug, Clone)]
pub struct IntoIter {
    values: Bitmap,
    index: usize,
    end: usize,
}

impl IntoIter {
    /// Creates a new [`IntoIter`] from a [`Bitmap`]
    #[inline]
    pub fn new(values: Bitmap) -> Self {
        let end = values.len();
        Self {
            values,
            index: 0,
            end,
        }
    }
}

impl Iterator for IntoIter {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            return None;
        }
        let old = self.index;
        self.index += 1;
        Some(unsafe { self.values.get_bit_unchecked(old) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.index, Some(self.end - self.index))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let new_index = self.index + n;
        if new_index > self.end {
            self.index = self.end;
            None
        } else {
            self.index = new_index;
            self.next()
        }
    }
}

impl DoubleEndedIterator for IntoIter {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            None
        } else {
            self.end -= 1;
            Some(unsafe { self.values.get_bit_unchecked(self.end) })
        }
    }
}

unsafe impl TrustedLen for IntoIter {}
