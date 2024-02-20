use super::bitmask::BitMask;
use super::Bitmap;
use crate::trusted_len::TrustedLen;
use num_traits::Num;

use polars_utils::slice::load_padded_le_u64;

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
    shift: usize,
    remainder_bits: usize,
}

impl<'a> FastU32BitmapIter<'a> {
    pub fn new(bm: &'a Bitmap) -> Self {
        let (bytes, shift, len) = bm.as_slice();
        
        let fast_iter_count = bytes.len().saturating_sub(4) / 4;
        let remainder_bits = len - 32 * fast_iter_count;
        assert!(remainder_bits < 64);
        
        Self { bytes, shift, remainder_bits }
    }
}

impl<'a> Iterator for FastU32BitmapIter<'a> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Fast path, can load a whole u64.
        // FIXME: not correct when bytes.len() >= 8 but remainder < 64
        if let Some(next_chunk) = self.bytes.get(0..8) {
            let ret = u64::from_le_bytes(next_chunk.try_into().unwrap());
            self.bytes = &self.bytes[4..];
            return Some((ret >> self.shift) as u32);
        }
        
        // Slow path, remainder.
        if self.remainder_bits > 0 {
            let word = load_padded_le_u64(self.bytes);
            let ret = (word >> self.shift) & ((1 << self.remainder_bits) - 1);
            self.remainder_bits = self.remainder_bits.saturating_sub(32);
            return Some(ret as u32);
        }
        
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let fast_iters = self.bytes.len().saturating_sub(4) / 4;
        let remaining_iters = self.remainder_bits.div_ceil(32);
        let len = fast_iters + remaining_iters;
        (len, Some(len))
    }
}

unsafe impl<'a> TrustedLen for FastU32BitmapIter<'a> {}


pub struct FastU64BitmapIter<'a> {
    bytes: &'a [u8],
    shift: usize,
    len: usize,
}

impl<'a> FastU64BitmapIter<'a> {
    pub fn new(bm: &'a Bitmap) -> Self {
        let (bytes, shift, len) = bm.as_slice();
        Self { bytes, shift, len }
    }
}

impl<'a> Iterator for FastU64BitmapIter<'a> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len >= 65 {
            if let Some(next_chunk) = self.bytes.get(0..8) {
                let b = unsafe { *self.bytes.get_unchecked(8) };
                let ret = u64::from_le_bytes(next_chunk.try_into().unwrap());
                self.bytes = &self.bytes[8..];
                self.len -= 64;
                let merged = ((b as u128) << 64) | (ret as u128) >> self.shift % 64;
                return Some(merged as u64);
            } else {
                unsafe { std::hint::unreachable_unchecked() }
            }
        }
        
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = self.len.div_ceil(64);
        (hint, Some(hint))
    }
}

unsafe impl<'a> TrustedLen for FastU64BitmapIter<'a> {}

pub struct FastU56BitmapIter<'a> {
    bytes: &'a [u8],
    shift: usize,
    remainder_bits: usize,
}

impl<'a> FastU56BitmapIter<'a> {
    pub fn new(bm: &'a Bitmap) -> Self {
        let (bytes, shift, len) = bm.as_slice();
        
        // Precondition for fast iter: n * 7 + 8 <= b, where n is the number of
        // previous fast iterations and b = bytes.len(). Thus number of fast iters =
        //     smallest n such that n * 7 + 8 > b
        //     smallest n such that n * 7 + 8 >= 1 + b
        //     smallest n such that n * 7 >= b - 7
        //     smallest n such that n >= (b - 7) / 7
        //     n = ceil((b - 7) / 7) = floor((b - 7 + 6) / 7) = floor((b - 1) / 7)
        let fast_iter_count = bytes.len().saturating_sub(1) / 7;
        let remainder_bits = len - 56 * fast_iter_count;
        assert!(remainder_bits < 64);
        
        Self { bytes, shift, remainder_bits }
    }
}

impl<'a> Iterator for FastU56BitmapIter<'a> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Fast path, can load a whole u64.
        if let Some(next_chunk) = self.bytes.get(0..8) {
            let ret = u64::from_le_bytes(next_chunk.try_into().unwrap());
            self.bytes = &self.bytes[7..];
            return Some(ret >> self.shift);
        }
        
        // Slow path, remainder.
        if self.remainder_bits > 0 {
            let word = load_padded_le_u64(self.bytes);
            let ret = (word >> self.shift) & ((1 << self.remainder_bits) - 1);
            self.remainder_bits = self.remainder_bits.saturating_sub(56);
            return Some(ret);
        }
        
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let fast_iters = self.bytes.len().saturating_sub(4) / 4;
        let remaining_iters = self.remainder_bits.div_ceil(32);
        let len = fast_iters + remaining_iters;
        (len, Some(len))
    }
}

unsafe impl<'a> TrustedLen for FastU56BitmapIter<'a> {}

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
